
import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
# from human_eval.data import read_problems, stream_jsonl
from vllm import LLM
from vllm import SamplingParams
import json

from utils import extract_predicted_answer, extract_ground_truth, extract_answer
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:"
    )
SHOT="""
# Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
# A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

# Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
# A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

# Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
# A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

# Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
# A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

# Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
# A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

# Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

# Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

# Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

# Q: """
def read_from_json(dataset):
    datas = json.load(open(f"./datas/{dataset}.json"))
    datas = {i:data for i, data in enumerate(datas)}
    return datas


def construct_prompt(dataset,tokenizer=None):
    problems = read_from_json(dataset)
    if dataset == "math" or dataset == "math-500":
        prompts = [problems[task_id]['instruction'].strip() for task_id in problems]
    else:
        prompts = [problems[task_id]['instruction'].strip() for task_id in problems]

    # prompt_batch = [PROMPT_DICT['prompt_no_input'].format_map(dict(instruction=prompt.replace('    ', '\t'))) for prompt in prompts]
    prompt_batch = [prompt + "\n\n" for prompt in prompts]
    
    # prompt_batch = [SHOT + prompt.strip() + '\n# A:' for prompt in prompts]
    if tokenizer is not None:
        prompt_batch = [tokenizer.apply_chat_template([
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."}, 
            {"role": "user", "content": prompt }
            ], tokenize=False, add_generation_prompt=True) for prompt in prompt_batch]
    print(prompt_batch[0])
    return problems, prompt_batch

def score_completion(dataset, problems, answers):
    import re
    if dataset == "gsm8k":
        true_results = [extract_ground_truth(problems[task_id]['output']) for task_id in problems]
        our_results = [extract_answer(answer) for answer in answers]
        # our_results = [re.findall("\\boxed\{(.+)\}", answer)[0].strip() for answer in answers]
        scores = [t==o for t, o in zip(true_results, our_results)]
        return sum(scores)/len(scores)
    if dataset == "math" or dataset == "math-500":
        true_results = [extract_answer(problems[task_id]['output']) for task_id in problems]
        our_results = [extract_answer(answer) for answer in answers]
        # our_results = [re.findall("\\boxed\{(.+)\}", answer)[0].strip() for answer in answers]
        scores = [t==o for t, o in zip(true_results, our_results)]
        return sum(scores)/len(scores)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--dataset', type=str, default='gsm8k', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--top_p', type=float, default=1, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--num_gpus', type=int, default=8, help="")
    
    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    tokenizer = None
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    problems, prompt_batch = construct_prompt(args.dataset)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=8192)
    stop = ["</s>", "\n\n\n\n", "<|EOT|>", "# Q"]
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_len, stop=stop)
    with torch.no_grad():
        completions = llm.generate(prompt_batch, sampling_params)
    answers = []
    records = []
    for i, completion in enumerate(completions):
        gen_seq = completion.outputs[0].text.strip()
        answers.append(gen_seq)
        records.append({
            "instruction": problems[i]['instruction'],
            "output": gen_seq
        })
    # print(records)
    json.dump(records, open(args.output_path + '/results.json', 'w'))
    score = score_completion(args.dataset, problems, answers)
    print(f"Score: {score}")
        
if __name__ == "__main__":            
    main()
