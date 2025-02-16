
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

from utils import extract_predicted_answer, extract_ground_truth
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:"
    )

def read_from_json(dataset):
    datas = json.load(open(f"./datas/{dataset}.json"))
    datas = {i:data for i, data in enumerate(datas)}
    return datas


def construct_prompt(dataset,tokenizer=None):
    problems = read_from_json(dataset)
    prompts = [problems[task_id]['instruction'].strip() for task_id in problems]

    # prompt_batch = [PROMPT_DICT['prompt_no_input'].format_map(dict(instruction=prompt.replace('    ', '\t'))) for prompt in prompts]
    prompt_batch = [PROMPT.format(prompt) for prompt in prompts]
    if tokenizer is not None:
        prompt_batch = [tokenizer.apply_chat_template([{"role": "user", "content": prompt }], tokenize=False, add_generation_prompt=True) for prompt in prompt_batch]
    print(prompt_batch[0])
    return problems, prompt_batch

def score_completion(dataset, problems, answers):
    import re
    if dataset == "gsm8k":
        true_results = [extract_ground_truth(problems[task_id]['output']) for task_id in problems]
        our_results = [extract_predicted_answer(answer) for answer in answers]
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
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    problems, prompt_batch = construct_prompt(args.dataset)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=8192)
    stop = ["</s>", "\n\n\n\n", "<|EOT|>"]
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
        
            
main()
