import json
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict
import os
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from prompt import generate_tags_prompt, generate_reduce_prompt, generate_instruction_prompt

class TagReduce:
    def __init__(self, model_name_or_path: str, tensor_parallel_size: int = 8, dtype: str = "bfloat16") -> None:
        """
        Initializes the EvolInstruct object with model details and engine configuration.

        Args:
            model_name_or_path (str): Path to the model or model name.
            tensor_parallel_size (int): Number of tensor parallel workers (default: 8).
            dtype (str): The data type used for inference (default: "bfloat16").
        """
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.llm_engine = self.create_llm_engine(model_name_or_path)
        self.history_tags = defaultdict(int)
    def create_llm_engine(self, model_name_or_path: str) -> LLM:
        """Creates an instance of the LLM engine."""
        logger.info("Initializing LLM engine...")
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
        )

    def generate_tags(self, datas, temperature, max_tokens):
        import re
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        instructions = [data['instruction']+ '\n###Input:' + data['input']\
         if data['input'] != '' else data['instruction'] for data in datas]
        tag_inputs = [[{"role": "user", "content": generate_tags_prompt(inst)}] for inst in instructions]
        id2input = {i:tag_inputs[i] for i in range(len(tag_inputs))}
        use_ids = list(id2input.keys())
        id2datapoints = {}
        while len(use_ids) > 0:
            tag_inputs = [tokenizer.apply_chat_template(id2input[id], tokenize=False, add_generation_prompt=True) for id in use_ids]
            tag_outputs = self.llm_engine.generate(tag_inputs, sampling_params)
            temp_ids = []
            for i in range(len(use_ids)):
                output = tag_outputs[i].outputs[0].text.strip()
                try:
                    tags = re.findall("```(.+?)```", output, re.DOTALL)[0]
                    tags = tags.strip().strip("json").strip()
                    tags = json.loads(tags)
                    id2input[use_ids[i]].append({"role": "assistant", "content": output})
                    id2datapoints[use_ids[i]] = {
                        "tag_context":id2input[use_ids[i]],
                        "tags": tags,
                        "instruction": instructions[use_ids[i]]
                    }
                except Exception as e:
                    temp_ids.append(use_ids[i])
                    continue
            use_ids = temp_ids
        return id2datapoints

    def tags_reduce(self, data_points, temperature, max_tokens, budget, num_shot, reject_top_k):
        import random
        import re,math
        from collections import defaultdict
        random.seed(42)
        generate_data_points = defaultdict(dict)
        batch = 100
        budget_tags_ratio = 1/num_shot
        num_batch = math.ceil(budget/batch)
        for batch_id in range(num_batch):
            instructions = []
            for i in range(min([batch, budget-batch*batch_id])):
                idx = batch*batch_id + i
                shot_ids = random.sample(list(data_points.keys()), k=num_shot)
                shot_tags = [data_points[id]['tags'] for id in shot_ids]
                shot_instructions = [data_points[id]['instruction'] for id in shot_ids]
                generate_data_points[idx]['shot_instructions'] = shot_instructions
                generate_data_points[idx]['shot_tags'] = shot_tags
                generate_data_points[idx]['ids'] = shot_ids
                merge_tags = defaultdict(list)
                all_vs = []
                for tags in shot_tags:
                    for key in tags:
                        clean_key = key.lower().strip()
                        clean_tags = [str(v).lower().strip() for v in tags[key]]
                        merge_tags[clean_key] += clean_tags
                        all_vs += clean_tags
                        merge_tags[clean_key] = list(set(merge_tags[clean_key]))

                all_vs = sorted(list(set(all_vs)), key=lambda k: self.history_tags[k], reverse=True)
                if batch_id != 0:
                    rejected_v = all_vs[:reject_top_k]
                    for k in merge_tags:
                        for v in rejected_v:
                            if v in merge_tags[k]:
                                merge_tags[k].remove(v)
                # print(merge_tags)
                generate_data_points[idx]['merge_tags'] = merge_tags
                instructions.append(generate_reduce_prompt(merge_tags))

            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            inputs = [[{"role": "user", "content": inst}] for inst in instructions]
            inputs = [tokenizer.apply_chat_template(it, tokenize=False, add_generation_prompt=True) for it in inputs]
            id2inputs = {batch_id*batch+i: inputs[i] for i in range(len(inputs))}
            use_ids = [batch_id*batch+i for i in range(len(inputs))]
            while len(use_ids) > 0:
                now_inputs = [id2inputs[id] for id in use_ids]
                # outputs = self.llm_engine.generate(now_inputs, sampling_params)
                temp_ids = []
                for i in range(len(now_inputs)):
                    # output = outputs[i].outputs[0].text.strip()
                    try:
                        merge_tags = generate_data_points[use_ids[i]]['merge_tags']
                        sub_tags = defaultdict(list)
                        for k in merge_tags:
                            for v in merge_tags[k]:
                                if random.random() <= budget_tags_ratio:
                                    sub_tags[k].append(v)
                        output = sub_tags
                        for k in output:
                            for v in output[k]:
                                self.history_tags[v.lower().strip()] += 1
                        generate_data_points[use_ids[i]]['sub_tags'] = output
                    except Exception as e:
                        print(e)
                        temp_ids.append(use_ids[i])
                use_ids = temp_ids
        return generate_data_points


    def tags2instruction(self, generate_data_points, temperature, max_tokens):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        instructions = [generate_instruction_prompt(data_pint) for data_pint in generate_data_points.values()]
        inputs = [[{"role": "user", "content": inst}] for inst in instructions]
        inputs = [tokenizer.apply_chat_template(it, tokenize=False, add_generation_prompt=True) for it in inputs]
        outputs = self.llm_engine.generate(inputs, sampling_params)
        all_result = []
        for i in range(len(inputs)):
            output = outputs[i].outputs[0].text.strip()
            output = output.split("Instruction:")[-1].strip()
            result = {
                "instruction":output,
                "input":"",
                "output":"",
                "tags": list(generate_data_points.values())[i]['sub_tags']
            }
            all_result.append(result)
        return all_result

def get_ori_datas(source_file) -> List[str]:
    """
    Extract datas from the source file.

    Args:
        source_file (str): Path to the source JSON file.
        round_num (int): The evolutionary round number to determine prompt transformations.
        use_breadth (bool): Whether to include breadth-based prompt evolution.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the new instructions and inputs.
    """
    try:
        if source_file.endswith('jsonl'):
            with open(source_file, "r") as f:
                source_data = [json.loads(l) for l in f]
        else:
            with open(source_file, "r") as f:
                source_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Source file not found: {source_file}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the file: {source_file}")
        raise
    return source_data


def main(model_name_or_path: str, source_file: str,
         target_file: str,
         temperature: float, max_tokens: int,
         budget: int, num_shot: int,
         debug: bool, tp: int, reject_top_k: int) -> None:
    """
    Main function to generate tagreuce instructions and store the results in the target file.

    Args:
        model_name_or_path (str): Path to the model or model name.
        source_file (str): Path to the source JSON file containing the instructions.
        target_file (str): Path to the target JSON file to save the results.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum tokens to generate.
        debug (bool): Whether to enable debug mode (limits results to 100).
        tp (int): Number of tensor parallel workers.
    """
    ori_datas = get_ori_datas(source_file)

    if debug:
        ori_datas = ori_datas[:100]  # Limit to the first 100 instructions for debugging
        budget = 10
    engine = TagReduce(model_name_or_path, tensor_parallel_size=tp)
    if os.path.exists(f"{source_file}.tag_datas.json"):
        print(f"{source_file}.tag_datas.json exsits")
        data_points = json.load(open(f"{source_file}.tag_datas.json"))
    else:
        print(f"{source_file}.tag_datas.json not exsits")
        data_points = engine.generate_tags(ori_datas, temperature, max_tokens)
        json.dump(data_points, open(f"{source_file}.tag_datas.json", 'w'))
    generate_data_points = engine.tags_reduce(data_points, temperature, max_tokens, budget, num_shot, reject_top_k)
    all_results = engine.tags2instruction(generate_data_points, temperature, max_tokens)
    # Write results to the target file
    try:
        with open(target_file, "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        with open(f"{target_file}.history_tags.json", "w", encoding='utf-8') as f:
            json.dump(engine.history_tags, f, indent=4, ensure_ascii=False)
        logger.info(f"Results successfully saved to {target_file}")
    except Exception as e:
        logger.error(f"Error saving results to {target_file}: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TagReduce Instructions Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--source_file', type=str, required=True, help='Path to the source JSON file')
    parser.add_argument('--target_file', type=str, required=True, help='Path to the target JSON file')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens for generation')
    parser.add_argument('--budget', type=int, default=10, help='Maximum tokens for generation')
    parser.add_argument('--reject_top_k', type=int, default=10, help='Maximum tokens for generation')
    parser.add_argument('--num_shot', type=int, default=10, help='Maximum tokens for generation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--tp', type=int, default=8, help='Number of tensor parallel workers')

    args = parser.parse_args()

    main(args.model_name_or_path, args.source_file, args.target_file, args.temperature, args.max_tokens, args.budget, args.num_shot,args.debug, args.tp, args.reject_top_k)
