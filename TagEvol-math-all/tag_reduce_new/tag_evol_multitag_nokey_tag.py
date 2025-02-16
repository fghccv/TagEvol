import json
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict
import os
import tqdm
import random
from ordered_set import OrderedSet
random.seed(42)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from prompt import generate_nokeytags_prompt, generate_reduce_prompt, generate_evol_prompt_multitag

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
        id2instructions = {data['id']:data['instruction'] for data in datas}
        id2input = {data['id']: \
            [{"role": "user", "content": generate_nokeytags_prompt(data['instruction'])}] \
             for data in datas}
        use_ids = [data['id'] for data in datas]
        id2datapoints = {}
        while len(use_ids) > 0:
            tag_inputs = [tokenizer.apply_chat_template(id2input[id], tokenize=False, add_generation_prompt=True) for id in use_ids]
            tag_outputs = self.llm_engine.generate(tag_inputs, sampling_params)
            temp_ids = []
            for i in range(len(tag_inputs)):
                output = tag_outputs[i].outputs[0].text.strip()
                try:
                    # print(tag_inputs[i])
                    # print(output)
                    # input()
                    if "```" in output:
                        tags = re.findall("```(.+?)```", output, re.DOTALL)[0]
                    else:
                        tags = output
                    tags = tags.strip().strip("json").strip()
                    tags = json.loads(tags)
                    id2input[use_ids[i]].append({"role": "assistant", "content": output})
                    id2datapoints[use_ids[i]] = {
                        "tag_context":id2input[use_ids[i]],
                        "tags": tags,
                        "instruction": id2instructions[use_ids[i]]
                    }
                except Exception as e:
                    # print(e)
                    temp_ids.append(use_ids[i])
                    continue
            use_ids = temp_ids
        return id2datapoints

    def tags_reduce_instruction(self, data_points, temperature, max_tokens, batch, evol_use_tag_data_points, round, num_tag):
        import re,math
        from collections import defaultdict
        generate_data_points = defaultdict(dict)
        id2inputs = {}

        all_evol_use_tag_ids = list(evol_use_tag_data_points.keys())
        for _ in range(round):
            random.shuffle(all_evol_use_tag_ids)
        for id in tqdm.tqdm(data_points):
            # if int(id) > 100:
            #     break
            shot_ids = random.sample(all_evol_use_tag_ids, k=batch)
            shot_tags = [evol_use_tag_data_points[x]['tags'] for x in shot_ids]
            merge_tags = []
            for tags in shot_tags:
                len_v = 0
                clean_tags = [str(v['tag']).lower().strip().replace('_', ' ') for v in tags]
                merge_tags += clean_tags
            # print(merge_tags)
            ori_instruction = data_points[id]['instruction']
            ori_tags = [str(v['tag']).lower().strip().replace('_', ' ') for v in data_points[id]['tags']]
            final_merge_tags = list(OrderedSet(merge_tags) - OrderedSet(ori_tags))
            # print(ori_tags)
            # print(merge_tags)
            # input()
            id2inputs[id] = [{"role": "user",
                             "content": generate_evol_prompt_multitag(final_merge_tags, ori_instruction, num_tag)}]

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        use_ids = list(id2inputs.keys())
        all_results = []
        while len(use_ids) > 0:
            temp_inputs = [tokenizer.apply_chat_template(id2inputs[id], tokenize=False, add_generation_prompt=True) for id in use_ids]
            temp_outputs = self.llm_engine.generate(temp_inputs, sampling_params)
            temp_ids = []
            for i in range(len(temp_inputs)):
                output = temp_outputs[i].outputs[0].text.strip()
                try:
                    assert len(tokenizer.tokenize(temp_outputs[i].outputs[0].text)) < max_tokens
                    assert "#Finally Rewritten Instruction#" in output
                    rewrite_inst = output.split("#Finally Rewritten Instruction#")[1].strip(":").strip()
                    assert len(rewrite_inst) > 0
                    # print(temp_inputs[i])
                    # print("-------")
                    # print(output)
                    # input()
                    all_results.append(
                        {
                            "instruction":rewrite_inst,
                            "input":"",
                            "output":"",
                            "id":use_ids[i],
                            "ori_instruction":data_points[use_ids[i]]['instruction'],
                        }
                    )
                except Exception as e:
                    temp_ids.append(use_ids[i])
                    continue
            use_ids = temp_ids


            
        return all_results



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
         debug: bool, tp: int, batch: int, tag_file:str, round: int, num_tag: int) -> None:
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
    evol_tags_datas = get_ori_datas(tag_file)
    if debug:
        ori_datas = ori_datas[:100]  # Limit to the first 100 instructions for debugging
    engine = TagReduce(model_name_or_path, tensor_parallel_size=tp)
    #加载当前待进化文本的tag文件
    if os.path.exists(f"{source_file}.nokeyfgtag_datas.json"):
        print(f"{source_file}.nokeyfgtag_datas.json exsits")
        data_points = json.load(open(f"{source_file}.nokeyfgtag_datas.json"))
    else:
        print(f"{source_file}.nokeyf gtag_datas.json not exsits")
        data_points = engine.generate_tags(ori_datas, temperature, max_tokens)
        json.dump(data_points, open(f"{source_file}.nokeyfgtag_datas.json", 'w'))
    #加载待进化使用的tag文件
    if os.path.exists(f"{tag_file}.nokeyfgtag_datas.json"):
        print(f"{tag_file}.nokeyfgtag_datas.json exsits")
        evol_use_tag_data_points = json.load(open(f"{tag_file}.nokeyfgtag_datas.json"))
    else:
        print(f"{tag_file}.nokeyfgtag_datas.json not exsits")
        evol_use_tag_data_points = engine.generate_tags(evol_tags_datas, temperature, max_tokens)
        json.dump(evol_use_tag_data_points, open(f"{tag_file}.nokeyfgtag_datas.json", 'w'))
    all_results = engine.tags_reduce_instruction(data_points, temperature, max_tokens, batch, evol_use_tag_data_points, round, num_tag)
    # Write results to the target file
    try:
        with open(target_file, "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Results successfully saved to {target_file}")
    except Exception as e:
        logger.error(f"Error saving results to {target_file}: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TagReduce Instructions Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--source_file', type=str, required=True, help='Path to the source JSON file')
    parser.add_argument('--tag_file', type=str, required=True, help='Path to the source JSON file')
    parser.add_argument('--target_file', type=str, required=True, help='Path to the target JSON file')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens for generation')
    parser.add_argument('--batch', type=int, default=10, help='Maximum tokens for generation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--tp', type=int, default=8, help='Number of tensor parallel workers')
    parser.add_argument('--round', type=int, default=8, help='Number of tensor parallel workers')
    parser.add_argument('--num_tag', type=int, default=8, help='Number of tensor parallel workers')

    args = parser.parse_args()

    main(args.model_name_or_path, args.source_file, args.target_file, args.temperature, args.max_tokens, args.debug, args.tp, args.batch, args.tag_file, args.round, args.num_tag)
