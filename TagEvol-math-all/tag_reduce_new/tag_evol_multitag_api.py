import json
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer
import asyncio
from collections import defaultdict
import os
import random
from ordered_set import OrderedSet
from openai import AsyncOpenAI
random.seed(42)
from tqdm.asyncio import tqdm
# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
from prompt import generate_cgtags_prompt, generate_reduce_prompt, generate_evol_prompt_multitag

class TagReduce:
    def __init__(self, model_name: str) -> None:
        """
        Initializes the EvolInstruct object with model details and engine configuration.

        Args:
            model_name_or_path (str): Path to the model or model name.
            tensor_parallel_size (int): Number of tensor parallel workers (default: 8).
            dtype (str): The data type used for inference (default: "bfloat16").
        """
        self.model_name = model_name
        self.aclient = AsyncOpenAI(
            base_url="https://api.deepseek.com",
            api_key="sk-5cf6f5aa7702472cb18420af57775a81"
        )
    async def async_query_openai(self, query, temp, max_tokens, idx):
        while True:
            try:
                completion = await self.aclient.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query}
                    ],
                    temperature=temp,
                    max_tokens=max_tokens
                )
                break
            except Exception:
                await asyncio.sleep(1) 
        print(completion.choices[0].message.content)
        return (idx, completion.choices[0].message.content)

    async def async_process_queries(self, queries, temp, max_tokens, idxs):
        tasks = [self.async_query_openai(query, temp, max_tokens, idx) for idx, query in zip(idxs, queries)]
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Querying OpenAI"):
            result = await future  # 等待任务完成并获取结果
            results.append(result)
        
        return results   

    async def generate_tags(self, datas, temperature, max_tokens):
        import re
        id2instructions = {data['id']:data['instruction'] for data in datas}
        id2input = {data['id']: generate_cgtags_prompt(data['instruction']) for data in datas}
        use_ids = [data['id'] for data in datas]
        id2datapoints = {}
        while len(use_ids) > 0:
            tag_inputs = [id2input[id] for id in use_ids]
            tag_outputs = await self.async_process_queries(tag_inputs, temperature, max_tokens)
            temp_ids = []
            for i in range(len(tag_inputs)):
                output = tag_outputs[i].strip()
                try:
                    tags = re.findall("```(.+?)```", output, re.DOTALL)[0]
                    tags = tags.strip().strip("json").strip()
                    tags = json.loads(tags)
                    id2input[use_ids[i]].append({"role": "assistant", "content": output})
                    id2datapoints[use_ids[i]] = {
                        "tag_context":id2input[use_ids[i]],
                        "tags": tags,
                        "instruction": id2instructions[use_ids[i]]
                    }
                except Exception as e:
                    temp_ids.append(use_ids[i])
                    continue
            use_ids = temp_ids
        return id2datapoints

    async def tags_reduce_instruction(self, data_points, temperature, max_tokens, batch, evol_use_tag_data_points, round, num_tag):
        import re,math
        from collections import defaultdict
        generate_data_points = defaultdict(dict)
        id2inputs = {}

        all_evol_use_tag_ids = list(evol_use_tag_data_points.keys())
        for _ in range(round):
            random.shuffle(all_evol_use_tag_ids)
        for id in data_points:
            shot_ids = random.sample(all_evol_use_tag_ids, k=batch)
            shot_tags = [evol_use_tag_data_points[x]['tags'] for x in shot_ids]
            merge_tags = []
            for tags in shot_tags:
                len_v = 0
                for key in tags:
                    clean_tags = [str(v).lower().strip().replace('_', ' ') for v in tags[key]]
                    merge_tags += clean_tags
            # print(merge_tags)
            ori_instruction = data_points[id]['instruction']
            ori_tags = []
            for key in data_points[id]['tags']:
                clean_tags = [str(v).lower().strip().replace('_', ' ') for v in data_points[id]['tags'][key]]
                ori_tags += clean_tags
            final_merge_tags = list(OrderedSet(merge_tags) - OrderedSet(ori_tags))
            # print(ori_tags)
            # print(merge_tags)
            # input()
            id2inputs[id] = generate_evol_prompt_multitag(final_merge_tags, ori_instruction, num_tag)

        use_ids = list(id2inputs.keys())
        all_results = []
        while len(use_ids) > 0:
            temp_inputs = [id2inputs[id] for id in use_ids]
            temp_outputs = []
            infer_batch = 200
            for i in range(0, len(temp_inputs), infer_batch):
                print(i)
                temp_outputs += await self.async_process_queries(temp_inputs[i:i+infer_batch], temperature, max_tokens, list(range(i, min(i+infer_batch, len(temp_inputs)))))
            temp_ids = []
            for i in range(len(temp_inputs)):
                output = temp_outputs[i].strip()
                try:
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
        # logger.error(f"Source file not found: {source_file}")
        print(f"Source file not found: {source_file}")
        raise
    except json.JSONDecodeError:
        # logger.error(f"Error decoding JSON from the file: {source_file}")
        print(f"Error decoding JSON from the file: {source_file}")
        raise
    return source_data


async def main(model_name_or_path: str, source_file: str,
         target_file: str,
         temperature: float, max_tokens: int,
         debug: bool, batch: int, tag_file:str, round: int, num_tag: int) -> None:
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
    engine = TagReduce(model_name_or_path)
    #加载当前待进化文本的tag文件
    if os.path.exists(f"{source_file}.cgtag_datas.json"):
        print(f"{source_file}.cgtag_datas.json exsits")
        data_points = json.load(open(f"{source_file}.cgtag_datas.json"))
    else:
        print(f"{source_file}.cgtag_datas.json not exsits")
        data_points = await engine.generate_tags(ori_datas, temperature, max_tokens)
        json.dump(data_points, open(f"{source_file}.cgtag_datas.json", 'w'))
    #加载待进化使用的tag文件
    if os.path.exists(f"{tag_file}.cgtag_datas.json"):
        print(f"{tag_file}.cgtag_datas.json exsits")
        evol_use_tag_data_points = json.load(open(f"{tag_file}.cgtag_datas.json"))
    else:
        print(f"{tag_file}.cgtag_datas.json not exsits")
        evol_use_tag_data_points = await engine.generate_tags(evol_tags_datas, temperature, max_tokens)
        json.dump(evol_use_tag_data_points, open(f"{tag_file}.cgtag_datas.json", 'w'))
    all_results = await engine.tags_reduce_instruction(data_points, temperature, max_tokens, batch, evol_use_tag_data_points, round, num_tag)
    # Write results to the target file
    try:
        with open(target_file, "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        # logger.info(f"Results successfully saved to {target_file}")
        print(f"Results successfully saved to {target_file}")
    except Exception as e:
        # logger.error(f"Error saving results to {target_file}: {e}")
        print(f"Error saving results to {target_file}: {e}")
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
    parser.add_argument('--round', type=int, default=8, help='Number of tensor parallel workers')
    parser.add_argument('--num_tag', type=int, default=8, help='Number of tensor parallel workers')

    args = parser.parse_args()

    asyncio.run(main(args.model_name_or_path, args.source_file, args.target_file, args.temperature, args.max_tokens, args.debug, args.batch, args.tag_file, args.round, args.num_tag))
