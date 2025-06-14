import json
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer
import asyncio
from collections import defaultdict
import os
import tqdm
import random
from ordered_set import OrderedSet
from openai import AsyncOpenAI

random.seed(42)
# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
from prompt import generate_cgtags_prompt, generate_reduce_prompt, generate_evol_prompt_multitag_20mul3up

class TagReduce:
    def __init__(self, model_name) -> None:
        """
        Initializes the EvolInstruct object with model details and engine configuration.

        Args:
            model_name_or_path (str): Path to the model or model name.
            tensor_parallel_size (int): Number of tensor parallel workers (default: 8).
            dtype (str): The data type used for inference (default: "bfloat16").
        """
        self.model_name = model_name

        self.aclient = AsyncOpenAI(

            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",

            api_key="sk-c4ab5f8eb099437ead903e7890068e72"
        )
    async def async_query_openai(self, query, temp, max_tokens, idx):
        repeat = 0
        max_repeat = 20
        while True:
            try:
                repeat += 1
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
            except Exception as e:
                print(e.message)
                if repeat >= max_repeat or "Input data may contain inappropriate content." in e.message or "Output data may contain inappropriate content." in e.message:
                    return (idx, "erro")
                await asyncio.sleep(2) 
        # print(completion.choices[0].message.content)
        return (idx, completion.choices[0].message.content)
    
    async def async_process_queries(self, queries, temp, max_tokens):
        idxs = list(range(len(queries)))
        tasks = [self.async_query_openai(query, temp, max_tokens, idx) for idx, query in zip(idxs, queries)]
        results = []
        for future in tqdm.asyncio.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Querying OpenAI"):
            result = await future  # 等待任务完成并获取结果
            results.append(result)
        results = sorted(results, key=lambda x:x[0])
        final = []
        for i, (_, res) in enumerate(results):
            if res != "erro":
                final.append(res)
            else:
                final.append(queries[i])
        return final

    async def generate_tags(self, datas, temperature, max_tokens):
        import re
        id2instructions = {data['id']:data['instruction'] for data in datas}
        id2input = {data['id']: \
            generate_cgtags_prompt(data['instruction']) \
             for data in datas}
        use_ids = [data['id'] for data in datas]
        id2datapoints = {}
        repeat=0
        max_repeat=10
        while len(use_ids) > 0:
            repeat += 1
            tag_inputs = [id2input[id] for id in use_ids]
            tag_outputs = await self.async_process_queries(tag_inputs, temperature, max_tokens)

            temp_ids = []
            for i in range(len(tag_inputs)):
                output = tag_outputs[i].strip()
                try:
                    tags = {}
                    if repeat < max_repeat:
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

    async def tags_reduce_instruction(self, data_points, temperature, max_tokens, num_pool, evol_use_tag_data_points, num_tag):
        import re,math
        from collections import defaultdict
        generate_data_points = defaultdict(dict)
        id2inputs = {}

        all_evol_use_tags = []
        for id in evol_use_tag_data_points:
            for k in evol_use_tag_data_points[id]['tags']:
                all_evol_use_tags += [str(v).replace('_', ' ').lower().strip() for v in evol_use_tag_data_points[id]['tags'][k]]
        all_evol_use_tags = OrderedSet(all_evol_use_tags)
        for id in tqdm.tqdm(data_points):
            # if int(id) > 100:
            #     break
            merge_tags = random.sample(all_evol_use_tags, k=num_pool)
            ori_instruction = data_points[id]['instruction']
            ori_tags = []
            for key in data_points[id]['tags']:
                clean_tags = [str(v).lower().strip().replace('_', ' ') for v in data_points[id]['tags'][key]]
                ori_tags += clean_tags
            final_merge_tags = list(OrderedSet(merge_tags) - OrderedSet(ori_tags))
            # print(ori_tags)
            # print(merge_tags)
            # input()
            id2inputs[id] = generate_evol_prompt_multitag_20mul3up(final_merge_tags, ori_instruction, num_tag)


        use_ids = list(id2inputs.keys())
        all_results = []
        repeat=0
        max_repeat=10
        while len(use_ids) > 0:
            repeat += 1
            temp_inputs = [id2inputs[id] for id in use_ids]
            infer_batch = 128
            temp_outputs = []
            for i in range(0, len(temp_inputs), infer_batch):
                print(i)
                temp_outputs += await self.async_process_queries(temp_inputs[i:i+infer_batch], temperature, max_tokens)
            temp_ids = []
            for i in range(len(temp_inputs)):
                output = temp_outputs[i].strip()
                try:
                    rewrite_inst = data_points[use_ids[i]]['instruction']
                    if repeat < max_repeat:
                        # assert len(tokenizer.tokenize(temp_outputs[i].outputs[0].text)) < max_tokens
                        assert "#Finally Rewritten Instruction#" in output
                        rewrite_inst = output.split("#Finally Rewritten Instruction#")[1].strip(":").strip()
                        assert len(rewrite_inst) > 0
                    
                    tag_datas = output
                    # print(temp_inputs[i])
                    # print("-------")
                    # print(output)
                    # print(tag_datas)
                    # input()
                    all_results.append(
                        {
                            "instruction":rewrite_inst,
                            "input":"",
                            "output":"",
                            "id":use_ids[i],
                            "ori_instruction":data_points[use_ids[i]]['instruction'],
                            "tags":tag_datas
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
        raise
    except json.JSONDecodeError:
        # logger.error(f"Error decoding JSON from the file: {source_file}")
        raise
    return source_data


async def main(model_name_or_path: str, source_file: str,
         target_file: str,
         temperature: float, max_tokens: int,
         debug: bool, tp: int, num_pool: int, tag_file:str, num_tag: int) -> None:
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
        data_points = engine.generate_tags(ori_datas, temperature, max_tokens)
        json.dump(data_points, open(f"{source_file}.cgtag_datas.json", 'w'))
    #加载待进化使用的tag文件
    if os.path.exists(f"{tag_file}.cgtag_datas.json"):
        print(f"{tag_file}.cgtag_datas.json exsits")
        evol_use_tag_data_points = json.load(open(f"{tag_file}.cgtag_datas.json"))
    else:
        print(f"{tag_file}.cgtag_datas.json not exsits")
        evol_use_tag_data_points = engine.generate_tags(evol_tags_datas, temperature, max_tokens)
        json.dump(evol_use_tag_data_points, open(f"{tag_file}.cgtag_datas.json", 'w'))
    all_results = await engine.tags_reduce_instruction(data_points, temperature, max_tokens, num_pool, evol_use_tag_data_points, num_tag)
    id2oridatas = {str(data['id']):data for data in ori_datas}
    for result in all_results:
        result['input'] = id2oridatas[str(result['id'])]['input']
    # Write results to the target file
    try:
        with open(target_file, "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        # logger.info(f"Results successfully saved to {target_file}")
    except Exception as e:
        # logger.error(f"Error saving results to {target_file}: {e}")
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
    parser.add_argument('--num_pool', type=int, default=10, help='Maximum tokens for generation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--tp', type=int, default=8, help='Number of tensor parallel workers')
    # parser.add_argument('--round', type=int, default=8, help='Number of tensor parallel workers')
    parser.add_argument('--num_tag', type=int, default=8, help='Number of tensor parallel workers')

    args = parser.parse_args()

    asyncio.run(main(args.model_name_or_path, args.source_file, args.target_file, args.temperature, args.max_tokens, args.debug, args.tp, args.num_pool, args.tag_file, args.num_tag))
