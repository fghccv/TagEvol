import json
import logging
import re
from typing import List, Dict, Any
from transformers import AutoTokenizer
import asyncio
from auto_prompt import create_auto_prompt
from openai import AsyncOpenAI
import tqdm
# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class EvolInstruct:
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

    async def generate_responses(self, instructions: List[str], inputs: List[str], ids: List[str], temperature: float = 0, max_tokens: int = 2048) -> List[Dict[str, Any]]:
        """
        Generate responses based on provided instructions and inputs.

        Args:
            instructions (List[str]): List of instructions to be used for generation.
            inputs (List[str]): Corresponding inputs to the instructions.
            temperature (float): Sampling temperature (default: 0.7).
            max_tokens (int): Maximum tokens to generate (default: 2048).

        Returns:
            List[Dict[str, Any]]: List of responses with the instruction, input, and output.
        """
        
        # Prepare instructions for chat generation format
        evol_inputs = [inst for inst in instructions]
        infer_batch = 128
        evol_outputs = []
        for i in range(0, len(evol_inputs), infer_batch):
            print(i)
            evol_outputs += await self.async_process_queries(evol_inputs[i:i+infer_batch], temperature, max_tokens)
        
        final_insts = []
        # process the output
        # import pdb
        # pdb.set_trace()
        for output in evol_outputs:
            processed_output = output.strip()
            # print(processed_output)
            # input()
            final_inst = processed_output.split("#Final Rewritten Instruction#:\n", 1)
            if len(final_inst) > 1:
                final_insts.append(final_inst[1])
            else:
                final_insts.append("")
        
        results = []
        for i in range(len(instructions)):
            if final_insts[i]:
                results.append({
                    "instruction": final_insts[i],
                    "input": inputs[i],
                    "output": "",
                    "id":ids[i]
                })
            else:
                continue
        
        # logger.info(f"Final evolution instructions are: {len(results)}")
        # logger.info(f"The original instructions are: {len(instructions)}")

        return results

def get_evol_instructions(source_file: str) -> List[str]:
    """
    Extract instructions from the source file and construct new instructions based on the round number.

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

    # Generate evolutionary instructions by applying each prompt to the given instructions
    evol_inst = [
        create_auto_prompt(item["instruction"])
        for item in source_data
    ]
    # logger.debug(f"Generated instructions...")  # Debug output for first few instructions

    # Extract inputs from the source data
    inputs = [item["input"] for item in source_data]
    ids = [str(item["id"]) for item in source_data]
    return evol_inst, inputs, ids

async def main(model_name_or_path: str, source_file: str,
         target_file: str, temperature: float,
         max_tokens: int, debug: bool, tp: int) -> None:
    """
    Main function to generate evolutionary instructions and store the results in the target file.

    Args:
        model_name_or_path (str): Path to the model or model name.
        source_file (str): Path to the source JSON file containing the instructions.
        target_file (str): Path to the target JSON file to save the results.
        round_num (int): Evolutionary round number.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum tokens to generate.
        debug (bool): Whether to enable debug mode (limits results to 100).
        use_breadth (bool): Whether to use breadth-based evolution.
        tp (int): Number of tensor parallel workers.
    """
    instructions, inputs, ids = get_evol_instructions(source_file)

    if debug:
        instructions = instructions[:100]  # Limit to the first 100 instructions for debugging

    engine = EvolInstruct(model_name_or_path)
    all_results = await engine.generate_responses(instructions, inputs, ids, temperature, max_tokens)

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

    parser = argparse.ArgumentParser(description="Run Evolutionary Instructions Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--source_file', type=str, required=True, help='Path to the source JSON file')
    parser.add_argument('--target_file', type=str, required=True, help='Path to the target JSON file')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens for generation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--tp', type=int, default=1, help='Number of tensor parallel workers')

    args = parser.parse_args()

    asyncio.run(main(args.model_name_or_path, args.source_file, args.target_file, args.temperature, args.max_tokens, args.debug, args.tp))
