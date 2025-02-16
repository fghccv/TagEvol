import json
from typing import Dict, Iterable
from evalplus.data import get_human_eval_plus, get_mbpp_plus
import gzip
import json
def read_jsonl(path):
    datas = []
    with open(path) as f:
        for l in f:
            datas.append(json.loads(l))
    return datas

INSTRUCTION = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:'''


instructions = {
    "humaneval": '''Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```python
{}
```''',
    "ds1000": 'Write a short code following the given format and indentation. Place the executable code between <code> and </code> tags, without any other non-executable things.\n{}',
    "default": "Create a Python script for this problem:```python\n{}\n```",
}
PROMPT_DICT = {
    "prompt_input_humaneval": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```python
{instruction}
```

### Input:
{input}

### Response:"""
    ,
    "prompt_input_mbpp": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
```python
{instruction}
```

### Input:
{input}

### Response:""",
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def generate_sft_prompt(inp, dataset):
    inst = inp
    if dataset in instructions:
        inst = instructions[dataset].format(inp)
    else:
        inst = instructions["default"].format(inp)      
    return INSTRUCTION.format(inst)


def read_from_jsonl(dataset):
    if dataset == "humaneval":
        problems = get_human_eval_plus()
        for p in problems.keys():
            problems[p]["prompt"] = generate_sft_prompt(problems[p]["prompt"].strip(), "default")
            # problems[p]["prompt"] = problems[p]["prompt"].strip()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
        for p in problems.keys():
            problems[p]["prompt"] = generate_sft_prompt(problems[p]["prompt"].strip(), dataset)
            # problems[p]["prompt"] = problems[p]["prompt"].strip()
    elif dataset == "ds1000":    
        problems = {}
        path = f"../dataset/{dataset}.jsonl"
        f = open(path, "r")
        for line in f.readlines():
            d = json.loads(line)
            task_id = d["metadata"]["problem_id"]
            d["task_id"] = task_id    
            prompt,response_start = preprocess_completion_prompt(d["prompt"])
            d["prompt"] = INSTRUCTION.format(instructions["ds1000"].format(prompt)) + response_start
            problems[d["task_id"]] = d
    return problems


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def preprocess_completion_prompt(prompt: str) -> tuple[str, str]:
    """Preprocess the DS-1000 prompt (Completion mode) into instruction and response prefix"""
    # hit = False
    if not "SOLUTION START" in prompt:
        answer_index = prompt.rindex("A:")
        answer = prompt[answer_index + 2 :].strip()
        instruction: str = prompt[:answer_index].strip()
        if instruction.startswith("Problem:"):
            instruction = instruction[len("Problem:") :].strip()
        if "### BEGIN SOLUTION" in prompt:
            assert prompt.count("<code>") == 1
            assert prompt.count("</code>") == 0
            lines = answer.splitlines(keepends=True)
            return_line, result_line, begin_line = lines[-3:]
            assert return_line.strip().startswith("# return")
            assert result_line.strip().startswith("# ")
            assert begin_line.strip() == "### BEGIN SOLUTION"
            response = "".join(lines[:-3]).strip()
            hint = begin_line.replace("###", "#").replace("BEGIN SOLUTION", "Solution")
            response += f"\n{hint}\n"
        else:
            assert "BEGIN SOLUTION" in prompt
            assert prompt.count("<code>") == 2
            assert prompt.count("</code>") == 1
            first_block_start = prompt.index("<code>")
            first_block_end = prompt.index("</code>")
            second_block_start = prompt.index("<code>", first_block_start + 1)
            assert first_block_end < second_block_start
            lines = answer.splitlines(keepends=True)
            block_end, instruction_line, begin_line, block_start = lines[-4:]
            assert begin_line.strip() == "BEGIN SOLUTION"
            assert block_start.strip() == "<code>"
            if not block_end.strip() == "</code>":
                if lines[-6].strip() == "</code>":
                    response_prefix = lines[:-6]
                    starting_lines = lines[-5:-2]
                else:
                    assert instruction_line.strip() == "</code>"
                    response_prefix = lines[:-3]
                    starting_lines = lines[-2:-2]
            else:
                response_prefix = lines[:-4]
                starting_lines = lines[-3:-2]
            starting_lines = [f"# {line.lstrip()}" for line in starting_lines]
            response = "".join([*response_prefix, *starting_lines]).strip()
            response += "\n# Solution\n"
    else:
        # hit = True
        assert prompt.count("<code>") == 0
        assert prompt.count("</code>") == 0
        assert prompt.strip().endswith("# SOLUTION START")
        code_prefix = prompt[: prompt.rindex("# SOLUTION START")].strip()
        instruction = f"""Write a solution to the following problem:
```python
{code_prefix}
```"""
        response = f"```python\n{code_prefix}\n# Solution\n"
    instruction = instruction.replace("<code>", "```python").replace("</code>", "```")
    response = response.replace("<code>", "```python").replace("</code>", "```")
    # if hit:
    #     print("[Instruction]")
    #     print(instruction)
    #     print("[Response]")
    #     print(response)
    #     breakpoint()
    return instruction, response

if __name__ == "__main__":
    problems = read_from_jsonl("ds1000")
    for p in problems.keys():
        print("PROMPT")
        print(problems[p]["prompt"])
        print("Response")
        print(problems[p]["response_start"])
        print("===")