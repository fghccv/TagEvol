
import json


def generate_instruction_prompt(instructions):
    prompt = (
        "You are an instruction generator capable of creating high-quality, challenging instructions.\n"
        "Please imitate the way humans ask questions, generate a high-quality, challenging instruction.\n"
        "You need't generate the answer of the instruction.\n"
    )
    for i, instruction in enumerate(instructions):
        prompt += f"# Example {i+1}\n"
        prompt += f"## Instruction:\n{instruction}\n\n"
    prompt += f"# Example {len(instructions)+1}\n"
    prompt += "## Instruction:"
    return prompt