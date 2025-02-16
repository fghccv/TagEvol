import json
import re

split_pattern = "#+ ?[Ii]nput ?:?"
split_words = ["Your task is"]
data_path = "/home/zhoushiqi/workplace/TagReduce/TagReduce/tag_reduce/datas/code_alpaca_20k_zeroshot_cr.json"
dest_path = "/home/zhoushiqi/workplace/TagReduce/TagReduce/tag_reduce/datas/code_alpaca_20k_zeroshot_cr_input.json"
datas = json.load(open(data_path))
new_datas = []
for data in datas:
    instruction = data['instruction']
    split = re.split(split_pattern, instruction)
    inputs = ''
    if len(split) >=2:
        instruction, inputs = split[0], split[-1]
        inputs = inputs.split("###")[0]
        instruction = instruction.strip()
        inputs = inputs.strip()
        if inputs.lower() in ['', 'none']:
            inputs = ''
    data['instruction'] = instruction
    data['input'] = inputs
    new_datas.append(data)
json.dump(new_datas, open(dest_path, 'w'))