import json
import re
path = "/home/zhoushiqi/workplace/TagReduce/TagReduce/tag_reduce/datas/code_alpaca_20k_zeroshot_cr.json"
output_path = "/home/zhoushiqi/workplace/TagReduce/TagReduce/tag_reduce/datas/code_alpaca_20k_zeroshot_cr_filter_new.json"
split_words = ["### Python Code:", "### Implementation:", "###Instructions:",\
               "### Solution:", "### Expected SQL Query:", "###Expected Output (Ruby):", \
               "### JavaScript:"
               ]
patterns = ["### ?Example.*:", "### ?Explanation.*:", "###.*Implementation:"]

datas = json.load(open(path))

# for data in datas:
#     instruction = data['instruction']
#     for words in split_words:
#         instruction = instruction.split(words)[0]
#     for pattern in patterns:
#         instruction = re.split(pattern, instruction)[0]
#     data['instruction'] = instruction
#     data['output'] = ""
for data in datas:
    instruction = data['instruction']
    split_sens = re.findall("###(.+?):", instruction)
    flag = 0
    # print(instruction)
    for i, words in enumerate(split_sens):
        if "input" == words.lower().strip():
            flag = 1
            break
    # print(i, flag)
    if flag and i == len(split_sens)-1:
        for words in split_words:
            instruction = instruction.split(words)[0]
        for pattern in patterns:
            instruction = re.split(pattern, instruction)[0]
    elif flag:
        idx = i
        tags = f"###{split_sens[i+1]}:"
        instruction = instruction.split(tags)[0]
    elif len(split_sens) != 0:
        tags = f"###{split_sens[0]}:"
        instruction = instruction.split(tags)[0]
    else:
        pass
    instruction = instruction.strip().strip("###Input:")
    for words in split_words:
        instruction = instruction.split(words)[0]
    for pattern in patterns:
        instruction = re.split(pattern, instruction)[0]
    # print('----------')
    # print(instruction)
    # print('=============')
    # input()
    data['instruction'] = instruction
    data['output'] = data['output']
    
    

json.dump(datas, open(output_path, 'w'))
