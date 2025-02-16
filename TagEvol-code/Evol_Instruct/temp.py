import json
# with open("/home/zhoushiqi/workplace/icaed/code/cot-instruct/Evol_Instruct/code_alpaca_20k.jsonl") as f:
#     datas = [json.loads(l) for l in f]
# new_datas = []
# for data in datas:
#     new_datas.append({"instruction":data['instruction'], 'output':'', 'input':""})
# f = open("/home/zhoushiqi/workplace/icaed/code/cot-instruct/Evol_Instruct/code_alpaca_20k_qwen72b_res.json", 'w')
# json.dump(new_datas, f)
datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/Evol_Instruct/datas/code_alpaca_20k-evol-roud0.json"))
datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/Evol_Instruct/datas/code_alpaca_20k-evol-roud1.json"))

datas = datas1 + datas2
print(len(datas))
json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/Evol_Instruct/datas/code_alpaca_20k-evol-roud01.json", 'w'))