import json
datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/Auto_Evol_Instruct/datas/gsm8k_Auto_evol-roud0.json"))
datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/Auto_Evol_Instruct/datas/gsm8k_Auto_evol-roud1.json"))
datas3 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/Auto_Evol_Instruct/datas/gsm8k_Auto_evol-roud2.json"))

datas = datas1 + datas2 + datas3
print(len(datas))
json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/Auto_Evol_Instruct/datas/gsm8k_Auto_evol-roud0+1+2.json", 'w'))