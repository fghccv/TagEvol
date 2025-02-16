# import json

# datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/Evol_Instruct/datas/gsm8k_evol-roud1.json"))
# datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/Evol_Instruct/datas/gsm8k_evol-roud2.json"))
# datas3 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/Evol_Instruct/datas/gsm8k_evol-roud3.json"))
# datas = datas1 + datas2  + datas3
# print(len(datas))
# json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/Evol_Instruct/datas/gsm8k_evol-roud1+2+3.json", 'w'))
import json
datas = json.load(open("/data/scir/yixuanwang/TagReduce/TagReduce-math-all/Auto_Evol_Instruct/datas/Auto_evol-round1.json"))
print(datas[0])