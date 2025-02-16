# import json
# tag_datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k.json.sctag_datas.json"))
# tag_datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k.json.cgtag_datas.json"))
# ln1 = []
# for k in tag_datas1:
#     ln1 += tag_datas1[k]['tags']
# ln1 = set(ln1)
# print(len(ln1))
# ln2 = []
# for k in tag_datas2:
#     for t in tag_datas2[k]['tags']:
#         ln2  += [str(v).replace("_", " ").lower().strip() for v in tag_datas2[k]['tags'][t]]
# ln2 = set(ln2)
# print(len(ln2))
# import json
# oridatas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k.json"))
# datas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round0/tag_evol_batch5_round0.json"))
# datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round1_tag3_newprompt/tag_evol_batch5_round1_tag3_newprompt.json"))
# datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round2_tag5/tag_evol_batch5_round2_tag5.json"))
# id2oridatas = {str(data['id']):data for data in oridatas}
# id2datas = {data['id']:data for data in datas}
# id2datas1 = {data['id']:data for data in datas1}
# id2datas2 = {data['id']:data for data in datas2}
# for id in id2datas:
#     print(f"ori:\n{id2oridatas[id]['instruction']}")
#     print(f"tag1:\n{id2datas[id]['instruction']}")
#     print(f"tag3:\n{id2datas1[id]['instruction']}")
#     print(f"tag5:\n{id2datas2[id]['instruction']}")
#     print("-"*10)
#     input()
# # ln = [len(data['instruction'].split(" ")) for data in datas]
# # print(sum(ln)/len(ln))
# # for data in datas:
# #     print(data['instruction'])
# #     print("-"*10)
# #     input()

# import json
# datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce_new/datas/tag_evol_batch5_round0_tag1/tag_evol_batch5_round0_tag1.json"))
# datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce_new/datas/tag_evol_batch5_round1_tag3/tag_evol_batch5_round1_tag3.json"))
# datas3 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce_new/datas/tag_evol_batch5_round2_tag5/tag_evol_batch5_round2_tag5.json"))
# datas = datas1 + datas2 +datas3
# for data in datas:
#     data['id'] = str(data['id'])
# json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce_new/datas/tag_evol_batch5_round0+1+2/tag_evol_batch5_round0+1+2.json", 'w'))

import json
ori_datas=json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce_new/datas/code_alpaca_20k/code_alpaca_20k.json"))
datas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce_new/datas/tag_evol_tag3_70b_yxprompt/tag_evol_tag3_70b_yxprompt.json"))
for data in datas:
    print(data['id'])
    print(data['instruction'])
    print(data['input'])
    print(data['ori_instruction'])
    print("-"*10)
    input()
# id2oridatas = {str(data['id']):data for data in ori_datas}
# for data in datas:
#     data['input'] = id2oridatas[str(data['id'])]['input']
# json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce_new/datas/tag_evol_tag5_70b_yxprompt/tag_evol_tag5_70b_yxprompt_new.json", "w"))
