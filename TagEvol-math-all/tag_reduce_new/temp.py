# import json
# tag_datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k.json.nokeyfgtag_datas.json"))
# tag_datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k.json.cgtag_datas.json"))
# ln1 = []
# for k in tag_datas1:
#     tags = tag_datas1[k]['tags']
#     ln1 += [str(tag['tag']).replace("_", " ").lower().strip() for tag in tags]
# print(len(set(ln1)))
# ln2 = []
# for k in tag_datas2:
#     tags = tag_datas2[k]['tags']
#     for t in tags:
#         ln2 += [str(tag).replace("_", " ").lower().strip() for tag in tags[t]]
# print(len(set(ln2)))

# for k in tag_datas1:
#     inst = tag_datas1[k]['instruction']
#     tags1 = tag_datas1[k]['tags']
#     tags2 = tag_datas2[k]['tags']
#     print(inst)
#     print(tags1)
#     print(tags2)
#     print("-"*10)
#     input()
# import json
# datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round0_tag1_alltags/tag_evol_batch5_round0_tag1_alltags.json"))
# datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_tag5_pool_21_20mul3up_alltags_shuffle1/tag_evol_tag5_pool_21_20mul3up_alltags_shuffle1.json"))
# datas3 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_tag5_pool_21_20mul3up_alltags_shuffle0/tag_evol_tag5_pool_21_20mul3up_alltags_shuffle0.json"))
# datas = datas1 + datas2 + datas3
# print(len(datas))
# json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round0+1+2/tag_evol_tag5_pool_21_20mul3up_alltags_tag155.json", "w"))
# import json
# import random
# tag_datas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k.json.cgtag_datas.json"))
# all_ln = []

# for _ in range(1000):
#     batchs = random.sample(list(tag_datas.keys()), 5)
#     ln = []
#     for id in batchs:
#         temp_v = []
#         for k in tag_datas[id]['tags']:
#             temp_v += [str(v).replace('_', ' ').lower().strip() for v in tag_datas[id]['tags'][k]]
#         ln += temp_v
#     ln = set(ln)
#     all_ln.append(len(ln))
# print(sum(all_ln)/len(all_ln))
# import json
# datas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k.json"))
# for data in datas:
#     data['ori_instruction'] = ""
# json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k.json", 'w'))
# import json
# datas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_tag5_pool_21_20mul3up_alltags_shuffle2/tag_evol_tag5_pool_21_20mul3up_alltags_shuffle2.json"))
# for data in datas:
#     print(data['instruction'])
#     print(data['ori_instruction'])
#     print('-'*10)
#     input()

import json
datas = json.load(open("/data/scir/yixuanwang/TagReduce/TagReduce-math-all/tag_reduce_new/datas/all_math_15k/all_math_15k.json.cgtag_datas.json"))
# ids = set(datas.keys())
# print(len(set([str(i) for i in list(range(14973))])-ids))
print(datas['10'])
# import json
# datas = json.load(open("/data/scir/yixuanwang/TagReduce/TagReduce-math-all/Evol_Instruct/datas/all_math_15k.json"))
# print(len(datas))
# ids = set([str(data['id']) for data in datas])
# print(sorted([int(x) for x in set([str(i) for i in list(range(14929))])-ids]))