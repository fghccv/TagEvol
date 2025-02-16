# import json
# tag_datas1 = json.load(open("/data/scir/yixuanwang/TagReduce/TagReduce-math/tag_reduce/datas/gsm8k_7k/gsm8k_7k.json.cgtag_datas_72b.json"))
# tag_datas2 = json.load(open("/data/scir/yixuanwang/TagReduce/TagReduce-math/tag_reduce/datas/gsm8k_7k/gsm8k_7k.json.cgtag_datas.json"))
# # ln1 = []
# # for id in tag_datas1:
# #     tags = tag_datas1[id]['tags']
# #     # print(tags)
# #     for k in tags: 
# #         ln1 += [str(v).replace("_", " ").lower().strip() for v in tags[k]]
# # print(len(set(ln1)))
# ln1 = []
# for id in tag_datas2:
#     tags = tag_datas2[id]['tags']
#     # print(tags)
#     for k in tags: 
#         ln1 += [str(v).replace("_", " ").lower().strip() for v in tags[k]]
# print(len(set(ln1)))
import json
# datas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round0_tag1_alltags/tag_evol_batch5_round0_tag1_alltags.json"))
datas2 = json.load(open("/data/scir/yixuanwang/TagReduce/TagReduce-math/tag_reduce/datas/tag_evol_tag5_pool_21_20mul3up_alltags/tag_evol_tag5_pool_21_20mul3up_alltags.json"))
# id2datas = {data['id']:data for data in datas}
id2datas2 = {data['id']:data for data in datas2}
for id in id2datas2:
    print(id2datas2[id]['ori_instruction'])
    print("-"*10)
    # print(id2datas[id]['instruction'])
    # print("-"*10)
    print(id2datas2[id]['instruction'])
    print("============")
    input()
ln = [len(data['instruction'].split(" ")) for data in datas2]
print(sum(ln)/len(ln))