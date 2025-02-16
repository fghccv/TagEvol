import json
datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce/datas/tag_evol_batch5_roud0_sametags_ifdx_reject.json"))
datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce/datas/tag_evol_batch5_roud1_sametags_ifdx_reject.json"))
scores1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce/score_datas/tag_evol_batch5_roud0_sametags_ifdx_reject.json"))['ifdx_score']
scores2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce/score_datas/tag_evol_batch5_roud1_sametags_ifdx_reject.json"))['ifdx_score']
id2datas1 = {d['id']: [d,s] for d,s in zip(datas1, scores1)}
id2datas2 = {d['id']: [d,s] for d,s in zip(datas2, scores2)}
final_datas = []
x=0
# for d1, d2, s1, s2 in zip(datas1, datas2, scores1, scores2):
#     assert d1['id'] == d2['id']
#     # print(d1['instruction'])
#     # print(s1)
#     # print(s2)
#     # print(d2['instruction'])
#     # input()
#     if s1> s2:
#         final_datas.append(d1)
#         x += 1
#         all_scores.append(s1)
#     else:
#         final_datas.append(d2)
#         all_scores.append(s2)
# print(x)
# print(sum(all_scores)/len(all_scores))
# # json.dump(final_datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce/datas/tag_evol_batch5_roud1_sametags_ifdx_reject_final.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
datas1_ids = set([d['id'] for d in datas1])
datas2_ids = set([d['id'] for d in datas2])
print(len(datas1_ids))
print(len(datas1))
print(len(datas2_ids))
print(sorted(list(datas1_ids - datas2_ids)))