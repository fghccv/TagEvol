import json
oridatas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/gsm8k_7k/gsm8k_7k_res72b.json"))
datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round0_tag1_alltags/tag_evol_batch5_round0_tag1_alltags.json"))
datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round1_tag3_20mul3up_alltags/tag_evol_batch5_round1_tag3_20mul3up_alltags.json"))
datas3 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round2_tag5_20mul3up_alltags/tag_evol_batch5_round2_tag5_20mul3up_alltags.json"))
oriscores = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/score_datas/Meta-Llama-3-8B_ms/gsm8k_7k_res72b.json"))['ifdx_score']
scores1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/score_datas/Meta-Llama-3-8B_ms/tag_evol_batch5_round0_tag1_alltags.json"))['ifdx_score']
scores2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/score_datas/Meta-Llama-3-8B_ms/tag_evol_batch5_round1_tag3_20mul3up_alltags.json"))['ifdx_score']
scores3 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/score_datas/Meta-Llama-3-8B_ms/tag_evol_batch5_round2_tag5_20mul3up_alltags.json"))['ifdx_score']
assert len(datas1) == len(scores1)
assert len(datas2) == len(scores2)
assert len(datas3) == len(scores3)
assert len(oridatas) == len(oriscores), f"{len(len(oridatas))},{len(oriscores)}"
id2oridatas = {d['id']: [d,s] for d,s in zip(oridatas, oriscores)}
id2datas1 = {d['id']: [d,s] for d,s in zip(datas1, scores1)}
id2datas2 = {d['id']: [d,s] for d,s in zip(datas2, scores2)}
id2datas3 = {d['id']: [d,s] for d,s in zip(datas3, scores3)}
final_datas = []
x1, x2, x3=0,0,0
for idx, id in enumerate(id2oridatas):
    oriscore=id2oridatas[id][1]
    id = str(id)
    if id in id2datas1:
        score = id2datas1[id][1]
        assert id2datas1[id][0]['ori_instruction'] == id2oridatas[int(id)][0]['instruction']
        if oriscore < score:
            x1 += 1
            final_datas.append(id2datas1[id][0])
    if id in id2datas2:
        score = id2datas2[id][1]
        assert id2datas2[id][0]['ori_instruction'] == id2oridatas[int(id)][0]['instruction']
        if oriscore < score:
            x2 += 1
            final_datas.append(id2datas2[id][0])
    if id in id2datas3:
        score = id2datas3[id][1]
        assert id2datas3[id][0]['ori_instruction'] == id2oridatas[int(id)][0]['instruction']
        if oriscore < score:
            x3 += 1
            final_datas.append(id2datas3[id][0])

json.dump(final_datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round0+1+2/tag_evol_batch5_round2_tag135_20mul3up_alltags_inevol.json", 'w'))
print(len(final_datas))
print(x1, x2, x3)
ids = set([data['id'] for data in final_datas])
print(len(ids))