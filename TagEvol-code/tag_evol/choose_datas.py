import json
datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round0/tag_evol_batch5_round0.json"))
datas2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round1/tag_evol_batch5_round1.json"))
datas3 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round2/tag_evol_batch5_round2.json"))
scores1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/score_datas/Meta-Llama-3-8B_ms/tag_evol_batch5_round0.json"))['ifdx_score']
scores2 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/score_datas/Meta-Llama-3-8B_ms/tag_evol_batch5_round1.json"))['ifdx_score']
scores3 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/score_datas/Meta-Llama-3-8B_ms/tag_evol_batch5_round2.json"))['ifdx_score']
assert len(datas1) == len(scores1)
assert len(datas2) == len(scores2)
id2datas1 = {d['id']: [d,s] for d,s in zip(datas1, scores1)}
id2datas2 = {d['id']: [d,s] for d,s in zip(datas2, scores2)}
id2datas3 = {d['id']: [d,s] for d,s in zip(datas3, scores3)}
final_datas = []
x=0
y = 0
for idx, id in enumerate(id2datas1):
    final_datas.append(id2datas1[id][0])
    if id in id2datas2:
        score1 = id2datas1[id][1]
        score2 = id2datas2[id][1]
        assert id2datas2[id][0]['ori_instruction'] == id2datas1[id][0]['instruction']
        # print(id2datas1[id][0]['instruction'])
        # print(score1)
        # print(id2datas2[id][0]['instruction'])
        # print(score2)
        # print("====================================")
        # input()
        assert id2datas1[id][0]['id'] == id2datas2[id][0]['id'] and id == id2datas1[id][0]['id']
        if score1 < score2:
            final_datas.append(id2datas2[id][0])
            x+=1
            if id in id2datas3:
                score3 = id2datas3[id][1]
                if score3 > score2:
                    final_datas.append(id2datas3[id][0])
                    y += 1
json.dump(final_datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/tag_reduce_new/datas/tag_evol_batch5_round0+1+2/tag_evol_batch5_round0+1+2_inevol.json", 'w'))
print(x,y)
print(len(final_datas))