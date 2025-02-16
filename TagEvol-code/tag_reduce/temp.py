import json
datas1 = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce/datas/tag_evol_batch5_roud0.json"))
# oridatas = [json.loads(l) for l in open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce/datas/code_alpaca_20k.jsonl")]
# datas2=json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-code/tag_reduce/datas/code_alpaca_20k_zeroshot_oritags.json"))
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("/home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct")
total = 0
all_ln = []
new_datas = []
for i,data in enumerate(datas1):
    instruction = data['instruction']
    # if len(tokenizer.tokenize(data['output'])) < 2040:
    #     new_datas.append(data)
    #     continue
    # data['tags'] = json.dumps(data['tags'])
    print(data['instruction'])
    print(data['input'])
    # print('***********')
    # print(oridatas[i]['instruction'])
    # print("***********")
    # print(datas2[i]['instruction'])
    # print(datas2[i]['input'])
    # print(len(instruction)+len(data['input']))
    print('---------------')
    # print(data['tags'])
    # print('---------------')
    # print(data['output'])
    # print("====================")
    # tags = j
    # son.loads(data['tags'])
    # tags = data['tags']
    # temp_ln = 0
    # for k in tags:
    #     temp_ln += len(tags[k])
    # all_ln.append(temp_ln)
    input()
    total += 1
print(total)
print(sum(all_ln)/len(all_ln))
# json.dump(new_datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce/tag_reduce/datas/code_alpaca_20k_zeroshot_oritags.json", 'w'))
# # #     input()
# ln = [len(data['instruction'])+len(data['input']) for data in datas1]
# print(sum(ln)/len(ln))

# # # # def calculate_variance(data):
# # # #     n = len(data)
# # # #     if n == 0:
# # # #         return 0  # 如果列表为空，返回方差为0
# # # #     mean = sum(data) / n  # 计算平均值
# # # #     variance = sum((x - mean) ** 2 for x in data) / n  # 计算方差
# # # #     return variance

# # # # # 示例数据
# # # import json
# # # # data = list(json.load(open("/home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/code_alpaca_20k_random.json.history_tags.json")).values())
# # # # while 0 in data:
# # # #     data.remove(0)
# # # # variance = calculate_variance(data)
# # # # print("方差是:", variance)

# # # # datas = json.load(open("/home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/code_alpaca_20k_random_roud1.json.history_tags.json"))
# # # # datas = {key:datas[key] for key in datas if datas[key]!=0}
# # # # print(len(datas))

# # # # x=[1,1,1,1,1,2]
# # # # print(all(xx==1 for xx in x))
# # # import json
# # # datas = json.load(open("/home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/datas/code_alpaca_20k.jsonl.tag_datas.json"))
# # # all_ln = []
# # # for k in datas:
# # #     tags = datas[k]['tags']
# # #     all_v = []
# # #     for k in tags:
# # #         all_v += tags[k]
# # #     all_ln.append(len(all_v))

# # # print(sum(all_ln)/len(all_ln))

# # import json
# # datas = json.load(open("/home/zhoushiqi/workplace/icaed/code/cot-instruct/code-evaluate/preds_humaneval/llama3/tag_reduce/code_alpaca_20k_random-roud0-uncoupled_shot0/checkpoint-117/T0_N1_eval_results.json"))
# # for i in range(1, 165): 
# #     print(datas['eval'][f'HumanEval/{i}'][0]['solution'])
# #     input()

# import json

# datas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce/tag_reduce/datas/code_alpaca_20k_random-roud0-uncoupled_shot0.json"))
# scores = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce/tag_reduce/analys_datas/code_alpaca_20k_random-roud0-uncoupled_shot0.json"))['ifdx_score']
# inst_loss = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce/tag_reduce/analys_datas/code_alpaca_20k_random-roud0-uncoupled_shot0.json"))['inst_loss']
# orders = sorted(list(range(len(datas))), key=lambda id:inst_loss[id], reverse=True)
# for id in orders:
#     print(datas[id]['instruction'])
#     print("--------")
#     print(datas[id]['tags'])
#     print(f"len:{len(datas[id]['instruction'])}, score:{scores[id]}, isnt_loss:{inst_loss[id]}")
#     print("=======")
#     input()
