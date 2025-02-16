import json
datas = [json.loads(l) for l in open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/math_evaluate/datas/math.jsonl")]
for data in datas:
    data['instruction'] = data['problem']
    data['output'] = data['solution']
json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/math_evaluate/datas/math.json", 'w'))
