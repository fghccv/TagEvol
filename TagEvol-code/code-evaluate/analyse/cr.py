import os
import re
from collections import defaultdict
import json
needs = [
    "ds_icondition_k_1m_sample_1_temp_1_add_ori-response_qwen32b",
    "ds-magic-instruct-1m_add_ori-response_qwen32b"
]
dirs = {
    "humaneval":"/home/zhoushiqi/zsq_lib/CTFCoder/preds_humaneval",
    "mbpp":"/home/zhoushiqi/zsq_lib/CTFCoder/preds_mbpp",
    "lcb":"/home/zhoushiqi/zsq_lib/CTFCoder/eval/LiveCodeBench/output"
}

results = {}
for need in needs:
    results[need] = defaultdict(dict)
    for k in dirs:
        dir = dirs[k]
        use_dirs = list(filter(lambda x:x.startswith(need), os.listdir(dir)))
        if k != 'lcb':
            for it in use_dirs:
                ck = it.split('-')[-1]

                if os.path.exists(f"{dir}/{it}/T0_N1_pass@k.txt"):
                    txt = open(f"{dir}/{it}/T0_N1_pass@k.txt").read()
                    scores = re.findall("pass@1:	(.+)\n?", txt)
                    assert len(scores) == 2
                    results[need][ck][k] = scores
        else:
            for it in use_dirs:
                ck = it.split('-')[-1]
                if os.path.exists(f"{dir}/{it}/Scenario.codegeneration_10_0.2_eval.json"):
                    scores = json.load(open(f"{dir}/{it}/Scenario.codegeneration_10_0.2_eval.json"))
                    score = scores[0]["pass@1"]
                results[need][ck][k] = score

for need in needs:
    with open(f"/home/zhoushiqi/zsq_lib/CTFCoder/analyse/{need}.txt",'w') as f:
        cks = sorted(int(x) if x!='final' else 100000 for x in list(results[need].keys()))
        for ck in cks:
            ck = str(ck)
            if ck == '100000':
                ck = 'final'
            f.write(str(ck) + '\t')
            for k in dirs:
                if k not in results[need][ck]:
                    f.write('\t\t\t\t\t')
                    continue
                f.write(json.dumps(results[need][ck][k]) + '\t')
            f.write('\n')

                