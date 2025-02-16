import json
need = [
    ["humaneval", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_1w-checkpoint-473"],
    ["humaneval", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_1w-checkpoint-710"],
    ["mbpp", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_1w-checkpoint-473"],
    ["mbpp", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_1w-checkpoint-710"],
    ["humaneval", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_2w-checkpoint-768"],
    ["humaneval", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_2w-checkpoint-1024"],
    ["mbpp", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_2w-checkpoint-768"],
    ["mbpp", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_2w-checkpoint-1024"],
    ["humaneval", "dc-6.7b-evol-checkpoint-651"],
    ["humaneval", "dc-6.7b-evol-checkpoint-868"],
    ["mbpp", "dc-6.7b-evol-checkpoint-651"],
    ["mbpp", "dc-6.7b-evol-checkpoint-868"],
    ["humaneval", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_30000-checkpoint-551"],
    ["humaneval", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_30000-checkpoint-825"],
    ["mbpp", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_30000-checkpoint-551"],
    ["mbpp", "dc-6.7b-evol_instruct_ctf_gpt4_all_kcenter_30000-checkpoint-825"],
    ["mbpp", "dc-6.7b-evol_instruct_random_ctf_evol_30000-checkpoint-551"],
    ["humaneval", "dc-6.7b-evol+ctf_seman_0.9_-0.1-checkpoint-816"],
    ["mbpp", "dc-6.7b-evol+ctf_seman_0.9_-0.1-checkpoint-1088"],
]

def tranform():
    for task, name in need:
        file = f"../preds_{task}/{name}/T0_N1_eval_results.json"
        ds = json.load(open(file, "r"))
        file = f"results/{name}_{task}.json"
        json.dump(ds, open(file, "w"), indent=4)

def merge():
    from collections import defaultdict
    import pandas as pd
    res = defaultdict(list)
    for task, name in need:
        file = f"results/{name}_{task}.json"
        ds = json.load(open(file, "r"))
        ds = ds["eval"]
        res[task].append(ds)
    for task, ds_lst in res.items():
        keys = ds_lst[0].keys()
        keys = sorted(keys, key=lambda x:int(x.split("/")[-1]))
        rows = defaultdict(list)
        for k in keys:
            pass_cnt = 0
            for ds in ds_lst:
                base_status = ds[k][0]["base_status"] if ds[k][0]["base_status"] == "pass" else str(ds[k][0]["base_fail_tests"])
                plus_status = ds[k][0]["plus_status"] if ds[k][0]["plus_status"] == "pass" else str(ds[k][0]["plus_fail_tests"]) 
                pass_cnt += int(plus_status == "pass")
                solution = ds[k][0]["solution"]
                rows[k].append(f"Base: {base_status}\nPlus: {plus_status}\n{solution}")
            if pass_cnt == len(ds_lst):
                del rows[k]
        columns = [n for t, n in need if t==task]
        df = pd.DataFrame.from_dict(rows, orient='index', columns=columns)
        excel_path = f"{task}.xlsx"
        df.to_excel(excel_path)

# merge()
def tomd(task, id):
    import pandas as pd
    id = f"HumanEval/{id}" if task == "humaneval" else f"MBPP/{id}"
    df = pd.read_excel(f"{task}.xlsx", index_col=0)
    infos = df.loc[id]
    columns = df.columns
    assert len(infos) == len(columns)
    writer = open(f"{id.replace('/', '_')}.md", "w")
    for info, column in zip(infos, columns):
        print(f"## {column}\n\n", file=writer)
        print(f"```python\n{info}\n```\n\n", file=writer)
    writer.close()


tomd("humaneval", 76)

