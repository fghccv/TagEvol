from multiprocessing import cpu_count
import json
from tqdm import tqdm
import argparse
from generate_utils import read_from_jsonl, stream_jsonl
from typing import List

def eval_ds1000(answers: List[str]):
    import concurrent.futures as cfuts
    import ds1000_execution 
    import pandas as pd
    ds1000_results = []
    ds1000 = [json.loads(d) for d in open("../dataset/ds1000.jsonl", "r").readlines()]
    with cfuts.ProcessPoolExecutor(
        max_workers=cpu_count()
    ) as executor:
        futs = []
        for p in ds1000:
            id = int(p['metadata']['problem_id'])
            lib = p['metadata']['library']
            test_program = (
                p['code_context'] + '\n'
                + f'code = {repr(answers[id])}\n'
                + 'test_execution(code)\n'
                + ('test_string(code)\n'  if 'test_string(' in p['code_context']  else '\n')
            )
            # you have some options on how to actually execute the program here.
            futs.append(executor.submit(ds1000_execution.check_correctness, test_program, timeout=120, completion_id=id))

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            result = f.result()
            cid = result['completion_id']
            result['score'] = 1 if result['passed'] else 0
            result['library'] = ds1000[cid]['metadata']['library']
            result['perturbation_type'] = ds1000[cid]['metadata']['perturbation_type']
            ds1000_results.append(result)

    df_res = pd.DataFrame.from_records(ds1000_results)
    pd.set_option('display.precision', 3)
    summary = df_res.agg({'score': ['count', 'mean']}).to_string()
    summary += '\n' + df_res[['library', 'score']].groupby('library').agg({'score': ['count', 'mean']}).to_string()
    summary += '\n' + df_res[['perturbation_type', 'score']].groupby('perturbation_type').agg({'score': ['count', 'mean']}).to_string()
    return summary, ds1000_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Inputs
    parser.add_argument('--path', type=str, default="/Users/s1mple/Projects/codespec/MakeSomeNoise/codes/T0_N1.jsonl", help="")
    parser.add_argument('--dataset', type=str, default="ds1000", help="")
    parser.add_argument('--N', type=int, default=1, help='')
    # args of evalplus
    parser.add_argument("--i-just-wanna-run", action="store_true")
    parser.add_argument("--test-details", action="store_true")
    parser.add_argument("--min-time-limit", default=1, type=float)
    parser.add_argument("--gt-time-limit-factor", default=4.0, type=float)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument(
        "--noextreme", action="store_true", help="Omit extreme test inputs"
    )
    parser.add_argument(
        "--version", default="default", type=str, help="Version of the dataset"
    )
    args = parser.parse_args()

    outputs = [d for d in stream_jsonl(args.path)]
    problems = read_from_jsonl(args.dataset)

    assert len(problems) * args.N == len(outputs) 
    if args.dataset in ["mbpp", "humaneval"]:
        from evalplus.evaluate import evaluate
        args.parallel = cpu_count()
        args.samples = args.path
        evaluate(args)
    elif args.dataset == "ds1000":
        outputs = sorted(outputs, key=lambda x: x["task_id"])
        answers = [output["solution"] for output in outputs]
        summary, ds1000_results = eval_ds1000(answers)
        print(summary)
        json.dump(ds1000_results, open(args.path.replace(".jsonl", "-results.json"), "w"), indent=4)
        
