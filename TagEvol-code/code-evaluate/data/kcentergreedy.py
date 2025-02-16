from scipy.spatial import distance
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import argparse
import torch
import json


def compute_distances(chunk, centers):
    dist_matrix = distance.cdist(centers, chunk, 'euclidean')
    min_dists = np.min(dist_matrix, axis=0)
    return min_dists


def k_center_greedy_parallel(old, new, new_data, k, n_processors):
    n_samples = len(new)
    new_ds = []
    new_dist = []
    new_embeddings = []
    pool = mp.Pool(processes=n_processors)

    # 初始化距离矩阵
    dists = np.full(n_samples, np.inf)
    remaining_indices = np.arange(n_samples)
    
    for _ in tqdm(range(k)):
        # Divide data into chunks for each processor
        chunk_size = int(np.ceil(len(remaining_indices) / n_processors))
        chunks = [remaining_indices[i * chunk_size:min((i + 1) * chunk_size, len(remaining_indices))] for i in range(n_processors)]
        
        # 使用Pool的map方法并行计算距离并更新最小距离
        results = pool.starmap(compute_distances, [(new[chunk], old) for chunk in chunks])
        results = np.concatenate(results)
        dists[remaining_indices] = np.minimum(dists[remaining_indices], results)
        # print(results)
        # print(dists[remaining_indices])

        # 找到距离最大的点作为下一个中心
        cur_dist = np.max(dists[remaining_indices])
        if cur_dist == 0.0:
            print("now dist is 0")
            break 
        next_index_local = np.argmax(dists[remaining_indices])
        next_index = remaining_indices[next_index_local]
        old = new[next_index].reshape(1, -1)
        new_ds.append(new_data[next_index])
        new_dist.append(dists[next_index])
        new_embeddings.append(new[next_index].unsqueeze(0))
        
        # 更新remaining_indices
        remaining_indices = np.delete(remaining_indices, next_index_local)

        if len(new_ds) % 10000 == 0:
            json.dump(new_ds, open(f"kcenter_middle_{len(new_ds)}.json", "w"))

    return new_ds, new_dist, new_embeddings


def extract_emdedding():
    ds = json.load(open("ctf_gpt4_all_kcenter_30000.json", "r"))
    new = torch.load("ctf_gpt4_all_embeddings.pt")
    new_data = json.load(open("ctf_gpt4_all.json", "r"))
    already = {}
    for d, e in zip(new_data, new):
        if d["instruction"]+d["output"] in already:
            raise ValueError(f"Duplicate : {d['instruction']+d['output']}")
        already[d["instruction"]+d["output"]] = e
    embeddings = []
    for d in ds:
        embeddings.append(already[d["instruction"]+d["output"]].unsqueeze(0))
    # print(embeddings[2].size())
    embeddings = torch.cat(embeddings, dim=0)
    print(embeddings.size())
    torch.save(embeddings, "ctf_gpt4_all_kcenter_30000_embeddings.pt")


def last_embedding():
    ds = json.load(open("ctf_gpt4_all_kcenter_30000.json", "r"))
    already = {}
    for d in ds:
        if d["instruction"]+d["output"] in already:
            raise ValueError(f"Duplicate : {d['instruction']+d['output']}")
        already[d["instruction"]+d["output"]] = 1
    
    new = torch.load("ctf_gpt4_all_embeddings.pt")
    print("all data has:", len(new))
    new_data = json.load(open("ctf_gpt4_all.json", "r"))
    embeddings = []
    last_ds = []
    for d, e in zip(new_data, new):
        if d["instruction"]+d["output"] not in already:
            last_ds.append(d)
            embeddings.append(e.unsqueeze(0))
    embeddings = torch.cat(embeddings, dim=0)
    print("last ds has:", len(last_ds))
    print("already ds:", len(new) - len(last_ds))
    torch.save(embeddings, "ctf_gpt4_all_kcenter_last_embeddings.pt")
    json.dump(last_ds, open("ctf_gpt4_all_kcenter_last.json", "w"))


def judge_oss():
    ds = json.load(open("oss_ctf.json", "r"))
    ds2 = json.load(open("oss_instruct.json", "r"))
    print("oss_ctf len:", len(ds))
    print("oss len:", len(ds2))
    for d, d2 in zip(ds, ds2):
        if d["instruction"] != d2["instruction"] or d["output"] != d2["output"]:
            print("Different")
            print(d["instruction"], d2["instruction"])
            print(d["output"], d2["output"])
            break
    print("Same!")


def split_oss():
    ds = json.load(open("oss_ctf.json", "r"))
    ds2 = json.load(open("oss_instruct.json", "r"))
    embeddings = torch.load("oss_ctf_embeddings.pt")
    oss_instruct_embeddings = embeddings[:len(ds2)]
    ctf_oss_embeddings = embeddings[len(ds2):]
    ctf_oss = ds[len(ds2):]
    torch.save(oss_instruct_embeddings, "oss_instruct_embeddings.pt")
    torch.save(ctf_oss_embeddings, "ctf_oss_embeddings.pt")
    json.dump(ctf_oss, open("ctf_oss.json", "w"))
    print("all len", len(ds))
    print("all embeddings size", embeddings.size())
    print("oss len", len(ds2))
    print("oss embeddings size", oss_instruct_embeddings.size())
    print("ctf len", len(ds2))
    print("ctf embeddings size", ctf_oss_embeddings.size())


def merge_embedding():
    old = torch.load("evol_instruct_embeddings.pt")
    new = torch.load("ctf_gpt4_all_kcenter_30000_embeddings.pt")
    embeddings = torch.cat([old, new], dim=0)
    torch.save(embeddings, "evol_instruct_ctf_gpt4_all_kcenter_30000_embeddings.pt")


def process_embedding():
    import numpy as np
    from collections import Counter
    array = json.load(open("code_feedback_kcenter_100000_dist.json", "r"))
    
    # Counting numbers > 4
    greater_than_4_count = sum(x > 4 for x in array)

    # Counting numbers from 0 to 4 in increments of 0.5
    bins = np.arange(0, 4.5, 0.5)
    binned_counts = Counter(np.digitize(array, bins) - 1)

    # Preparing the formatted output
    output = f"Count of numbers > 4: {greater_than_4_count}\n"
    output += "Counts for 0-4 with 0.5 increments:\n"

    for i in range(len(bins)):
        count = binned_counts[i] if i in binned_counts else 0
        output += f"{bins[i]:.1f} - {bins[i] + 0.5:.1f}: {count}\n"
    
    print(output)


if __name__ == "__main__":
    # last_embedding()
    # judge_oss()
    # split_oss()
    # process_embedding()
    # extract_emdedding()
    # merge_embedding()
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=str, default="evol_instruct")
    parser.add_argument('--new', type=str, default="ctf_gpt4_all")
    parser.add_argument('--k', type=int, default=30000)
    args = parser.parse_args()
    old = torch.load(args.old+"_embeddings.pt")
    print(f"Size of {args.old} embeddings:", old.size())
    new = torch.load(args.new+"_embeddings.pt")
    print(f"Size of {args.new} embeddings:", new.size())
    new_data = json.load(open(args.new+".json", "r"))
    n_processors = mp.cpu_count()
    if args.k == 0:
        args.k = len(new_data)
    new_ds, new_dist, new_embeddings = k_center_greedy_parallel(old, new, new_data, min(args.k, len(new_data)), n_processors)
    new_embeddings = torch.cat(new_embeddings, dim=0)
    json.dump(new_ds, open(f"{args.new}_kcenter_{args.k}.json", "w"))
    json.dump(new_dist, open(f"{args.new}_kcenter_{args.k}_dist.json", "w"))
    torch.save(new_embeddings, f"{args.new}_kcenter_{args.k}_embeddings.pt")
    print(f"Size of {args.new} kcenter embeddings:", new_embeddings.size())