# 计算CTF相似度
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import cosine_similarity
import os
from tqdm import tqdm
from multiprocessing import Process, Manager
import math
import sys
import random


prompt_no_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_fn(batch, tokenizer, max_length):
    tokens = tokenizer(batch, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    return tokens['input_ids'], tokens['attention_mask']


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.hidden_states[-1]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_texts(rank, model_name, texts, batch_size, max_length, return_dict):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(rank)
    
    print("Model Done!")
    
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer, max_length))
    print("Dataloader Done!")
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(rank), attention_mask.to(rank)
            # print(f"rank:{rank} tokens:{tokenizer.convert_ids_to_tokens(input_ids[0,:1].tolist())}")
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embeddings = mean_pooling(outputs, attention_mask)
            all_embeddings.append(embeddings)
    
    if len(all_embeddings) != 0:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return_dict[rank] = all_embeddings.cpu()
        # print("rank: embedding", rank, all_embeddings[:, :1].tolist())


def calculate_embeddings(texts, save_name=None):
    model_name = "/home/storages/gpu0232/disk3/xzluo/models/dc-1.3b-code-feedback/checkpoint-306"  # 你选择的模型
    batch_size = 32
    max_length = 2048
    world_size = 8
    text_splits = []
    chunk = math.ceil(len(texts) / world_size)
    for i in range(world_size):
        text_splits.append(texts[i * chunk:(i + 1) * chunk])
    print("text_splits", [len(text_splits[i]) for i in range(world_size)])

    manager = Manager()
    return_dict = manager.dict()
    processes = []
    # 启动进程
    for i in range(world_size):
        if len(text_splits[i]) == 0:
            continue
        p = Process(target=encode_texts, args=(i, model_name, text_splits[i], batch_size, max_length, return_dict))
        p.start()
        processes.append(p)

    results = []
    # 收集结果
    for p in processes:
        p.join()

    # 合并结果
    return_keys = sorted(return_dict.keys())
    results = [return_dict[i] for i in return_keys]
    embeddings = torch.cat(results, axis=0)
    # print("Embeddings shape:", embeddings.shape)
    if save_name is not None:
        torch.save(embeddings, f"{save_name}_embeddings.pt")
        return embeddings
    else:
        length = embeddings.shape[0] // 2
        print(f"Embeddings shape: {embeddings.shape}, length: {length}")
        torch.save(embeddings[:length], "ori_insts_embeddings.pt")
        torch.save(embeddings[length:], "ori_outs_embeddings.pt")
        return embeddings[:length], embeddings[length:]
    # print("Embeddings of", save_name, ":", embeddings[:, :1].tolist())
    

def encode_similarity(rank, embeddings, batch_size, return_dict):
    A, B = embeddings
    similarities = []
    with torch.no_grad():
        for i in range(0, len(A), batch_size):
            end = i + batch_size
            similarities.append(cosine_similarity(A[i:end], B[i:end]))
    if len(similarities) != 0:
        similarities = torch.cat(similarities, dim=0)
        return_dict[rank] = similarities
        # print("rank: similarity", rank, similarities.tolist())


def calculate_similarity(A, B, save_name):
    assert A.shape == B.shape
    world_size = 8
    batch_size = 1024
    embeddings_splits = []
    chunk = math.ceil(len(A) / world_size)
    for i in range(world_size):
        embeddings_splits.append((A[i * chunk:(i + 1) * chunk], B[i * chunk:(i + 1) * chunk]))
    print("embedding_splits", [len(embeddings_splits[i][0]) for i in range(world_size)])

    manager = Manager()
    return_dict = manager.dict()
    processes = []
    # 启动进程
    for i in range(world_size):
        if len(embeddings_splits[i][0]) == 0:
            continue
        p = Process(target=encode_similarity, args=(i, embeddings_splits[i], batch_size, return_dict))
        p.start()
        processes.append(p)

    results = []
    # 收集结果
    for p in processes:
        p.join()

    # 合并结果
    return_keys = sorted(return_dict.keys())
    results = [return_dict[i] for i in return_keys]
    similarities = torch.cat(results, axis=0)
    # print("similarities shape:", similarities.shape)
    torch.save(similarities, f"{save_name}_similarity.pt")
    # print(f"{save_name} similarities", similarities)
    return similarities


if __name__ == "__main__":
    files = sys.argv[1]
    ds = json.load(open(f"{files}.json", "r"))
    print("Number of data:", len(ds))
    texts = [prompt_no_input.format(instruction=d["instruction"]) + d["output"] for d in ds]
    for i in range(3):
        example = random.choice(texts)
        print(f"Example {i}:\n", example)
    calculate_embeddings(texts, files)
    # insts = [d["instruction"] for d in ds]
    # outs = [d["output"] for d in ds]
    # ori_insts = [d["ori"]["instruction"] for d in ds]
    # ori_outs = [d["ori"]["output"] for d in ds]
    # texts = [insts, outs, ori_insts, ori_outs]
    # save_names = ["insts", "outs", "ori_insts", "ori_outs"]
    # texts = [ori_insts, ori_outs]
    # save_names = ["ori_insts", "ori_outs"]
    
    embeddings = []
    # for text, save_name in zip(texts, save_names):
        # embeddings.append(calculate_embeddings(text, save_name))
        # print("Embeddings of", save_name, "calculated.")

    # inst_embeddings = torch.load("insts_embeddings.pt")
    # outs_embeddings = torch.load("outs_embeddings.pt")
    # ori_insts_embeddings = torch.load("ori_insts_embeddings.pt")
    # ori_outs_embeddings = torch.load("ori_outs_embeddings.pt")
    # embeddings = [inst_embeddings, outs_embeddings, ori_insts_embeddings, ori_outs_embeddings]
    # insts_similarity = calculate_similarity(embeddings[0], embeddings[2], "insts")
    # print("Similarities of insts calculated.")
    # outs_similarity = calculate_similarity(embeddings[1], embeddings[3], "outs")
    # print("Similarities of outs calculated.")
    # for d, a, b in zip(ds, insts_similarity, outs_similarity):
    #     d["seman_inst_score"] = a.item()
    #     d["seman_out_score"] = b.item()
    # json.dump(ds, open("ctf_gpt4_seman_results.json", "w"), indent=4)
    
    