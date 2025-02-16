import json
import re
from collections import defaultdict
import tqdm
test_data_path="/data/scir/yixuanwang/TagReduce/TagReduce-math-all/math_evaluate/datas/gsm8k.json"
train_data_path="/data/scir/yixuanwang/TagReduce/TagReduce-math-all/tag_reduce_new/datas/tag_evol_tag1_pool_21_20mul3up_alltags_wyxprompt/tag_evol_tag1_pool_21_20mul3up_alltags_wyxprompt.json"
n_gram=8
test_datas = json.load(open(test_data_path))
train_datas = json.load(open(train_data_path))
test_datas = [data['instruction'] for data in test_datas]
train_datas = [data['instruction'] for data in train_datas]
def process_text(text, n_gram):
    text = re.sub("[.,?!:;]","",text)
    text = re.sub("[0-9]+","",text)
    text = text.split(" ")
    while "" in text:
        text.remove("")
    gram_counter=defaultdict(int)
    for i in range(0, max([len(text)-n_gram+1, 1])):
        gram_counter[" ".join(text[i:i+n_gram])] += 1
    return gram_counter

key2num = defaultdict(int)
all_key = set()

for data in test_datas:
    all_key = all_key.union(set(process_text(data, n_gram).keys()))
total = 0
for data in tqdm.tqdm(train_datas):
    counter = process_text(data, n_gram)
    for k in counter:
        if k in all_key:
            # key2num[k] += counter[k]
            total += 1
            break
print(total)

