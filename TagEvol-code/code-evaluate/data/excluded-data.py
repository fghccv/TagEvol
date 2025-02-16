import json
import torch


exclude = json.load(open('excluded-data.json', "r"))
exclude = [d["data"] for d in exclude]
already = {}
for d in exclude:
    if d["instruction"] + d["output"] in already:
        print("Duplicate: ", d)
    already[d["instruction"] + d["output"]] = True

ds0 = json.load(open("ctf_gpt4_all_kcenter_30000.json", "r"))
ds1 = json.load(open("ctf_gpt4_all_kcenter_last_kcenter_72692.json", "r"))
print("start:", len(ds0) + len(ds1))

embeddings0 = torch.load("ctf_gpt4_all_kcenter_30000_embeddings.pt")
embeddings1 = torch.load("ctf_gpt4_all_kcenter_last_kcenter_72692_embeddings.pt")
assert embeddings0.size(0) == len(ds0)
assert embeddings1.size(0) == len(ds1)

res = []
embeddings = []
for i, d in enumerate(ds0):
    if d["instruction"] + d["output"] in already:
        print(f"{i} Already:", json.dumps(d, indent=4))
    else:
        res.append(d)
        embeddings.append(embeddings0[i].unsqueeze(0))

for i, d in enumerate(ds1):
    if d["instruction"] + d["output"] in already:
        print(f"{i} Already:", json.dumps(d, indent=4))
    else:
        res.append(d)
        embeddings.append(embeddings1[i].unsqueeze(0))

print("Total:", len(res))
json.dump(res, open(f"ctf_gpt4_kcenter_{len(res)}.json", "w"), indent=4)
embeddings = torch.cat(embeddings, dim=0)
print(embeddings.size())
torch.save(embeddings, f"ctf_gpt4_kcenter_{len(res)}_embeddings.pt")
