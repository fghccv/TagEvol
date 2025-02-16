import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
import pprint
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
parser.add_argument('--output_path', type=str, help="")
parser.add_argument('--data_path', type=str, help="")
parser.add_argument('--gpu', type=int, default=0, help="")

args = parser.parse_args()
argsdict = vars(args)
print(pprint.pformat(argsdict))
# 初始化Llama模型和tokenizer
model_name = args.model  # 替换为你使用的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token为eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
model.eval()  # 设为评估模式

# 示例json数据结构
data = json.load(open(args.data_path))
gpu = args.gpu
gap = len(data)//8 + 1
print(f"start:{gpu*gap},end:{(gpu+1)*gap}")
data = data[gpu*gap:(gpu+1)*gap]
batch_size = 1  # 你可以根据实际情况调整batch_size的大小

IFD = []
IFDX = []
INST_LOSS = []
num_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(data))
    batch_data = data[start_idx:end_idx]

    model_inst_batch = []
    inst_batch = []
    output_text_batch = []
    for d in batch_data:
        if 'input' in d and d['input']!='':
            instruction = PROMPT_DICT['prompt_input'].format_map(dict(instruction=d["instruction"],input=d['input']))
        else:
            instruction = PROMPT_DICT['prompt_no_input'].format_map(dict(instruction=d["instruction"]))
        # instruction = tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)
        input_text = ""
        
        output_text = d["output"]


        model_inst = instruction

        all_input = model_inst + "\n" + output_text

        model_inst_batch.append(all_input)
        inst_batch.append(model_inst)
        output_text_batch.append(output_text)

    input_ids_batch = tokenizer(model_inst_batch, return_tensors="pt", padding=True).input_ids
    output_ids_batch = tokenizer(output_text_batch, return_tensors="pt", padding=True).input_ids
    instruction_and_input_lengths = [len(tokenizer(m, return_tensors="pt").input_ids[0]) for m in inst_batch]

    input_ids_batch = input_ids_batch.cuda()
    output_ids_batch = output_ids_batch.cuda()
    label_ids_batch = input_ids_batch.clone()
    inst_ids_batch = input_ids_batch.clone()
    for i in range(len(batch_data)):
        label_ids_batch[i, :instruction_and_input_lengths[i]] = -100
        inst_ids_batch[i, instruction_and_input_lengths[i]:] = -100

    with torch.no_grad():
        # import pdb
        # pdb.set_trace()
        inst_loss = model(input_ids=input_ids_batch, labels=inst_ids_batch).loss
        output_only_loss = model(input_ids=output_ids_batch, labels=output_ids_batch).loss
        output_loss = model(input_ids=input_ids_batch, labels=label_ids_batch).loss
        INST_LOSS.append(inst_loss.item())
        IFD.append(output_loss.item() / output_only_loss.item())
        IFDX.append(output_loss.item() / (inst_loss.item() * output_only_loss.item()))
        # IFDX.append(output_loss.item() * inst_loss.item() / output_only_loss.item())

ifd_score = sum(IFD) / len(IFD)
ifdx_score = sum(IFDX) / len(IFDX)
inst_loss = sum(INST_LOSS) / len(INST_LOSS)
print({
                "base":gpu*gap,
                "ifd_score": ifd_score,
                "ifdx_score": ifdx_score,
                "inst_loss": inst_loss, 
                })
with open(args.output_path, "w", encoding="utf-8") as f:
    json.dump([{
                "base":gpu*gap,
                "ifd_score": IFD,
                "ifdx_score": IFDX,
                "inst_loss": INST_LOSS, 
                }], f, indent=4, ensure_ascii=False)