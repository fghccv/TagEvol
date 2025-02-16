#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr

python3 self_instruct.py \
    --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-7B-Instruct \
    --source_file code_alpaca_20k.jsonl \
    --target_file code_alpaca_20k_selfinstruct.json\
    --temperature 0.7 \
    --max_tokens 2048 \
    --num_shot 5 \
    --budget 20000 \
    --tp 4

# python3 gen_response.py \
#     --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct  \
#     --inst_file code_alpaca_20k_selfinstruct.json