#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr
data_path=/home/zhoushiqi/workplace/icaed/code/cot-instruct/Evol_Instruct/code_alpaca_20k.jsonl
for i in {0..3}
do
  echo $i
  target_data_path=/home/zhoushiqi/workplace/icaed/code/cot-instruct/Evol_Instruct/code_alpaca_20k-evol-roud${i}.json
  python3 evol_instruct.py \
    --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-7B-Instruct \
    --source_file ${data_path} \
    --target_file ${target_data_path} \
    --round_num $i \
    --temperature 0.7 \
    --max_tokens 2048 \
    --use_breadth \
    --tp 4
  python3 gen_response.py \
    --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct  \
    --inst_file ${target_data_path}
  data_path=${target_data_path}
done