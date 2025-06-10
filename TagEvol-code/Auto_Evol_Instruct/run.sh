#!/bin/bash


data_name="code_alpaca_20k"
model_path=""
for i in {1..3}
do
  echo "Running iteration: $i"
  
  target_data_name="Auto_evol-round${i}"

  python3 auto_evol_instruct.py \
    --model_name_or_path $model_path \
    --source_file "./datas/${data_name}.json" \
    --target_file "./datas/${target_data_name}.json" \
    --temperature 0.7 \
    --max_tokens 2048 \
    --tp 8 \
    --debug 

  python3 gen_response.py \
    --model_name_or_path $model_path  \
    --inst_file "./datas/${target_data_name}.json"

  data_name=${target_data_name}  # 更新 data_name 以用于下一轮
done

