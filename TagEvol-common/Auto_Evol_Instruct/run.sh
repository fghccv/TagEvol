#!/bin/bash

data_name=databricks-dolly-15k
model_path=""

for i in {3..3}
do
  # echo $i
  # conda activate apr
  target_data_name=Auto_evol-roud${i}
  python3 auto_evol_instruct.py \
    --model_name_or_path ${model_path} \
    --source_file ./datas/${data_name}.json \
    --target_file ./datas/${target_data_name}.json \
    --temperature 0.7 \
    --max_tokens 2048 \
    --tp 4 \
    
  python3 gen_response.py \
    --model_name_or_path $model_path  \
    --inst_file ./datas/${target_data_name}.json

