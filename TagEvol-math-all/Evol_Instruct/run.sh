#!/bin/bash


data_name="all_math_15k"
mode_path=""
for i in {1..1}
do
  echo "Running iteration: $i"
  
  target_data_name="Evol-round${i}_7bins"

  python3 evol_instruct.py \
    --model_name_or_path ${model_path} \
    --source_file "./datas/${data_name}.json" \
    --target_file "./datas/${target_data_name}.json" \
    --temperature 0.7 \
    --max_tokens 2048 \
    --use_breadth \
    --tp 4 \
    --round_num $((i+1)) 

  python3 gen_response.py \
    --model_name_or_path ${model_path}  \
    --inst_file "./datas/${target_data_name}.json"

  data_name=${target_data_name}  # 更新 data_name 以用于下一轮
done

