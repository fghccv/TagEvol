#!/bin/bash
target_data_name=all_math_15k
python3 gen_response.py \
    --model_name_or_path /data/scir/yixuanwang/models/qwen2.5-72b-ins  \
    --inst_file ./datas/${target_data_name}/${target_data_name}.json

cp ./datas/$target_data_name/$target_data_name.json ./datas/final_datas/$target_data_name.json

bash scripts/train.sh $target_data_name
