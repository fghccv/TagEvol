#!/bin/bash


data_name="code_alpaca_20k"

for i in {1..3}
do
  echo "Running iteration: $i"
  
  target_data_name="Auto_evol-round${i}_initial"

  python3 auto_evol_instruct.py \
    --model_name_or_path /data/scir/yixuanwang/models/qwen2.5-72b-ins \
    --source_file "./datas/${data_name}.json" \
    --target_file "./datas/${target_data_name}.json" \
    --temperature 0.7 \
    --max_tokens 2048 \
    --tp 8 \
    --debug 

  python3 gen_response.py \
    --model_name_or_path /data/scir/yixuanwang/models/qwen2.5-72b-ins  \
    --inst_file "./datas/${target_data_name}.json"

  data_name=${target_data_name}  # 更新 data_name 以用于下一轮
done

确保 merge_datas.py 用 python3 运行
python3 merge_datas.py  --data1_path ./datas/Auto_evol-round1_initial.json --data2_path ./datas/Auto_evol-round2_initial.json --target_path ./datas/Auto_evol-round12_initial.json
python3 merge_datas.py  --data1_path ./datas/Auto_evol-round12_initial.json --data2_path ./datas/Auto_evol-round3_initial.json --target_path ./datas/Auto_evol-round123_initial.json

# 训练阶段
all_data_names=("Auto_evol-round1_initial" "Auto_evol-round12_initial" "Auto_evol-round123_initial")
for name in "${all_data_names[@]}"
do
  bash train.sh "$name"
done
