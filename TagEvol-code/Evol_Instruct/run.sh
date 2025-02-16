#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr  # 只需要激活一次

data_name="Evol-round1"

# for i in {2..3}
# do
#   echo "Running iteration: $i"
  
#   target_data_name="Evol-round${i}"

#   python3 evol_instruct.py \
#     --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct \
#     --source_file "./datas/${data_name}.json" \
#     --target_file "./datas/${target_data_name}.json" \
#     --temperature 0.7 \
#     --max_tokens 2048 \
#     --use_breadth \
#     --tp 4 \
#     --round_num $i 

#   python3 gen_response.py \
#     --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct  \
#     --inst_file "./datas/${target_data_name}.json"

#   data_name=${target_data_name}  # 更新 data_name 以用于下一轮
# done

# 确保 merge_datas.py 用 python3 运行
python3 merge_datas.py  --data1_path ./datas/Evol-round1.json --data2_path ./datas/Evol-round2.json --target_path ./datas/Evol-round12.json
python3 merge_datas.py  --data1_path ./datas/Evol-round12.json --data2_path ./datas/Evol-round3.json --target_path ./datas/Evol-round123.json

# 训练阶段
all_data_names=("Evol-round1" "Evol-round12" "Evol-round123")
for name in "${all_data_names[@]}"
do
  bash train_func.sh "$name"
done
