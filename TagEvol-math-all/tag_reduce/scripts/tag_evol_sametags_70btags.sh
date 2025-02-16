#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"




echo instruct generation


batch=5
data_name=tag_evol_batch${batch}_roud2_sametags_70btags
for roud in {3..3}
do
target_data_name=tag_evol_batch${batch}_roud${roud}_sametags_70btags
conda activate apr
python3 tag_evol_sametags_70btags.py \
    --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct \
    --source_file ./datas/${data_name}.json \
    --target_file ./datas/${target_data_name}.json \
    --temperature 0.7 \
    --max_tokens 2048 \
    --batch ${batch} \
    --tp 4 \
    --tags_file ./datas/gsm8k_7k.json \
    --roud ${roud}

python3 gen_response.py \
    --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct  \
    --inst_file ./datas/${target_data_name}.json

conda activate train
rm -rf ./models/llama3/${target_data_name}
deepspeed --master_port 25125 ../train_response.py \
  --model_name_or_path /home/zhoushiqi/workplace/model/Meta-Llama-3-8B_ms \
  --data_path ./datas/${target_data_name}.json \
  --output_dir ./models/llama3/${target_data_name} \
  --full_determinism True\
  --num_train_epochs 3 \
  --model_max_length 2048 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --warmup_steps 15\
  --save_total_limit 300 \
  --learning_rate 2e-5 \
  --logging_steps 2 \
  --lr_scheduler_type "cosine" \
  --report_to "tensorboard" \
  --gradient_checkpointing True \
  --deepspeed ../train_config/stage2.json \
  --fp16 True\
  --shuffle True

cd ../math_evaluate
bash evaluate.sh ${target_data_name} tag_reduce
cd ../tag_reduce

data_name=${target_data_name}
done