#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"




echo instruct generation


batch=5
tag_name=gsm8k_7k
for round in {3..3}
do
target_data_name=tag_evol_batch${batch}_round${round}


conda activate train
# rm -rf ./models/llama3/${target_data_name}
deepspeed --master_port 25125 ../train_response.py \
  --model_name_or_path /home/zhoushiqi/workplace/model/Meta-Llama-3-8B_ms \
  --data_path ./datas/${target_data_name}/${target_data_name}.json \
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
bash evaluate.sh ${target_data_name} tag_reduce_new
cd ../tag_reduce

data_name=${target_data_name}
done