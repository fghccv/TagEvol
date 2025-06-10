#!/bin/bash




data_name=$1
model_path=""
model_name=$(basename "$model_path")
deepspeed --master_port 25125 ../train_response.py \
  --model_name_or_path $model_path \
  --data_path ./datas/final_datas/${data_name}.json \
  --output_dir ./models/${model_name}/${data_name} \
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
