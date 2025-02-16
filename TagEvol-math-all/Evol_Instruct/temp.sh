#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
# conda activate apr
# python3 evol_instruct.py \
#     --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-7B-Instruct \
#     --source_file /home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/code_alpaca_20k-round0.json \
#     --target_file /home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/code_alpaca_20k-round0_evol.json \
#     --round_num 0 \
#     --temperature 0.7 \
#     --max_tokens 2048 \
#     --use_breadth \
#     --tp 4
# python3 gen_response.py \
#     --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct  \
#     --inst_file /home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/code_alpaca_20k-round0_evol.json
# python3 gen_response.py \
#     --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct  \
#     --inst_file /home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/code_alpaca_20k-roud0-reject-3.json

conda activate llamax
deepspeed --master_port 25125 /home/zhoushiqi/workplace/icaed/code/cot-instruct/train_response.py \
    --model_name_or_path /home/zhoushiqi/workplace/model/deepseek-coder-6.7b \
    --data_path /home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/code_alpaca_20k-round0_evol.json \
    --output_dir /home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/model/code_alpaca_20k-round0_evol \
    --full_determinism True\
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 8 \ \
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
    --deepspeed /home/zhoushiqi/workplace/codectf/config/stage2.json \
    --fp16 True\
    --shuffle True

# deepspeed --master_port 25125 /home/zhoushiqi/workplace/icaed/code/cot-instruct/train_response.py \
#     --model_name_or_path /home/zhoushiqi/workplace/model/deepseek-coder-6.7b \
#     --data_path /home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/code_alpaca_20k-round0_evol.json \
#     --output_dir /home/zhoushiqi/workplace/icaed/code/cot-instruct/tag_reduce/model/code_alpaca_20k-roud0-reject-3 \
#     --full_determinism True\
#     --num_train_epochs 3 \
#     --model_max_length 2048 \
#     --per_device_train_batch_size 8 \ \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --warmup_steps 15\
#     --save_total_limit 300 \
#     --learning_rate 2e-5 \
#     --logging_steps 2 \
#     --lr_scheduler_type "cosine" \
#     --report_to "tensorboard" \
#     --gradient_checkpointing True \
#     --deepspeed /home/zhoushiqi/workplace/codectf/config/stage2.json \
#     --fp16 True\
#     --shuffle True

# cd /home/zhoushiqi/workplace/icaed/code/cot-instruct/code-evaluate/eval
# bash evol_epoch3.sh code_alpaca_20k-round0_evol
# bash evol_epoch3.sh code_alpaca_20k-roud0-reject-3