#!/bin/bash
#SBATCH -J llama2-13b-greedy                             # 作业名为 test
#SBATCH -o /home/zhoushiqi/workplace/codectf/script/log/train4.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -e /home/zhoushiqi/workplace/codectf/script/log/train4.err
#SBATCH -p hit                          # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 20:00:00                            # 任务运行的最长时间为 1 小时                          
#SBATCH --gres=gpu:8
#SBATCH --mem 500GB
#SBATCH -c 128

# # 设置运行环境
# source activate
# conda activate llamax

# 输入要执行的命令，例如 ./hello 或 python test.py 等
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate llamax

deepspeed --master_port 25125 /home/zhoushiqi/workplace/icaed/code/cot-instruct/train_response.py \
    --model_name_or_path /home/zhoushiqi/workplace/model/Meta-Llama-3-8B \
    --data_path /home/zhoushiqi/workplace/icaed/code/cot-instruct/Auto_Evol_Instruct/code_alpaca_20k-evol-roud012.json \
    --output_dir /home/zhoushiqi/workplace/icaed/code/cot-instruct/Auto_Evol_Instruct/model/llama3/code_alpaca_20k-evol-roud012 \
    --full_determinism True\
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \ \
    --gradient_accumulation_steps 16 \
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