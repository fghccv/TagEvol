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
# conda activate apr

data_name=tag_reduce_newprompt2_newbatch_batch5_nokey
# python3 gen_response.py \
#     --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct  \
#     --inst_file $data_name.json
conda activate train

deepspeed --master_port 25125 ../train_response.py \
    --model_name_or_path /home/zhoushiqi/workplace/model/Meta-Llama-3-8B_ms \
    --data_path ./datas/${data_name}.json \
    --output_dir ./models/llama3/${data_name} \
    --full_determinism True\
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
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
    --deepspeed ../train_config/stage2.json \
    --fp16 True\
    --shuffle True

cd ../code-evaluate/eval
bash eval_epoch3.sh $data_name