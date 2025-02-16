#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"


for batch in $(seq 17 2 17)
do
echo $batch
data_name=code_alpaca_20k_random-uncoupled_shot0_new_prompt2_batch${batch}_newbatch
echo $data_name
pwd
conda activate apr
python3 tag_random_uncoupled_clean_newprompt_newbatch.py \
    --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-7B-Instruct \
    --source_file ./datas/code_alpaca_20k.jsonl \
    --target_file ./datas/${data_name}.json \
    --temperature 0.7 \
    --max_tokens 2048 \
    --num_shot 0 \
    --batch ${batch} \
    --budget 20000 \
    --tp 4

python3 gen_response.py \
    --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-72B-Instruct  \
    --inst_file ./datas/${data_name}.json

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
bash eval_epoch3.sh ${data_name}
cd ../../tag_reduce
done