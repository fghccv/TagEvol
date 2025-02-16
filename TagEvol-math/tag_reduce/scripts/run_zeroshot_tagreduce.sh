#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr



echo instruct generation
batch=5
alpha=1
data_name=gsm8k_7k_tagreduce_batch${batch}_alpha${alpha}
python3 tag_reduce.py \
    --model_name_or_path /home/guanjiannan/share/models/Qwen--Qwen2.5-7B-Instruct \
    --source_file ./datas/gsm8k_7k.json \
    --target_file ./datas/${data_name}.json \
    --temperature 0.7 \
    --max_tokens 2048 \
    --num_shot 0 \
    --batch ${batch} \
    --budget 7500 \
    --tp 4 \
    --alpha ${alpha} \
    # --debug

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
    
cd ../../math_evaluate
bash evaluate.sh ${data_name} tag_reduce