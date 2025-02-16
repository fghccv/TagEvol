#!/bin/bash


export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 定义模型检查点和数据集数组
model=$1
method=$2
base_model=$3

model_path=../${method}/models/$base_model/$1
checkpoints=($(ls -d ${model_path}/*/ | grep -v 'runs'))
max_checkpoint=$(printf "%s\n" "${checkpoints[@]}" | awk -F'checkpoint-' '/checkpoint-[0-9]+/ {print $2}' | sort -n | tail -n1)
max_checkpoint_path=$(printf "%s\n" "${checkpoints[@]}" | grep "checkpoint-$max_checkpoint" | head -n1)
# echo $max_checkpoint_path
datasets=("gsm8k" "math-500" "math")
# 模型参数
temp=0
max_len=2048
# 循环每一个模型检查点和数据集

# checkpoint=$max_checkpoint_path
checkpoint=${checkpoints[2]}
for dataset in "${datasets[@]}"
do
    checkpoint_name=$(basename "${checkpoint}")
    output_path=./preds_${dataset}/$base_model/${method}/${model}/${checkpoint_name}
    # output_path=test
    if [ -d $output_path ]; then
        echo "Directory $output_path exists"
        rm -rf $output_path

        # continue  # 跳过此次循环的剩余部分
    fi
    echo 'Output path: '$output_path
    echo 'Model to eval: '$checkpoint
    mkdir -p $output_path
    python generate.py --model ${checkpoint} --dataset ${dataset} --temperature ${temp} \
        --max_len ${max_len} --output_path ${output_path} --num_gpus 8 | tee  ${output_path}/score.txt
    done
