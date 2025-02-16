#!/bin/bash

. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 定义模型检查点和数据集数组
model=$1
method=$2
model_path=../${method}/models/llama3/$1
checkpoints=($(ls -d ${model_path}/*/ | grep -v 'runs'))
datasets=("gsm8k")
# 模型参数
temp=0
max_len=2048

# 循环每一个模型检查点和数据集
for (( i=2; i>=1; i-- ))
do
    checkpoint=${checkpoints[i]}
    for dataset in "${datasets[@]}"
    do
        checkpoint_name=$(basename "${checkpoint}")
        output_path=./preds_${dataset}/llama3/${method}/${model}/${checkpoint_name}
        # output_path=test
        if [ -d $output_path ]; then
            echo "Directory $output_path exists"
            # rm -rf $output_path
            # rm -rf $output_path.jsonl
            # rm -rf ${output_path}_eval_results.json
            # rm -rf ${output_path}_pass@k.txt
            # continue  # 跳过此次循环的剩余部分
        fi
        echo 'Output path: '$output_path
        echo 'Model to eval: '$checkpoint
        mkdir -p $output_path
        python generate.py --model ${checkpoint} --dataset ${dataset} --temperature ${temp} \
            --max_len ${max_len} --output_path ${output_path} --num_gpus 8 | tee  ${output_path}/score.txt
        done
done