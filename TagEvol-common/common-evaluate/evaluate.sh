#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 定义模型检查点和数据集数组
# model=$1
# method=$2
# model_path=../${method}/models/llama3/$1
# checkpoints=($(ls -d ${model_path}/*/ | grep -v 'runs'))
model=$1
base_model=$2
method=$3
model_path=../$method/models/$base_model/$1
checkpoints=($(ls -d ${model_path}/*/ | grep -v 'runs'))
# checkpoints=(/home/share/models/$model)

datasets=("ifeval")
# 模型参数
temp=0
max_len=1024
export NLTK_DATA="/home/sqshou/zsq_lib/nltk_data"
# 循环每一个模型检查点和数据集
for (( i=1; i<=1; i++ ))
do
    checkpoint=${checkpoints[i]}
    for dataset in "${datasets[@]}"
    do
        checkpoint_name=$(basename "${checkpoint}")
        output_path=./preds_${dataset}/${method}/${model}/${checkpoint_name}
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
        python3 generate.py --model ${checkpoint} --dataset ${dataset} --temperature ${temp} \
            --max_len ${max_len} --output_path ${output_path} --num_gpus 8 

        python3 -m instruction_following_eval.evaluation_main \
            --input_data ./datas/${dataset}.jsonl \
            --input_response_data ${output_path}/results.jsonl \
            --output_dir ${output_path} | tee  ${output_path}/score.txt

        done
done