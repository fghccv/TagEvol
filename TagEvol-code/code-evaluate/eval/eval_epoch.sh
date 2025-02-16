#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 定义模型检查点和数据集数组
model=$1
method=$2
base_model=$3
model_path=../../$method/models/$base_model/$1
checkpoints=($(ls -d ${model_path}/*/ | grep -v 'runs'))
datasets=("humaneval" "mbpp")

# 模型参数
temp=0
max_len=2048
pred_num=1
num_seqs_per_iter=1
export MBPP_OVERRIDE_PATH="../data/MbppPlus.jsonl.gz"
export HUMANEVAL_OVERRIDE_PATH="../data/HumanEvalPlus.jsonl.gz"
# 循环每一个模型检查点和数据集
for (( i=0; i<=2; i++))
do
    checkpoint=${checkpoints[i]}
    for dataset in "${datasets[@]}"
    do
        checkpoint_name=$(basename "${checkpoint}")
        output_path=../preds_${dataset}/$base_model/$method/${model}/${checkpoint_name}/T${temp}_N${pred_num}
        if [ -d $output_path ]; then
            echo "Directory $output_path exists"
            rm -rf $output_path
            rm -rf $output_path.jsonl
            rm -rf ${output_path}_eval_results.json
            rm -rf ${output_path}_pass@k.txt
            # continue  # 跳过此次循环的剩余部分
        fi
        mkdir -p ${output_path}
        echo 'Output path: '$output_path
        echo 'Model to eval: '$checkpoint
        python generate.py --model ${checkpoint} --dataset ${dataset} --temperature ${temp} \
            --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path} --num_gpus 8
        python process.py --path ${output_path} --out_path ${output_path}.jsonl --dataset ${dataset}
        python evalute.py --path ${output_path}.jsonl --dataset ${dataset} --N 1 | tee ${output_path}_pass@k.txt
    done
done