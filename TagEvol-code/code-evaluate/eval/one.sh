#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate codellama
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 定义模型检查点和数据集数组
model=$1
# model_path=/var/s3fs-hgd/xzluo/models/ctf/$1
model_path=/home/zhoushiqi/workplace/codectf/model/deepseek-6.7b/$1
datasets=("humaneval" "mbpp")

# 模型参数
temp=0
max_len=2048
pred_num=1
num_seqs_per_iter=1
export MBPP_OVERRIDE_PATH="/home/zhoushiqi/zsq_lib/evalplus/evalplus/MbppPlus.jsonl.gz"
# 循环每一个模型检查点和数据集
for dataset in "${datasets[@]}"
do
    output_path=/home/zhoushiqi/zsq_lib/CTFCoder/preds_${dataset}/${model}-final/T${temp}_N${pred_num}
    mkdir -p ${output_path}
    echo 'Output path: '$output_path
    echo 'Model to eval: '$model_path
    python generate.py --model ${model_path} --dataset ${dataset} --temperature ${temp} \
        --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path} --num_gpus 8
    python process.py --path ${output_path} --out_path ${output_path}.jsonl --dataset ${dataset}
    python evalute.py --path ${output_path}.jsonl --dataset ${dataset} --N 1 | tee ${output_path}_pass@k.txt
done