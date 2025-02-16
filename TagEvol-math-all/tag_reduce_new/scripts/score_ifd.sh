#!/bin/bash
#SBATCH -J greedy                             # 作业名为 test
#SBATCH -o /home/zhoushiqi/workplace/graduate_project/script/infer_score/log/greedy.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -e /home/zhoushiqi/workplace/graduate_project/script/infer_score/log/greedy.err
#SBATCH -p gpu4                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 10:00:00                            # 任务运行的最长时间为 1 小时                          
#SBATCH --gres=gpu:8
#SBATCH --mem 500GB
#SBATCH -c 128
# conda
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr
export TOKENIZERS_PARALLELISM=false

model_name=Meta-Llama-3-8B_ms
model=/home/zhoushiqi/workplace/model/${model_name}
data_name=gsm8k_7k_res72b
output_dir=./score_datas/${model_name}/${data_name}
data_path=./datas/gsm8k_7k/${data_name}.json
# rm -rf ${output_dir}
mkdir -p ${output_dir}


index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  gpu=$((i))
  output_path=$output_dir/$gpu.json
  echo 'Running process #' ${i} 
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python ../score_ifd.py --model ${model} \
      --output_path ${output_path} \
      --gpu ${gpu} \
      --data_path ${data_path} \
  ) &
  if (($index % $gpu_num == 0)); then wait; fi;
done

echo merge

python ./score_datas/merge_result.py --dir_name ${output_dir}



