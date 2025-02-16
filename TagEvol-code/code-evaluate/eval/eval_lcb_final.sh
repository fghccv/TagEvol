. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate lcb
cd LiveCodeBench
model=$1
model_path=/home/zhoushiqi/workplace/model/
# model_path=/home/zhoushiqi/workplace/codectf/LLaMA-Factory/output
echo ${model_path}/${model}
python -m lcb_runner.runner.main \
    --model ${model} \
    --local_model_path ${model_path}/${model} \
    --scenario codegeneration \
    --evaluate \
    --num_process_evaluate 100 \
    --release_version release_v3 \
    --stop "###,<|EOT|>" && \

python -m lcb_runner.evaluation.compute_scores --eval_all_file output/${model}/Scenario.codegeneration_10_0.2_eval_all.json --start_date 2023-05-01 --end_date 2024-09-01
echo "========================="

