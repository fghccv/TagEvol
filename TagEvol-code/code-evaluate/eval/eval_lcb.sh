. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate lcb
cd LiveCodeBench
model=$1
checkpoint=$2
model_path=/home/zhoushiqi/workplace/icaed/model/
# model_path=/home/zhoushiqi/workplace/codectf/LLaMA-Factory/output
echo ${model_path}/${model}/${checkpoint}
python -m lcb_runner.runner.main \
    --model ${model}-${checkpoint} \
    --local_model_path ${model_path}/${model}/${checkpoint} \
    --scenario codegeneration \
    --evaluate \
    --num_process_evaluate 100 \
    --release_version release_v3 \
    --stop "###,<|EOT|>" && \

python -m lcb_runner.evaluation.compute_scores --eval_all_file output/${model}-${checkpoint}/Scenario.codegeneration_10_0.2_eval_all.json
echo "========================="
# python -m lcb_runner.runner.main \
#     --model magic-evol_0.1.3.aug.7.3.1-checkpoint-434 \
#     --local_model_path /home/zhoushiqi/workplace/codectf/model/deepseek-6.7b/magic-evol_0.1.3.aug.7.3.1/checkpoint-434\
#     --scenario codeexecution --cot_code_execution  \
#     --evaluate
