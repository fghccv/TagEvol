#!/bin/bash

. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate lcb
export VLLM_WORKER_MULTIPROC_METHOD=spawn


models=(
    # "magic-evol_0.2_base_ori2" 
    # "magic-evol_0.1.3.aug.7.3.1" 
    # "magic-evol_0.1.3.aug.6.3.1_config2"
    # "magic_v0.2_base_7.3.1"
    # "magic_v0.2_baseori_7.3.1"
    # "magicv0.1.3.aug.8.0.1"
    # "magic_v0.2_base_7.5.1"
    # "magic-base-determine"
    # "magic_v0.2_aug.7.3.1epoch"
    # "magic_baseline"
    # "magicv0.1.3.aug.8.0.1epoch"
    # "dpop_7.3.1epoch_merge"
    # "dpo_7.3.1.1epoch"
    # "magiccoder-evol_0.1_ori"
    # "magiccoder-evol_0.1_ds"
    # "magic-0.1.1_aug.7.3.1epoch"
    # "magic7.3.1epoch+7.3aug-mask"
    # "magic0.1.5-7.3.aug_mask_continue_7.3.1epoch"
    # "magic0.1.5-7.3.aug_mask_continue_7.3.1epoch_2e-5"
    # "magic7.3.1+7.3.aug_dedup-mask_5e-5"
    # "magic7.3.1+7.3.aug_dedup-mask"
    # "magic0.1.5-7.3.aug_mask_continue_ori_5e-5"
    # "magic0.1.5-7.3.aug_mask.1epoch_continue_7.3.1epoch_m1_2"
    # "magic0.1.5-7.3.aug_mask.1epoch_continue_ori_m1_2"
    # "magic_baseline"
    # "magic0.1.5-7.3.aug_mask.2epoch_continue_7.3.1epoch_m1_2"
    # "magic_7.3_all_mask_continue_ori"
    # "magic_7.3_all_ori_mask.1epoch_continue_7.3.1epoch"
    # "magic_7.3_all_ori_mask.2epoch_continue_7.3.1epoch"
    # "magic_7.3.1epoch_10.26"
    # "qw-magic_condition_k_111183_sample_1_temp_0-response"
    # "qw-magic_cqft_k_110k_sample_1-response"
    # "qw-magic_cqft_k_110k_sample_1-response2"
    # "qw-magic_ori-response"
    # "qw-magic_ds-response"
    # "qw-magic_condition_aug_270k-response"
    # "ds_inst_condition_k_111183_sample_1_temp_0_filter_3gram-response_lr_5e-5"
    # "ds_icondition_k_1m_sample_1_temp_1+ori-response"
    # "ds_icondition_k_1m_sample_1_temp_1_add_ori-response_qwen32b"
    # "ds_icondition_k_111k_sample_1_temp_1-response_qwen32b_temp0"
    # "ds-magic-instruct-111k-temp1-response_qwen32b_temp0"
    # "ds_condition_cross_interpolation_k1_n2_110k_temp_1-response_qwen7b_part0_1_temp0.2_add_ori"
    # "ds_qft_temp_1-response_qwen7b_part0_1_temp0.2_add_ori"
    # "ds_condition_embed-ds-6.7b-base-hidden-ck1086_interpolation_k1_n2_110k_temp_1-response_qwen7b_temp0.2_add_ori"
    # "qft_roud_1_part0_1_temp0.2_all"
    # "ds_condition_embed-ds-6.7b-base-hidden-ck1086_interpolation_k1_n2_110k_temp_1-response_qwen7b_temp0.2_add_ori"
    "qft_roud_0_filter_111k_response_32b"
)



# 循环每一个模型检查点和数据集
for model in "${models[@]}"; do
    model_path=/home/zhoushiqi/workplace/icaed/model/$model
    checkpoints=($(ls -d ${model_path}/*/ | grep -v 'runs'))
    for (( i=0; i<=${#checkpoints[@]}-1; i++))
    do
    checkpoint=${checkpoints[i]}
        checkpoint_name=$(basename "${checkpoint}")
        if [ -d "LiveCodeBench/output/${model}-${checkpoint_name}" ]; then
            echo "Directory $checkpoint exists, skipping..."
            # continue
            # rm -rf LiveCodeBench/output/${model}-${checkpoint_name}
        fi
        echo $model, $checkpoint_name
        bash /home/zhoushiqi/zsq_lib/CTFCoder/eval/eval_lcb.sh $model $checkpoint_name

    done
done