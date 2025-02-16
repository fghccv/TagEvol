

# num_pool=21
# cp ./datas/tag_evol_tag1_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b/tag_evol_tag1_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b.json ./datas/final_datas/tag_evol_tag1_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b.json
# python merge_datas.py \
#  --data1_path ./datas/tag_evol_tag1_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b/tag_evol_tag1_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b.json \
#  --data2_path ./datas/tag_evol_tag3_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b/tag_evol_tag3_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b.json \
#  --target_path ./datas/final_datas/tag_evol_tag13_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b.json
# python merge_datas.py \
#  --data1_path ./datas/final_datas/tag_evol_tag13_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b.json \
#  --data2_path ./datas/tag_evol_tag5_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b/tag_evol_tag5_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b.json \
#  --target_path ./datas/final_datas/tag_evol_tag135_pool_${num_pool}_20mul3up_alltags_wyxprompt2_70b.json

# # cp ./datas/tag_evol_tag3_pool_${num_pool}_20mul3up_alltags/tag_evol_tag3_pool_${num_pool}_20mul3up_alltags.json ./datas/final_datas/tag_evol_tag3_pool_${num_pool}_20mul3up_alltags.json
# data_names=("tag_evol_tag135_pool_21_20mul3up_alltags_wyxprompt2_70b")
# for data_name in ${data_names[@]};
# do
# bash ./scripts/train.sh ${data_name}
# done

data_name=all_math_15k_res72b
# python3 gen_response.py \
#     --model_name_or_path /data/scir/yixuanwang/models/qwen2.5-72b-ins  \
#     --inst_file ./datas/final_datas/${data_name}.json

bash ./scripts/train.sh ${data_name}
