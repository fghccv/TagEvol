#!/bin/bash





echo instruct generation


data_name=all_math_15k
tag_name=all_math_15k
num_pool=21
for round in 1
do
num_tag=$((2*round-1))
target_data_name=tag_evol_tag${num_tag}_pool_${num_pool}_20mul3up_alltags_wyxprompt_70btag_7bins
# target_data_name=test
echo $target_data_name
mkdir -p ./datas/${target_data_name}
python3 tag_evol_multitag_20mul3up_all_tags.py \
    --model_name_or_path /data/scir/yixuanwang/models/qwen2.5-7b-ins \
    --source_file ./datas/${data_name}/${data_name}.json \
    --target_file ./datas/${target_data_name}/${target_data_name}.json \
    --temperature 0.7 \
    --max_tokens 2048 \
    --tp 4 \
    --tag_file ./datas/${tag_name}/${tag_name}.json \
    --num_tag ${num_tag} \
    --num_pool ${num_pool}


python3 gen_response.py \
    --model_name_or_path /data/scir/yixuanwang/models/qwen2.5-72b-ins  \
    --inst_file ./datas/${target_data_name}/${target_data_name}.json

done

# bash scripts/temp.sh
