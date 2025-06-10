#!/bin/bash




echo instruct generation


num_pool=21
data_name=databricks-dolly-15k
tag_name=databricks-dolly-15k
model_path=""
for round in 2
do
num_tag=$((2*round-1))
target_data_name=tag_evol_tag${num_tag}
echo $target_data_name
mkdir -p ./datas/${target_data_name}
python3 tag_evol_api.py \
    --model_name_or_path $model_path \
    --source_file ./datas/${data_name}/${data_name}.json \
    --target_file ./datas/${target_data_name}/${target_data_name}.json \
    --temperature 0.7 \
    --max_tokens 2048 \
    --tp 2 \
    --tag_file ./datas/${tag_name}/${tag_name}.json \
    --num_tag ${num_tag} \
    --num_pool ${num_pool}

python3 gen_response.py \
    --model_name_or_path $model_path  \
    --inst_file ./datas/${target_data_name}/${target_data_name}.json

done
