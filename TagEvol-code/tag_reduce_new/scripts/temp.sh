#!/bin/bash

cp ./datas/tag_evol_tag3_7b_yxprompt/tag_evol_tag3_7b_yxprompt.json ./datas/final_datas/tag_evol_tag3_7b_yxprompt.json

python3 merge_datas.py  \
 --data1_path ./datas/tag_evol_tag3_7b_yxprompt/tag_evol_tag3_7b_yxprompt.json \
 --data2_path ./datas/tag_evol_tag5_7b_yxprompt/tag_evol_tag5_7b_yxprompt.json \
 --target_path ./datas/final_datas/tag_evol_tag35_7b_yxprompt.json

python3 merge_datas.py  \
 --data1_path ./datas/final_datas/tag_evol_tag35_7b_yxprompt.json \
 --data2_path ./datas/tag_evol_tag7_7b_yxprompt/tag_evol_tag7_7b_yxprompt.json \
 --target_path ./datas/final_datas/tag_evol_tag357_7b_yxprompt.json

data_names=("tag_evol_tag357_7b_yxprompt" "tag_evol_tag35_7b_yxprompt" "tag_evol_tag3_7b_yxprompt")

for name in ${data_names[@]}
do
bash ./scripts/train.sh $name
done