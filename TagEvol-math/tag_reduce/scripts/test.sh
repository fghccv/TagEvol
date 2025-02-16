#!/bin/bash



num_pool=21
python merge_datas.py \
 --data1_path ./datas/tag_evol_tag1_pool_${num_pool}_20mul3up_alltags/tag_evol_tag1_pool_${num_pool}_20mul3up_alltags.json \
 --data2_path ./datas/tag_evol_tag3_pool_${num_pool}_20mul3up_alltags/tag_evol_tag3_pool_${num_pool}_20mul3up_alltags.json \
 --target_path ./datas/final_datas/tag_evol_tag13_pool_${num_pool}_20mul3up_alltags.json


python merge_datas.py \
 --data1_path ./datas/final_datas/tag_evol_tag13_pool_${num_pool}_20mul3up_alltags.json \
 --data2_path ./datas/tag_evol_tag5_pool_${num_pool}_20mul3up_alltags/tag_evol_tag5_pool_${num_pool}_20mul3up_alltags.json \
 --target_path ./datas/final_datas/tag_evol_tag135_pool_${num_pool}_20mul3up_alltags.json

 