#!/bin/bash
data_names=("code_alpaca_20k-evol-roud01" "code_alpaca_20k-evol-roud012")
for data_name in ${data_names[@]}; do
bash train_func.sh $data_name
done
