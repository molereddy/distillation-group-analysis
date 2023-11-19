#!/bin/bash

export TZ=America/New_York

model=$1
gpu_index=$2

run_experiment() {
    local model=$1
    local dataset=$2
    local method=$3
    local lr=$4
    local wd=$5
    local gpu=$6

    echo $(date)
    echo "$dataset $method $model: with LR=$lr and WD=$wd"
    python3 run_exp.py --method "$method" -d "$dataset" --model_type "$model" \
    --lr "$lr" --weight_decay "$wd" \
    --device $gpu --batch_size 64 --hyperparam
}

run_experiment "$model" CUB ERM 1e-3 1e-4 "$gpu_index"
run_experiment "$model" CUB ERM 1e-4 1e-1 "$gpu_index"
run_experiment "$model" CUB ERM 1e-5 1 "$gpu_index"

echo $(date)
