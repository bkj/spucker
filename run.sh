#!/bin/bash

# run.sh

mkdir -p results

python main.py --dataset data/FB15k-237 | tee results/.jl
# {"epoch": 499, "train_loss": 0.0005535096377437062, "mrr": 0.36017325820962404, "h_at_10": 0.5436270316509837, "h_at_03": 0.3925292272597662, "h_at_01": 0.2696036498431708}

python main.py \
    --dataset data/FB15k \
    --lr 0.003 | tee results/FB15k.jl

python main.py \
    --dataset data/WN18 \
    --dr 30 \
    --lr 0.005 | tee results/WN18.jl

python main.py \
    --dataset data/WN18RR \
    --dr 30 \
    --lr 0.01 | tee results/WN18RR.jl