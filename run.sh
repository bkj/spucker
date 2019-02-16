#!/bin/bash

# run.sh

mkdir -p results

# --
# FB15k

python main.py \
    --dataset data/FB15k \
    --lr 0.003 \
    --lr-decay 0.99 \
    --ent-emb-dim 200 \
    --rel-emb-dim 200 \
    --sub-drop 0.2 \
    --hidden-drop1 0.2 \
    --hidden-drop2 0.3 \
    --label-smoothing 0.0 | tee results/FB15k.jl


# --
# WN18

python main.py \
    --dataset data/WN18 \
    --lr 0.005 \
    --lr-decay 0.995 \
    --ent-emb-dim 200 \
    --rel-emb-dim 30 \
    --sub-drop 0.2 \
    --hidden-drop1 0.1 \
    --hidden-drop2 0.2 \
    --label-smoothing 0.1 | tee results/WN18.jl


# --
# FB15k-237

python main.py \
    --dataset data/FB15k-237 \
    --lr 0.0005 \
    --lr-decay 1.0 \
    --ent-emb-dim 200 \
    --rel-emb-dim 200 \
    --sub-drop 0.3 \
    --hidden-drop1 0.4 \
    --hidden-drop2 0.5 \
    --label-smoothing 0.1 | tee results/FB15k-237.jl


# --
# WN18RR

python main.py \
    --dataset data/WN18RR \
    --lr 0.01 \
    --lr-decay 1.0 \
    --ent-emb-dim 200 \
    --rel-emb-dim 30 \
    --sub-drop 0.2 \
    --hidden-drop1 0.2 \
    --hidden-drop2 0.3 \
    --label-smoothing 0.1 | tee results/WN18RR.jl





