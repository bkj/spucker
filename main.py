#!/usr/bin/env python

"""
    spucker/main.py
"""

import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch

from spucker.helpers import load_dataset
from spucker.spucker import SpuckerModel, AdjList


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",         type=str,   default="data/FB15k-237")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=0.0005)
    parser.add_argument("--dr",              type=float, default=1.0)
    parser.add_argument("--edim",            type=int,   default=200)
    parser.add_argument("--rdim",            type=int,   default=200)
    parser.add_argument("--sub_drop",        type=float, default=0.3)
    parser.add_argument("--hidden_drop1",    type=float, default=0.4)
    parser.add_argument("--hidden_drop2",    type=float, default=0.5)
    # parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    parser.add_argument("--seed", type=int, default=123)
    
    return parser.parse_args()

# --
# Run

args = parse_args()
_ = np.random.seed(args.seed)
_ = torch.manual_seed(args.seed + 111)
_ = torch.cuda.manual_seed(args.seed + 222)

# --
# IO

print('loading %s' % args.dataset, file=sys.stderr)
train_data_idxs, valid_data_idxs, test_data_idxs, entity_lookup, relation_lookup =\
    load_dataset(args.dataset)

# Tensor adjacency list datastructures
train_adjlist = AdjList(train_data_idxs)
valid_adjlist = AdjList(valid_data_idxs)
test_adjlist  = AdjList(test_data_idxs)
all_adjlist   = AdjList(train_data_idxs + valid_data_idxs + test_data_idxs)

# --
# Define model

model = SpuckerModel(
    num_entities=len(entity_lookup),
    num_relations=len(relation_lookup),
    ent_emb_dim=args.edim,
    rel_emb_dim=args.rdim,
    sub_drop=args.sub_drop,
    hidden_drop1=args.hidden_drop1,
    hidden_drop2=args.hidden_drop2,
)
print(model, file=sys.stderr)
model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    
    # --
    # Train
    
    _ = model.train()
    
    idxs   = np.random.permutation(len(train_adjlist))
    chunks = np.array_split(idxs, idxs.shape[0] // args.batch_size)
    
    train_loss = []
    for chunk in tqdm(chunks):
        xb, yb = train_adjlist.get_batch_by_idx(chunk)
        x_s = torch.LongTensor(xb['s']).cuda()
        x_p = torch.LongTensor(xb['p']).cuda()
        y_i = torch.LongTensor(yb['i']).cuda()
        y_j = torch.LongTensor(yb['j']).cuda()
        
        opt.zero_grad()
        
        pred = model(x_s, x_p)
        loss = model.loss_fn(pred, y_i, y_j)
        
        loss.backward()
        opt.step()
        
        train_loss.append(loss.item())
    
    # --
    # Eval
    
    _ = model.eval()
    
    idxs   = np.arange(len(valid_adjlist))
    chunks = np.array_split(idxs, idxs.shape[0] // args.batch_size)
    
    all_ranks = []
    for chunk in chunks:
        xb, yb = valid_adjlist.get_batch_by_idx(chunk)
        x_s = torch.LongTensor(xb['s']).cuda()
        x_p = torch.LongTensor(xb['p']).cuda()
        y_i = torch.LongTensor(yb['i']).cuda()
        y_j = torch.LongTensor(yb['j']).cuda()
        
        pred = model(x_s, x_p)
        pred = torch.sigmoid(pred)
        target_pred = pred[(y_i, y_j)]
        
        # Zero all actual edges
        _, ayb = all_adjlist.get_batch_by_keys(zip(xb['s'], xb['p']))
        ay_i   = torch.LongTensor(ayb['i']).cuda()
        ay_j   = torch.LongTensor(ayb['j']).cuda()
        pred[(ay_i, ay_j)] = 0
        
        ranks = (target_pred.view(-1, 1) < pred[y_i]).sum(dim=-1)
        all_ranks.append(ranks.cpu().numpy())
    
    all_ranks = np.hstack(all_ranks)
    
    print(json.dumps({
        "epoch"      : int(epoch),
        "train_loss" : float(np.mean(train_loss)),
        "mrr"        : float(np.mean(1 / (1 + all_ranks))),
        "h_at_10"    : float(np.mean(all_ranks < 10)),
        "h_at_03"    : float(np.mean(all_ranks < 3)),
        "h_at_01"    : float(np.mean(all_ranks < 1)),
    }))
    sys.stdout.flush()

