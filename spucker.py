#!/usr/bin/env python

"""
    spucker.py
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

# --

class AdjList:
    def __init__(self, data_idxs):
        
        slice_dict = defaultdict(list)
        for s, p, o in data_idxs:
            slice_dict[(s, p)].append(o)
        
        self._slice_dict = dict(slice_dict)
        self._keys = list(self._slice_dict.keys())
    
    def __len__(self):
        return len(self._slice_dict)
    
    def get_batch_by_idx(self, idxs):
        keys = [self._keys[idx] for idx in idxs]
        return self.get_batch_by_keys(keys)
    
    def get_batch_by_keys(self, keys):
        xb = {"s" : [], "p" : []}
        yb = {"i" : [], "j" : []}
        
        s, p, o = [], [], []
        for offset, k in enumerate(keys):
            
            xb['s'].append(k[0])
            xb['p'].append(k[1])
            
            for oo in self._slice_dict[k]:
                yb['i'].append(offset)
                yb['j'].append(oo)
        
        return xb, yb



def sparse_bce_with_logits(x, i, j):
    # !! Add support for label smoothing
    t1 = x.clamp(min=0).mean()
    t2 = - x[(i, j)].sum() / x.numel()
    t3 = torch.log(1 + torch.exp(-torch.abs(x))).mean()
    
    return t1 + t2 + t3


class SpuckerModel(nn.Module):
    def __init__(self, num_entities, num_relations, ent_emb_dim, rel_emb_dim,
        sub_drop, hidden_drop1, hidden_drop2):
        super().__init__()
        
        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim
        
        self.sub_emb = torch.nn.Embedding(num_entities, ent_emb_dim, padding_idx=0)
        self.rel_emb = torch.nn.Embedding(num_relations, rel_emb_dim, padding_idx=0)
        
        torch.nn.init.xavier_normal_(self.sub_emb.weight.data)
        torch.nn.init.xavier_normal_(self.rel_emb.weight.data)
        
        self.W = torch.nn.Parameter(
            torch.tensor(
                np.random.uniform(-1, 1, (rel_emb_dim, ent_emb_dim, ent_emb_dim)),
                dtype=torch.float,
                requires_grad=True
            )
        )
        
        self.sub_drop      = torch.nn.Dropout(sub_drop)
        self.hidden_drop1  = torch.nn.Dropout(hidden_drop1)
        self.hidden_drop2  = torch.nn.Dropout(hidden_drop2)
        
        self.bn_sub     = torch.nn.BatchNorm1d(ent_emb_dim)
        self.bn_hidden  = torch.nn.BatchNorm1d(ent_emb_dim)
        
        self.loss_fn = sparse_bce_with_logits
    
    def forward(self, s, r):
        # !! Where should the batchnorms be?
        
        r = self.rel_emb(r)
        
        x = r @ self.W.view(self.rel_emb_dim, -1) # 'ij,jkb->ikb'
        x = x.view(-1, self.ent_emb_dim, self.ent_emb_dim)
        x = self.hidden_drop1(x)
        
        s = self.sub_emb(s)
        s = self.sub_drop(self.bn_sub(s))
        
        x = torch.bmm(s.unsqueeze(-2), x).squeeze()
        x = self.hidden_drop2(self.bn_hidden(x))
        
        x = x @ self.sub_emb.weight.transpose(1, 0) # Like linear layer, but tied to self.E
        return x

