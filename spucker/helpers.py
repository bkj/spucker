#!/usr/bin/env python

"""
    spucker/helpers.py
"""

import os

def load_fold(data_dir, data_type="train", reverse=False):
    with open("%s.txt" % os.path.join(data_dir, data_type), "r") as f:
        data = f.read().strip().split("\n")
        data = [i.split() for i in data]
        if reverse:
            data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
    
    return data

def load_dataset(data_dir):
    train_data = load_fold(data_dir, "train", reverse=True)
    valid_data = load_fold(data_dir, "valid", reverse=True)
    test_data  = load_fold(data_dir, "test", reverse=True)
    
    data = train_data + valid_data + test_data
    
    sub, pred, obj = list(zip(*data))
    
    entities  = sorted(list(set.union(set(sub), set(obj))))
    relations = sorted(list(set(pred)))
    
    entity_lookup   = dict(zip(entities, range(len(entities))))
    relation_lookup = dict(zip(relations, range(len(relations))))
    
    train_data_idxs = [(
        entity_lookup[x[0]], relation_lookup[x[1]], entity_lookup[x[2]],
    ) for x in train_data]
    
    valid_data_idxs = [(
        entity_lookup[x[0]], relation_lookup[x[1]], entity_lookup[x[2]],
    ) for x in valid_data]
    
    test_data_idxs = [(
        entity_lookup[x[0]], relation_lookup[x[1]], entity_lookup[x[2]],
    ) for x in test_data]
    
    return train_data_idxs, valid_data_idxs, test_data_idxs, entity_lookup, relation_lookup