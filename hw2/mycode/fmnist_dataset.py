#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from mnist import MNIST

class FashionMNISTDataset(Dataset):
    
    def __init__(self, data):
        
        x, y = data
        assert len(x) == len(y)
        self.x = x
        self.y = y
        
    def __len__(self):
        
        return len(self.x)
    
    def __getitem__(self, idx):
        
        return torch.tensor(self.x[idx]).float(), \
                torch.tensor(self.y[idx])
    
def load_fashion_mnist(fmnist_dir, n_dev=10000, random=None):
    
    fmnist = MNIST(fmnist_dir, return_type="lists")
    train = fmnist.load_training()
    test = fmnist.load_testing()
    
    assert n_dev >= 0 and n_dev <= len(train[0]), \
            "Invalid dev size %d, should be within 0 to %d" \
            % (n_dev, len(train[0]))
    if random is None:
        import random
    idx = random.sample(range(len(train[0])), len(train[0]))
    dev = [], []
    for i in idx[:n_dev]:
        dev[1].append(train[1][i])
        dev[0].append(train[0][i])
    _train = [], []
    for i in idx[n_dev:]:
        _train[1].append(train[1][i])
        _train[0].append(train[0][i])
    return FashionMNISTDataset(_train), FashionMNISTDataset(dev), FashionMNISTDataset(test)
