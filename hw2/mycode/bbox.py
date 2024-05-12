#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from fmnist_dataset import load_fashion_mnist
from model import CNN, PCNN
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import random
from matplotlib import pyplot as plt
import time

import pickle



labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def gettensor(x, y, device):
    return x.to(device), y.to(device)


def attack(classifier, X, y, eps):
    loss_fn = nn.CrossEntropyLoss()
    y_p = classifier(X)
    last_loss = loss_fn(y_p, y)
    for e in range(100):
        noise = torch.randn_like(X) * eps
        X_ = X + noise
        y_ = classifier(X_)
        new_loss = loss_fn(y_, y)
        if new_loss < last_loss:
            X = X_
            last_loss = new_loss
        if torch.argmax(y_, dim=-1) == y:
            return X_
    return X


def test_attack(classifier, dataset, device):
    loss_fn = nn.CrossEntropyLoss()
    classifier.eval()
    Xs = []
    for i in tqdm(range(1000)):
        Xs.append(gettensor(torch.from_numpy(dataset[0][i]), torch.from_numpy(dataset[1][i]).reshape(-1), device))
    AXs = []
    for X in tqdm(Xs):
        ny = (X[1]+1) % 10
        AX = attack(classifier, X[0], ny, 8)
        logits = classifier(AX)
        y_p = torch.argmax(logits, dim=-1)
        if y_p == ny:
            AXs.append((X[0], AX, X[1]))
    print(len(AXs))
    for i in tqdm(range(min(len(AXs), 10))):
        u, v, p = AXs[i]
        plt.subplot(1, 2, 1)
        plt.imshow(u.cpu().numpy().reshape([28, 28]))
        plt.subplot(1, 2, 2)
        plt.imshow(v.cpu().numpy().reshape([28, 28]))
        plt.tight_layout()
        plt.savefig(f'../blackbox/{i}_{p.item()}{labels[p.item()]}->{(p.item()+1)%10}{labels[(p.item()+1)%10]}.jpg')
    return len(AXs) / len(Xs)  # 0.548


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--save_path', type=str, default='../model/cnn.ckpt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()
    
    opt = parser.parse_args()
    classifier = PCNN()
    if int(opt.gpu) < 0:
        device = torch.device('cpu')
        torch.manual_seed(opt.rand_seed)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda")
        torch.manual_seed(opt.rand_seed)
        torch.cuda.manual_seed(opt.rand_seed)
        classifier.to('cuda')
    random.seed(opt.rand_seed)
    classifier.load_state_dict(torch.load(opt.save_path))
    # _, _, test = load_fashion_mnist("../data", random=random)
    # test_dataloader = DataLoader(test, batch_size=1)
    with open('../attack_data/correct_1k.pkl', 'rb') as f:
        data = pickle.load(f)
    r = test_attack(classifier, data, device)
    print(r)
