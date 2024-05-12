#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from fmnist_dataset import load_fashion_mnist
from model import CNN
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import random
from matplotlib import pyplot as plt
import time

labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def gettensor(x, y, device):
    return x.to(device), y.to(device)


def attack(classifier, X, y, eps):
    # iX = X.clone()
    loss_fn = nn.CrossEntropyLoss()
    for e in range(100):
        X.requires_grad = True
        y_pred = classifier(X)
        if torch.argmax(y_pred, dim=-1) == y:
            X.requires_grad = False
            break
        loss = loss_fn(y_pred, y)
        loss.backward()
        X.requires_grad = False
        X = X - eps * X.grad.data
        X = torch.clamp(X, 0, 255)
    return X


def test_attack(classifier, dataset, device):
    loss_fn = nn.CrossEntropyLoss()
    classifier.eval()
    Xs = []
    for x, y in tqdm(dataset):
        x, y = gettensor(x, y, device)
        logits = classifier(x)
        y_pred = torch.argmax(logits, dim=-1)
        if y_pred == y:
            Xs.append((x, y))
        if len(Xs) >= 1000:
            break
    AXs = []
    for X in tqdm(Xs):
        ny = (X[1]+1) % 10
        AX = attack(classifier, X[0], ny, 50)
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
        plt.savefig(f'../whitebox/new_model_{i}_{p.item()}{labels[p.item()]}->{(p.item()+1)%10}{labels[(p.item()+1)%10]}.jpg')
        # plt.imsave(f"../whitebox/{i}_{p}.jpg", u.reshape([28, 28]))
        # plt.imsave(f"../whitebox/{i}_{(p+1)%10}.jpg", v.reshape([28, 28]))
    return len(AXs) / len(Xs)  # 0.961
# old_model: 0.961
# new_model: 0.901

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--save_dir', type=str, default='../mymodel')
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--save_path', type=str, default='../mymodel/newmodel_89.38_.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()
    
    opt = parser.parse_args()
    classifier = CNN()
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
    _, _, test = load_fashion_mnist("../data", random=random)
    test_dataloader = DataLoader(test, batch_size=1)
    r = test_attack(classifier, test_dataloader, device)
    print(r)
