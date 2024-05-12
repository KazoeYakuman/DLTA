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


def evaluate(classifier, dataset, device):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    for x, y in dataset:
        
        with torch.no_grad():
            x, y = gettensor(x, y, device)
            logits = classifier(x)
            res = torch.argmax(logits, dim=1) == y
            testcorrect += torch.sum(res)
            testnum += len(y)
    
    acc = float(testcorrect) * 100.0 / testnum
    return acc


def trainEpochs(classifier, optimizer, loss_fn, epochs, training_set, dev_set, test_set,
                print_each, save_dir, device, AXs):

    for ep in tqdm(range(1, epochs + 1)):
        
        classifier.train()
        print_loss_total = 0
        
        print ('Ep %d' % ep)
        
        for i, (x, y) in enumerate(training_set):
            
            optimizer.zero_grad()
            x, y = gettensor(x, y, device)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            
            # print_loss_total += loss.item()
            # if (i + 1) % print_each == 0: 
            #     print_loss_avg = print_loss_total / print_each
            #     print_loss_total = 0
            #     print('    %.4f' % print_loss_avg)

        for X in AXs:
            optimizer.zero_grad()
            x, y = gettensor(X[0], X[1], device)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
                
        acc = evaluate(classifier, dev_set, device)
        print ('  dev acc = %.2f%%' % acc)
        acc = evaluate(classifier, test_set, device)
        print ('  test acc = %.2f%%' % acc)
        torch.save(classifier.state_dict(),
                   os.path.join(save_dir, 'ep_' + str(ep) + '_devacc_' + str(acc) + '_.pt'))


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
        AXs.append((AX, X[1]))
    print(len(AXs))
    return AXs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--save_dir', type=str, default='../mymodel')
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--save_path', type=str, default='../mymodel/mymodel_91.05_.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--log_per_step', type=int, default=100)
    args = parser.parse_args()
    
    opt = parser.parse_args()
    classifier = CNN()
    new_model = CNN()
    if int(opt.gpu) < 0:
        device = torch.device('cpu')
        torch.manual_seed(opt.rand_seed)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda")
        torch.manual_seed(opt.rand_seed)
        torch.cuda.manual_seed(opt.rand_seed)
        classifier.to('cuda')
        new_model.to('cuda')
    random.seed(opt.rand_seed)
    classifier.load_state_dict(torch.load(opt.save_path))
    train, dev, test = load_fashion_mnist("../data", random=random)
    train_dataloader = DataLoader(train, batch_size=1)
    dev_dataloader = DataLoader(dev, batch_size=opt.eval_batch_size)
    test_dataloader = DataLoader(test, batch_size=opt.eval_batch_size)
    AXs = test_attack(classifier, train_dataloader, device)

    optimizer = optim.Adam(new_model.parameters(), lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainEpochs(new_model, optimizer, criterion, opt.num_epochs,
                train_dataloader, dev_dataloader, test_dataloader,
                opt.log_per_step, opt.save_dir, device, AXs)
