#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataloader import random_split_dataloader, unlabeled_dataloader
from copy import deepcopy
from itertools import repeat, cycle


import models
from utils import progress_bar

#cartoon 2344
#photo 1670
#art 2048

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="dgresnet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=0.2, type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--targetid', type=int)
parser.add_argument('--labelid', type=int)
args = parser.parse_args()

domains =['photo', 'art_painting', 'cartoon', 'sketch']


use_cuda = torch.cuda.is_available()

best_acc = 0.  # best test accuracy
best_val_acc = 0.
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

source_unlabel_domain = deepcopy(domains)
target_domain = [source_unlabel_domain.pop(args.targetid)]
source_domain = [source_unlabel_domain.pop(args.labelid)]

print ('source_domain', source_domain)
print ('target_domain', target_domain)

trainloader, valloader, testloader = random_split_dataloader(
        data_root='/home/wrq/data/data/PACS/kfold/', source_domain=source_domain, target_domain=target_domain,
        batch_size=args.batch_size, get_domain_label=False, num_workers=4)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)


# def mixup_data(x, y, y_domain, alpha=1.0, use_cuda=True):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#
#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)
#
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     y_domain_a, y_domain_b = y_domain, y_domain[index]
#     return mixed_x, y_a, y_b, y_domain_a, y_domain_b, lam
#
#
# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.
    reg_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = map(Variable, (inputs, targets))
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


def val(epoch):
    global best_val_acc
    net.eval()
    test_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (inputs, targets) in enumerate(valloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        progress_bar(batch_idx, len(valloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_val_acc:
        checkpoint(acc, epoch)
    if acc >= best_val_acc:
        best_val_acc = acc
        print ('temporal best: {:.4f}'.format(best_val_acc))
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch == 30:
        lr /= 10
    if epoch == 50:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#if not os.path.exists(logname):
#    with open(logname, 'w') as logfile:
#        logwriter = csv.writer(logfile, delimiter=',')
#        #logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
#        #                    'test loss', 'test acc'])
#        logwriter.writerow(target_domain)


with open(logname, 'a') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    #logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
    #                    'test loss', 'test acc'])
    logwriter.writerow(target_domain)

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    #val_loss, val_acc = val(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    #if val_acc == best_val_acc:
    #    with open(logname, 'a') as logfile:
    #        logwriter = csv.writer(logfile, delimiter=',')
    #        logwriter.writerow([epoch, val_acc, test_acc])


    #with open(logname, 'a') as logfile:
    #    logwriter = csv.writer(logfile, delimiter=',')
    #    #logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
    #    #                    test_acc])
    #    logwriter.writerow([epoch, val_acc, reg_loss, train_acc, test_loss,
    #                        test_acc])
