#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

from itertools import cycle

from parser import parser
from dataloader import random_split_dataloader, unlabeled_dataloader
from utils import progress_bar, mixup_data, print2, set_randseed
import csv
import os

import numpy as np
import torch
import models
from trainer import SupervisedTrainer, SemiSupervisedTrainer

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from copy import deepcopy


def supervisedtrainer():
    st = SupervisedTrainer(model=net, optimizer=optimizer,
                                trainloader=labeledloader, valloader=valloader, testloader=testloader,
                                cuda=use_cuda, epoch=args.epoch, lr=args.lr, n_classes=n_classes,
                                ckptfilename=checkpoint_filename, logname=logname)
    return st


def semisupervisedtrainer():
    sst = SemiSupervisedTrainer(model=net, optimizer=optimizer,
                                labeledloader=labeledloader, unlabeledloader=unlabeledloader, valloader=valloader, testloader=testloader,
                                cuda=use_cuda, epoch=args.epoch, lr=args.lr, n_classes=n_classes,
                                ckptfilename=checkpoint_filename, logname=logname)
    return sst


# 原代码移动到parser.py中
args = parser()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
domains = ['photo', 'art_painting', 'cartoon', 'sketch']
n_classes = 7
use_cuda = torch.cuda.is_available()

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.name is None:
    args.name = domains[args.targetid]

if args.seed != 0:
    set_randseed(args.seed)

assert args.labeledid is not None
assert args.targetid is not None
unlabeled_domain = deepcopy(domains)
labeled_domain = []
for id in args.labeledid:
    labeled_domain.append(domains[id])
    unlabeled_domain.remove(domains[id])
unlabeled_domain.remove(domains[args.targetid])
target_domain = [domains[args.targetid]]
print('Labeled:', labeled_domain, 'Unlabeled:', unlabeled_domain, 'Target:', target_domain)

labeledloader, valloader, testloader = random_split_dataloader(
        data_root='/home/wrq/data/data/PACS/kfold/', source_domain=labeled_domain, target_domain=target_domain,
        batch_size=args.batch_size, num_workers=4, get_domain_label=True)

if len(unlabeled_domain) > 0:
    unlabeledloader = unlabeled_dataloader(
        data_root='/home/wrq/data/data/PACS/kfold/', unlabeled_domain=unlabeled_domain,
        batch_size=args.batch_size, num_workers=4, get_domain_label=True)
    trainloader = zip(cycle(labeledloader), unlabeledloader)


net = models.__dict__[args.model](num_classes=7, num_domains=3)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    print2('Using CUDA..')

checkpoint_filename = './checkpoint/' + args.name + '_' + str(args.seed) + '.ckpt'


# Model
if args.resume:
    # Load checkpoint.
    assert os.path.exists(checkpoint_filename)
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    state = torch.load(checkpoint_filename, map_location=torch.device('cpu'))
    net.load_state_dict(state['net_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    best_acc = state['acc']
    start_epoch = state['epoch'] + 1
    rng_state = state['rng_state']
    torch.set_rng_state(rng_state)
else:
    # assert not os.path.exists(checkpoint_filename)
    print2('==> Building model..')

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')


trainer = supervisedtrainer()
# trainer = semisupervisedtrainer()

with open(logname, 'a') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(target_domain)

trainer.train(start_epoch)


