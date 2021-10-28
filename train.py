#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
import csv
import os

import torch
import models
import setting

from itertools import cycle
from parser import parser
from dataloader import random_split_dataloader, unlabeled_dataloader
from utils import print2, set_randseed, construct_domains
from trainer.supervisedtrainer import SupervisedTrainer
from trainer.semisupervisedtrainer import SemiSupervisedTrainer

import torch.backends.cudnn as cudnn
import torch.optim as optim


def supervisedtrainer():
    st = SupervisedTrainer(model=net, optimizer=optimizer, args=args,
                           trainloader=labeledloader, valloader=valloader, testloader=testloader,)
    return st


def semisupervisedtrainer():
    sst = SemiSupervisedTrainer(model=net, optimizer=optimizer, args=args,
                                labeledloader=labeledloader, unlabeledloader=unlabeledloader, valloader=valloader, testloader=testloader,)
    return sst


# 原代码移动到parser.py中
args = parser()

if args.name is None:
    args.name = setting.domains[args.targetid]

if args.seed != 0:
    set_randseed(args.seed)


l, u, t = construct_domains(domains=setting.domains, labeled=args.labeled, target=args.target)

print('Labeled:', l, 'Unlabeled:', u, 'Target:', t)

labeledloader, valloader, testloader = random_split_dataloader(
        data_root='/home/wrq/data/data/PACS/kfold/', source_domain=l, target_domain=t,
        batch_size=args.batch_size, num_workers=4, get_domain_label=True)

if len(u) > 0:
    unlabeledloader = unlabeled_dataloader(
        data_root='/home/wrq/data/data/PACS/kfold/', unlabeled_domain=u,
        batch_size=args.batch_size, num_workers=4, get_domain_label=True)
    trainloader = zip(cycle(labeledloader), unlabeledloader)


net = models.__dict__[args.model](num_classes=setting.num_classes, num_domains=(setting.num_domains - 1))
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

if setting.cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

setting.set_ckptfilenmae(args=args)
setting.set_logname(args=args)

# Model
if args.resume:
    # Load checkpoint.
    assert os.path.exists(setting.ckptfilename)
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    state = torch.load(setting.ckptfilename, map_location=torch.device('cpu'))
    net.load_state_dict(state['net_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    start_epoch = state['epoch'] + 1
    best_acc = state['acc']
    rng_state = state['rng_state']
    torch.set_rng_state(rng_state)
else:
    # assert not os.path.exists(checkpoint_filename)
    print2('==> Building model..')
    start_epoch = 0
    best_acc = 0.

if not os.path.isdir('results'):
    os.mkdir('results')


trainer = supervisedtrainer()
# trainer = semisupervisedtrainer()

with open(setting.logname, 'a') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(t)

trainer.train(start_epoch)


