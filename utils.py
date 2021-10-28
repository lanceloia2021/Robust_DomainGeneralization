'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def construct_domains(domains, labeled: list, target: int):
    unlabeled_domain = deepcopy(domains)
    labeled_domain = []
    for i in labeled:
        labeled_domain.append(domains[i])
        unlabeled_domain.remove(domains[i])
    unlabeled_domain.remove(domains[target])
    target_domain = [domains[target]]
    return labeled_domain, unlabeled_domain, target_domain


def set_randseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print2(*args):
    PRINT = False
    if PRINT:
        print(args)
    pass


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a = lam * y
    y_b = (1-lam) * y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def visualize_feature(features, labels_class, labels_domain):
    X_tsne = TSNE().fit_transform(features)
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(111)
    plt.xticks([])
    plt.yticks([])

#    ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_class)
    index_labeled = (labels_domain == 0)
    features_labeled = X_tsne[index_labeled]
    class_labeled = labels_class[index_labeled]

    index_unlabeled_1 = (labels_domain == 1)
    features_unlabeled_1 = X_tsne[index_unlabeled_1]
    class_unlabeled_1 = labels_class[index_unlabeled_1]

    index_unlabeled_2 = (labels_domain == 2)
    features_unlabeled_2 = X_tsne[index_unlabeled_2]
    class_unlabeled_2 = labels_class[index_unlabeled_2]

    index_unlabeled_3 = (labels_domain == 3)
    features_unlabeled_3 = X_tsne[index_unlabeled_3]
    class_unlabeled_3 = labels_class[index_unlabeled_3]

    ax1.scatter(features_labeled[:, 0], features_labeled[:, 1], s=17, c=class_labeled, cmap='tab10', marker='o', label='Cartoon(Labeled)')
    ax1.scatter(features_unlabeled_1[:, 0], features_unlabeled_1[:, 1], s=17, c=class_unlabeled_1, cmap='tab10', marker='D', label='Photo(Unlabeled)')
    ax1.scatter(features_unlabeled_2[:, 0], features_unlabeled_2[:, 1], s=17, c=class_unlabeled_2, cmap='tab10', marker='*', label='Art(Unlabeled)')

    ax1.scatter(features_unlabeled_3[:, 0], features_unlabeled_3[:, 1], s=30, c='k', marker='o', label='Representatives')

    plt.legend()
    plt.savefig("x.eps")
    plt.show()


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
