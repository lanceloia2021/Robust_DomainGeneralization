from torch.utils.data import DataLoader, random_split
import numpy as np
from copy import deepcopy
from dataset import DG_Dataset

from utils import print2


def random_split_dataloader(data_root, source_domain, target_domain, batch_size,
                            num_workers=4, get_domain_label=False):
    split_rate = 0.9
    source = DG_Dataset(root_dir=data_root, domain=source_domain, split='val',
                        get_domain_label=get_domain_label, get_cluster=False, color_jitter=True, min_scale=0.8)
    source_train, source_val = random_split(source, [int(len(source) * split_rate),
                                                     len(source) - int(len(source) * split_rate)])
    source_train = deepcopy(source_train)
    source_train.dataset.split = 'train'
    source_train.dataset.set_transform('train')
    source_train.dataset.get_domain_label = get_domain_label
    # source_train.dataset.get_cluster = get_cluster

    target_test = DG_Dataset(root_dir=data_root, domain=target_domain, split='test',
                             get_domain_label=False, get_cluster=False)

    print2('Train: {}, Val: {}, Test: {}'.format(len(source_train), len(source_val), len(target_test)))

    source_train = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    source_val = DataLoader(source_val, batch_size=32, shuffle=False, num_workers=num_workers)
    target_test = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return source_train, source_val, target_test


def unlabeled_dataloader(data_root, unlabeled_domain, batch_size, num_workers=4, get_domain_label=True):
    source_unlabeled = DG_Dataset(root_dir=data_root, domain=unlabeled_domain, split='train_unlabeled',
                             get_domain_label=get_domain_label, get_cluster=False)

    print2('Train unlabeled: {}'.format(len(source_unlabeled)))

    source_unlabeled = DataLoader(source_unlabeled, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return source_unlabeled


def random_split_one_domain(data, data_root, source_domain, batch_size,
                            get_domain_label=False, get_cluster=False, num_workers=4, color_jitter=True, min_scale=0.8):

    source = DG_Dataset(root_dir=data_root, domain=source_domain, split='val',
                        get_domain_label=False, get_cluster=False, color_jitter=color_jitter, min_scale=min_scale)
    source_train, source_val, source_test = random_split(source, [int(len(source) * 0.7),
                                                     int(len(source) * 0.15),
                                                                  len(source) - int(len(source) * 0.7) - int(len(source) * 0.15)])
    source_train = deepcopy(source_train)
    source_train.dataset.split = 'train'
    source_train.dataset.set_transform('train')
    source_train.dataset.get_domain_label = get_domain_label
    source_train.dataset.get_cluster = get_cluster


    print2('Train: {}, Val: {}, Test: {}'.format(len(source_train), len(source_val), len(source_test)))

    source_train = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    source_val = DataLoader(source_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    source_test = DataLoader(source_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return source_train, source_val, source_test