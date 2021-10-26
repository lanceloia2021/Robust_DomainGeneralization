import argparse

def parser():
    _parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    _parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    _parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    _parser.add_argument('--model', default=None, type=str,
                        help='model type (default: None)')
    _parser.add_argument('--name', default=None, type=str, help='name of run')
    _parser.add_argument('--seed', default=0, type=int, help='random seed')
    _parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    _parser.add_argument('--epoch', default=60, type=int,
                        help='total epochs to run')
    _parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')
    _parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    _parser.add_argument('--alpha', default=0.2, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    _parser.add_argument('--targetid', type=int)
    _parser.add_argument('--labeledid', type=str)

    args = _parser.parse_args()
    args.labeledid = [int(i) for i in args.labeledid.split(',')]
    return args


