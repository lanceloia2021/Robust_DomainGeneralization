import torch

domains = ['photo', 'art_painting', 'cartoon', 'sketch']
num_classes = 7
num_domains = len(domains)
cuda = torch.cuda.is_available()


ckptfilename = None
logname = None


def set_ckptfilenmae(args):
    global ckptfilename
    ckptfilename = './checkpoint/' + args.name + '_' + str(args.seed) + '.ckpt'


def set_logname(args):
    global logname
    logname = ('./results/' + args.name + '_' + str(args.seed) + '.csv')