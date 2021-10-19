import csv
import os
import numpy as np
from itertools import cycle
import torch
import torch.nn as nn
from torch.cuda import amp


from utils import print2, progress_bar

def log_loss(prob_p, prob_q):
    eps = 1e-20
    logq = torch.log(prob_q + eps)
    return -torch.mean(torch.sum(prob_p.detach() * logq, dim=1))


class BaseTrainer:
    def __init__(self, model, optimizer, criterion,
                 cuda, epoch, lr, n_classes,
                 ckptfilename, logname):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda = cuda
        self.epoch = epoch
        self.lr = lr
        self.n_classes = n_classes
        self.ckptfilename = ckptfilename
        self.logname = logname

        self.best_val_loss = 1e128
        self.best_val_epoch = -1
        self.best_test_acc = 0.0
        self.update_flag = False

    def _train(self, epoch):
        raise NotImplementedError

    def _test(self, epoch, mode):
        self.model.eval()
        test_loss, correct, total = 0., 0, 0

        if mode == 'test':
            loader = self.testloader
            text = 'Test'
        elif mode == 'val':
            loader = self.valloader
            text = 'Val'
        else:
            raise KeyError

        for batch_idx, (inputs, targets) in enumerate(loader):
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            loss, inc_total, inc_correct = self.iteration(inputs, targets)

            test_loss += loss.item()
            total += inc_total
            correct += inc_correct

            # progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total,
            #                 correct, total))

        avg_loss = test_loss / (batch_idx + 1)
        print2('%s: Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (text, avg_loss,
                 100. * correct / total, correct, total))

        acc = 100. * correct / total

        if mode == 'val':
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                self.best_val_epoch = epoch
                self.checkpoint(acc, epoch)
                print2('Val loss best: {:.4f}'.format(self.best_val_loss))
                self.update_flag = True
            elif epoch == self.epoch - 1:
                self.checkpoint(acc, epoch)
        return test_loss / batch_idx, 100. * correct / total

    def iteration(self, inputs, targets):
        outputs = self.model(inputs)
        labels, domain_labels = outputs[0], outputs[1]
        loss = self.criterion(labels, targets)

        _, predicted = torch.max(labels.data, 1)
        inc_total = targets.size(0)
        inc_correct = predicted.eq(targets.data).cpu().sum().float()
        return loss, inc_total, inc_correct

    def checkpoint(self, acc, epoch):
        # Save checkpoint.
        print2('Saving..')
        state = {
            'net_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, self.ckptfilename)

    def adjust_learning_rate(self, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = self.lr
        if epoch == 30:
            lr /= 10
        if epoch == 50:
            lr /= 10

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, start_epoch):
        for epoch in range(start_epoch, self.epoch):
            train_loss, reg_loss, train_acc = self._train(epoch)
            val_loss, val_acc = self._test(epoch, mode='val')
            test_loss, test_acc = self._test(epoch, mode='test')

            with open(self.logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, val_acc, test_acc])

            # update_flag = True when val_loss decrease
            if self.update_flag:
                self.update_flag = False
                self.best_test_acc = test_acc

            self.adjust_learning_rate(epoch)
            progress_bar(epoch, self.epoch, 'Current test acc: %.3f' % test_acc)
        print("Val best loss: {:.4f}, at epoch {:d}, test acc: {:.4f}".format(self.best_val_loss, self.best_val_epoch, self.best_test_acc))
        pass


class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, optimizer,
                 trainloader, valloader, testloader,
                 cuda, epoch, lr, n_classes,
                 ckptfilename, logname):

        super().__init__(model, optimizer, nn.CrossEntropyLoss(),
                 cuda, epoch, lr, n_classes,
                 ckptfilename, logname)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def _train(self, epoch):
        print2('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss, correct, total = 0., 0, 0
        reg_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            loss, inc_total, inc_correct = self.iteration(inputs, targets)

            train_loss += loss.item()
            total += inc_total
            correct += inc_correct

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # progress_bar(batch_idx, len(labeledloader), 'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
            #                 100.*correct/total, correct, total))
        print2('Train: Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                 100.*correct/total, correct, total))
        return train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total

    pass


class SemiSupervisedTrainer(BaseTrainer):
    def __init__(self, model, optimizer,
                 labeledloader, unlabeledloader, valloader, testloader,
                 cuda, epoch, lr, n_classes,
                 ckptfilename, logname):

        super().__init__(model, optimizer, nn.CrossEntropyLoss(),
                 cuda, epoch, lr, n_classes,
                 ckptfilename, logname)
        self.labeledloader = labeledloader
        self.unlabeledloader = unlabeledloader
        self.valloader = valloader
        self.testloader = testloader

        self.scaler = amp.GradScaler()

    def _train(self, epoch):
        print2('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss, correct, total = 0., 0, 0
        reg_loss = 0.
        batch_idx = 0
        for (inputs, targets), (u_inputs, _) in zip(cycle(self.labeledloader), self.unlabeledloader):
            # mix-up data
            lam = np.random.beta(1.0, 1.0)
            lam = 1 - lam if lam < 0.5 else lam
            mix_inputs = lam * inputs + (1 - lam) * u_inputs
            if self.cuda:
                inputs, targets, mix_inputs = inputs.cuda(), targets.cuda(), mix_inputs.cuda()

            outputs = self.model(inputs)
            mix_outputs = self.model(mix_inputs)

            preds, mix_preds = outputs[0], mix_outputs[0]

            u_outputs = self.model(u_inputs)
            u_preds = u_outputs[0]
            u_targets = torch.max(u_preds, 1)[1]

            loss_labeled = self.criterion(preds, targets)
            loss_unlabeled = self.criterion(u_preds, u_targets)
            loss_mix = lam * loss_labeled + (1-lam) * loss_unlabeled

            loss = loss_labeled + 0.1 * loss_mix
            # print("loss_labeled: ", loss_labeled.data, "loss_mix", loss_mix.data, "total:", loss.data)

            _, predicted = torch.max(preds.data, 1)
            inc_total = inputs.size(0)
            inc_correct = predicted.eq(targets.data).cpu().sum().float()

            train_loss += loss.item()
            total += inc_total
            correct += inc_correct

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # progress_bar(batch_idx, len(labeledloader), 'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
            #                 100.*correct/total, correct, total))
            batch_idx += 1

        # print('Train: Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
        #       % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
        #          100.*correct/total, correct, total))
        print2('Train: Loss: %.3f ' % (train_loss/(batch_idx+1)))
        return train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total

    pass

