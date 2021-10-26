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

    def _val(self, epoch):
        self.model.eval()
        val_loss, correct, total = 0., 0, 0
        for batch_idx, (inputs, class_labels, domain_labels) in enumerate(self.valloader):
            if self.cuda:
                inputs, class_labels, domain_labels = inputs.cuda(), class_labels.cuda(), domain_labels.cuda()

            output_class_labels, output_domain_labels = self.model(inputs)
            # print(output_class_labels, class_labels)
            loss = self.criterion(output_class_labels, class_labels)
            loss += self.criterion(output_domain_labels, domain_labels)

            _, predicted = torch.max(output_class_labels.data, 1)
            inc_total = class_labels.size(0)
            inc_correct = predicted.eq(class_labels.data).cpu().sum().float()

            val_loss += loss.item()
            total += inc_total
            correct += inc_correct

        avg_loss = val_loss / (batch_idx + 1)
        acc = 100. * correct / total
        print2('%s: Loss: %.3f | Acc: %.3f%% (%d/%d)'
               % ("Val", avg_loss, acc, correct, total))

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_val_epoch = epoch
            self.checkpoint(acc, epoch)
            print2('Val loss best: {:.4f}'.format(self.best_val_loss))
            self.update_flag = True
        elif epoch == self.epoch - 1:
            self.checkpoint(acc, epoch)
        return val_loss / batch_idx, acc

    def _test(self, epoch):
        self.model.eval()
        test_loss, correct, total = 0., 0, 0

        for batch_idx, (inputs, class_labels) in enumerate(self.testloader):
            if self.cuda:
                inputs, class_labels = inputs.cuda(), class_labels.cuda()

            output_class_labels, output_domain_labels = self.model(inputs)
            loss = self.criterion(output_class_labels, class_labels)
            loss += 0.

            _, predicted = torch.max(output_class_labels.data, 1)
            inc_total = class_labels.size(0)
            inc_correct = predicted.eq(class_labels.data).cpu().sum().float()

            test_loss += loss.item()
            total += inc_total
            correct += inc_correct

        avg_loss = test_loss / (batch_idx + 1)
        acc = 100. * correct / total
        print2('%s: Loss: %.3f | Acc: %.3f%% (%d/%d)'
               % ("Test", avg_loss, acc, correct, total))

        return test_loss / batch_idx, acc

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
        if epoch == 18:
            lr /= 10
        if epoch == 30:
            lr /= 10

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, start_epoch):
        for epoch in range(start_epoch, self.epoch):
            train_loss, reg_loss, train_acc = self._train(epoch)
            val_loss, val_acc = self._val(epoch)
            test_loss, test_acc = self._test(epoch)

            with open(self.logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, val_acc])

            self.adjust_learning_rate(epoch)
            progress_bar(epoch, self.epoch, 'Current val acc: %.3f, test acc: %.3f' % (val_acc, test_acc))
            # progress_bar(epoch, self.epoch, 'Current train_loss: %.3f' % train_loss)
        test_loss, test_acc = self._test(epoch)
        print("Final test_acc:", test_acc)
        # print("Val best loss: {:.4f}, at epoch {:d}, test acc: {:.4f}".format(self.best_val_loss, self.best_val_epoch, self.best_test_acc))
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

        for batch_idx, (inputs, class_labels, domain_labels) in enumerate(self.trainloader):
            if self.cuda:
                inputs, class_labels, domain_labels = inputs.cuda(), class_labels.cuda(), domain_labels.cuda()

            output_class_labels, output_domain_labels = self.model(inputs)
            loss = self.criterion(output_class_labels, class_labels)
            loss += self.criterion(output_domain_labels, domain_labels)

            _, predicted = torch.max(output_class_labels.data, 1)
            inc_total = class_labels.size(0)
            inc_correct = predicted.eq(class_labels.data).cpu().sum().float()

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


fuck = 0

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
        self.mseloss = torch.nn.MSELoss()

        self.scaler = amp.GradScaler()

    def _train(self, epoch):
        print2('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss, correct, total = 0., 0, 0
        reg_loss = 0.
        batch_idx = 0
        for (inputs, class_labels, domain_labels), (u_inputs, cant_use, u_domain_labels) in zip(cycle(self.labeledloader), self.unlabeledloader):
            # mix-up data
            lam = np.random.beta(1.0, 1.0)
            # lam = 0.75
            lam = 1 - lam if lam < 0.5 else lam
            mix_inputs = lam * inputs + (1 - lam) * u_inputs
            if self.cuda:
                mix_inputs = mix_inputs.cuda()
                class_labels, domain_labels = class_labels.cuda(), domain_labels.cuda()
                u_inputs = u_inputs.cuda()
                u_domain_labels = u_domain_labels.cuda()

                # 获取伪标签
            outputs_u_class_labels, outputs_u_domain_labels = self.model(u_inputs)
            u_class_labels = torch.max(outputs_u_class_labels, 1)[1]

            output_mix_class_labels, output_mix_domain_labels = self.model(mix_inputs)

            # domain_targets = torch.zeros(domain_labels.shape)
            ###### 3 是domain的类别数-1
            domain_classes = 4
            idx1 = torch.zeros(domain_labels.shape[0], domain_classes-1).cuda()
            # print(idx1)
            idx1.scatter_(1, domain_labels.reshape(-1, 1), 1)
            idx2 = torch.zeros(u_domain_labels.shape[0], domain_classes-1).cuda()
            idx2.scatter_(1, (u_domain_labels + 1).reshape(-1, 1), 1)
            # print(domain_labels)
            # print(idx1)
            domain_targets = lam * idx1 + (1-lam) * idx2
            global fuck
            if fuck == 0:
                print(domain_targets)
                fuck = 1

            # 有标签部分的Loss
            loss1_labeled = self.criterion(output_mix_class_labels, class_labels)
            # loss2_labeled = self.criterion(output_mix_domain_labels, domain_labels)
            # 无标签部分的Loss
            loss1_unlabeled = self.criterion(output_mix_class_labels, u_class_labels)
            # loss2_unlabeled = self.criterion(output_mix_domain_labels, u_domain_labels)

            loss1_mix = lam * loss1_labeled + (1-lam) * loss1_unlabeled
            # loss2_mix = lam * loss2_labeled + (1 - lam) * loss2_unlabeled
            loss2_mix = self.mseloss(output_mix_domain_labels, domain_targets)

            loss = loss1_mix + loss2_mix
            print("loss: %.2f, \t%.2f\t%.2f\t%.2f" % (loss.data, loss1_labeled.data, loss1_unlabeled.data, loss2_mix))

            _, predicted = torch.max(output_mix_class_labels.data, 1)
            inc_total = inputs.size(0)
            inc_correct = predicted.eq(class_labels.data).cpu().sum().float()

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

