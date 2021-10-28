import csv
import os
import torch
import setting
from utils import print2, progress_bar


class BaseTrainer:
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.MSELoss = torch.nn.MSELoss()
        self.epoch = args.epoch
        self.lr = args.lr

        assert setting.ckptfilename is not None
        assert setting.logname is not None

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
            if setting.cuda:
                inputs, class_labels, domain_labels = inputs.cuda(), class_labels.cuda(), domain_labels.cuda()

            output_class_labels, output_domain_labels = self.model(inputs)
            # print(output_class_labels, class_labels)
            loss = self.CELoss(output_class_labels, class_labels)
            loss += self.CELoss(output_domain_labels, domain_labels)

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
            if setting.cuda:
                inputs, class_labels = inputs.cuda(), class_labels.cuda()

            output_class_labels, output_domain_labels = self.model(inputs)
            loss = self.CELoss(output_class_labels, class_labels)
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
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(state, setting.ckptfilename)

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

            with open(setting.logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, val_acc])

            self.adjust_learning_rate(epoch)
            progress_bar(epoch, self.epoch, 'Current val acc: %.3f, test acc: %.3f' % (val_acc, test_acc))
            # progress_bar(epoch, self.epoch, 'Current train_loss: %.3f' % train_loss)
        test_loss, test_acc = self._test(epoch)
        print("Final test_acc:", test_acc)
        # print("Val best loss: {:.4f}, at epoch {:d}, test acc: {:.4f}".format(self.best_val_loss, self.best_val_epoch, self.best_test_acc))
        pass