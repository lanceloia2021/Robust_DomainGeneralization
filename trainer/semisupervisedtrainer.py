import numpy as np
from itertools import cycle
import torch
import setting

from trainer.basetrainer import BaseTrainer
from utils import print2


class SemiSupervisedTrainer(BaseTrainer):
    def __init__(self, model, optimizer, args,
                 labeledloader, unlabeledloader, valloader, testloader):

        super().__init__(model, optimizer, args)
        self.labeledloader = labeledloader
        self.unlabeledloader = unlabeledloader
        self.valloader = valloader
        self.testloader = testloader

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
            if setting.cuda:
                mix_inputs = mix_inputs.cuda()
                class_labels, domain_labels = class_labels.cuda(), domain_labels.cuda()
                u_inputs = u_inputs.cuda()
                u_domain_labels = u_domain_labels.cuda()

            # 获取伪标签
            outputs_u_class_labels, outputs_u_domain_labels = self.model(u_inputs)
            u_class_labels = torch.max(outputs_u_class_labels, 1)[1]

            output_mix_class_labels, output_mix_domain_labels = self.model(mix_inputs)

            idx1 = torch.zeros(domain_labels.shape[0], setting.num_domains - 1).cuda()
            idx1.scatter_(1, domain_labels.reshape(-1, 1), 1)
            idx2 = torch.zeros(u_domain_labels.shape[0], setting.num_domains - 1).cuda()
            idx2.scatter_(1, (u_domain_labels + 1).reshape(-1, 1), 1)
            domain_targets = lam * idx1 + (1-lam) * idx2

            # class_label Loss
            loss1_labeled = self.CELoss(output_mix_class_labels, class_labels)
            loss1_unlabeled = self.CELoss(output_mix_class_labels, u_class_labels)
            loss1_mix = lam * loss1_labeled + (1-lam) * loss1_unlabeled

            # domain_label Loss
            loss2_mix = self.MSELoss(output_mix_domain_labels, domain_targets)

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

