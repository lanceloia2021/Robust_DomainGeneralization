import setting
import torch
from trainer.basetrainer import BaseTrainer
from utils import print2


class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, optimizer, args,
                 trainloader, valloader, testloader):

        super().__init__(model, optimizer, args)

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def _train(self, epoch):
        print2('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss, correct, total = 0., 0, 0
        reg_loss = 0.

        for batch_idx, (inputs, class_labels, domain_labels) in enumerate(self.trainloader):
            if setting.cuda:
                inputs, class_labels, domain_labels = inputs.cuda(), class_labels.cuda(), domain_labels.cuda()

            output_class_labels, output_domain_labels = self.model(inputs)
            loss = self.CELoss(output_class_labels, class_labels)
            loss += self.CELoss(output_domain_labels, domain_labels)

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
