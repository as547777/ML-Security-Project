import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from interfaces.TrainTimeDefense import TrainTimeDefense
from interfaces.AbstractDeffense import AbstractDefense
from .utils.util import *
from model.image.ImageModel import ImageModel
import copy

class CBD(AbstractDefense,TrainTimeDefense):
    __desc__ = {
    "display_name": "CBD (Contrastive Backdoor Defense)",
    "description": "A training-time defense that learns to disentangle clean and backdoor feature representations by jointly training a clean model and a backdoored reference model. A discriminator (disentangler) is optimized to distinguish hidden activations of clean and poisoned models, forcing the clean model to suppress backdoor-specific neurons. The method reduces ASR while preserving clean accuracy through contrastive representation learning.",
    "type": "Defense",
    "params": {
        "training_epochs": {
            "label": "Number of Training Epochs",
            "tooltip": "Number of epochs used to train the backdoored model.",
            "type": "number",
            "step": 1,
            "value": 10
        },
        "tuning_epochs": {
            "label": "Number of Tuning Epochs",
            "tooltip": "Number of epochs used to fine-tune the clean model with disentanglement.",
            "type": "number",
            "step": 1,
            "value": 10
        },
        "disentangle": {
            "label": "Enable Disentanglement",
            "tooltip": "If enabled, the disentanglement estimator is trained to separate clean and backdoor representations.",
            "type": "checkbox",
            "value": True
        }
    }
    }


    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        self.device = device

    def execute(self, model, data, params, context):
        x_train = context["x_train"]
        y_train = context["y_train"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        x_test_asr=context["x_test_asr"]
        y_test_asr=context["y_test_asr"]

        self.lr = context.get("learning_rate", 0.1)
        self.momentum = context.get("momentum", 0.9)


        w_res = context['w_res']
        h_res = context['h_res']
        color_channels = context['color_channels']
        classes = context['classes']
        init_params={"w_res":w_res,"h_res":h_res,"color_channels":color_channels,"classes":classes}
        backdoor_model=copy.deepcopy(context["model"])
        backdoor_model.init(init_params)
        clean_model=copy.deepcopy(context["model"])
        clean_model.init(init_params=init_params)

        poisoned_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), params.get("batch_size", 128), shuffle=True)
        test_clean_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), params.get("batch_size", 128), shuffle=False)
        test_bad_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_asr, y_test_asr), params.get("batch_size", 128), shuffle=False)
        
        defense_params = context.get("defense_params", params)
        context["model"]=clean_model
        clean_acc, asr, final_clean_acc, final_asr = self.train(defense_params, backdoor_model, clean_model, poisoned_data_loader, (test_clean_loader, test_bad_loader))
        context["acc"]=clean_acc[0]
        context["acc_asr"]=asr[0]
        return {
            "final_accuracy": final_clean_acc,
            "final_asr": final_asr,
        }
    
    def train_step_clean(self, params, train_loader, model_clean, model_backdoor, disen_estimator, optimizer, adv_optimizer,
                         criterion, epoch):
        criterion1 = nn.CrossEntropyLoss(reduction='none')
        losses = AverageMeter()
        disen_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        model_clean.model.train()
        model_backdoor.model.eval()

        if params.get("disentangle", True):
            for idx, (img, target) in enumerate(train_loader, start=1):
                img = img.type(torch.FloatTensor)
                img = img.to(self.device)

                output1, z_hidden = model_clean.model(img, True)
                with torch.no_grad():
                    output2, r_hidden = model_backdoor.model(img, True)

                r_hidden, z_hidden = r_hidden.detach(), z_hidden.detach()
                dis_loss = -disen_estimator(r_hidden, z_hidden)
                disen_losses.update(dis_loss.item(), img.size(0))
                adv_optimizer.zero_grad()
                dis_loss.backward()
                adv_optimizer.step()

                disen_estimator.spectral_norm()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.type(torch.FloatTensor)
            img = img.to(self.device)
            target = target.to(self.device)

            output1, z_hidden = model_clean.model(img, True)
            with torch.no_grad():
                output2, r_hidden = model_backdoor.model(img, True)
                loss_bias = criterion1(output2, target)
                loss_d = criterion1(output1, target).detach()

            r_hidden = r_hidden.detach()
            dis_loss = disen_estimator(r_hidden, z_hidden)

            weight = loss_bias / (loss_d + loss_bias + 1e-8)
            weight = weight * weight.shape[0] / torch.sum(weight)
            loss = torch.mean(weight * criterion1(output1, target))
            
            if params.get("disentangle", True):
                loss += dis_loss

            prec1, prec5 = accuracy(output1, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % params.get("print_freq", 100) == 0:
                print('Clean Epoch[{0}]:[{1:03}/{2:03}] '
                      'loss:{losses.val:.4f}({losses.avg:.4f})  '
                      'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                      'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
                                                                     top1=top1, top5=top5))

    def train_step_backdoor(self, params, train_loader, model_backdoor, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_backdoor.model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.type(torch.FloatTensor)
            img = img.to(self.device)
            target = target.to(self.device)

            output = model_backdoor.model(img)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % params.get("print_freq", 100) == 0:
                print('Backdoor Epoch[{0}]:[{1:03}/{2:03}] '
                      'loss:{losses.val:.4f}({losses.avg:.4f})  '
                      'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                      'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
                                                                     top1=top1, top5=top5))

    def test(self, params, test_clean_loader, test_bad_loader, model_clean, criterion, epoch):
        test_process = []
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        weight_record = np.array([])
        criterion1 = nn.CrossEntropyLoss(reduction='none')

        model_clean.model.eval()

        for idx, (img, target) in enumerate(test_clean_loader, start=1):
            img = img.type(torch.FloatTensor)
            img = img.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model_clean.model(img)
                loss = criterion(output, target)
                loss1 = criterion1(output, target)
                weight_record = np.concatenate([weight_record, loss1.cpu().numpy()])

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg, losses.avg]

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target) in enumerate(test_bad_loader, start=1):
            img = img.type(torch.FloatTensor)
            img = img.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model_clean.model(img)
                loss = criterion(output, target)
                loss1 = criterion1(output, target)
                weight_record = np.concatenate([weight_record, loss1.cpu().numpy()])

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_bd = [top1.avg, top5.avg, losses.avg]

        print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
        print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

        return acc_clean, acc_bd


    def train(self, params, model_backdoor, model_clean, poisoned_data_loader, test_data):

        model_backdoor.model.to(self.device)
        model_clean.model.to(self.device)

        hidden_dim = model_clean.model.nChannels
        
        disen_estimator = DisenEstimator(hidden_dim, hidden_dim, dropout=0.2)
        disen_estimator.to(self.device)

        adv_params = list(disen_estimator.parameters())
        adv_optimizer = torch.optim.Adam(adv_params, lr=0.2)
        adv_scheduler = torch.optim.lr_scheduler.StepLR(adv_optimizer, step_size=20, gamma=0.1)
        
        optimizer = torch.optim.SGD(model_clean.model.parameters(), lr=self.lr, momentum=self.momentum,
            weight_decay=params.get("weight_decay", 1e-4), nesterov=True)
        
        optimizer_backdoor = torch.optim.SGD(model_backdoor.model.parameters(), lr=self.lr, momentum=self.momentum,
            weight_decay=params.get("weight_decay", 1e-4), nesterov=True)

        criterion = nn.CrossEntropyLoss().to(self.device)
        
        test_clean_loader, test_bad_loader = test_data

        print('----------- Training Backdoored Model --------------')
        for epoch in range(0, params.get("training_epochs",5)):
            self.learning_rate(optimizer, epoch, params)
            self.train_step_backdoor(params, poisoned_data_loader, model_backdoor, optimizer_backdoor, criterion, epoch + 1)
            clean_acc, asr_acc=self.test(params, test_clean_loader, test_bad_loader, model_backdoor, criterion, epoch + 1)
                
        print('----------- Training Clean Model --------------')
        for epoch in range(0, params.get("tuning_epochs", 10)):
            self.learning_rate(optimizer, epoch, params)
            adv_scheduler.step()
            self.train_step_clean(params, poisoned_data_loader, model_clean, model_backdoor, disen_estimator, 
                                  optimizer, adv_optimizer, criterion, epoch + 1)
            
            acc_clean, acc_bd = self.test(params, test_clean_loader, test_bad_loader, model_clean, 
                                          criterion, epoch + 1)
            
            if epoch == params.get("tuning_epochs", 10) - 1:
                final_clean_acc = acc_clean[0] 
                final_asr = acc_bd[0]  
        
        return clean_acc, asr_acc,final_clean_acc, final_asr

    def learning_rate(self, optimizer, epoch, params):
        if epoch < 4:
            lr = 0.01
        elif epoch < 7:
            lr = 0.005
        else:
            lr = 0.0005
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



