import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.adam import Adam
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import random
from interfaces.AbstractDefense import AbstractDefense
from interfaces.TrainTimeDefense import TrainTimeDefense
from .utils.util import *
from model.image.ImageModel import ImageModel


class ABL(AbstractDefense, TrainTimeDefense):
    __desc__ = {
    "display_name": "ABL (Adversarial Backdoor Learning)",
    "description": "A training-time defense that identifies and removes poisoned samples by exploiting loss-based separation between clean and backdoor data. The model is first trained using gradient ascent to amplify the effect of backdoor samples. Samples with anomalously low loss are isolated as suspicious backdoor candidates. The defense then performs a two-stage procedure: fine-tuning on the remaining clean data and unlearning on the isolated suspicious samples, which suppresses backdoor behavior while preserving clean accuracy.",
    "type": "Defense",
    "params": {
        'tuning_epochs': {
            'label': 'Gradient Ascent Epochs',
            'tooltip': 'Number of epochs for adversarial ascent training used to expose backdoor samples.',
            'type': 'number',
            'step': 1,
            'value': 10
        },
        'finetuning_epochs': {
            'label': 'Fine-tuning Epochs',
            'tooltip': 'Number of epochs used to fine-tune the model on the remaining clean data.',
            'type': 'number',
            'step': 1,
            'value': 10
        },
        'unlearning_epochs': {
            'label': 'Unlearning Epochs',
            'tooltip': 'Number of epochs for unlearning on isolated poisoned samples using gradient ascent.',
            'type': 'number',
            'step': 1,
            'value': 10
        },
        'isolation_ratio': {
            'label': 'Isolation Ratio',
            'tooltip': 'Fraction of training samples with the lowest loss treated as suspicious backdoor candidates.',
            'type': 'number',
            'step': 0.01,
            'value': 0.05
        },
        'gradient_ascent_type': {
            'label': 'Gradient Ascent Type',
            'tooltip': 'Method used for ascent training: Loss Gradient Ascent (LGA) or Flooding.',
            'type': 'select',
            'options': ['LGA', 'Flooding'],
            'value': 'LGA'
        },
    }
    }

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        self.device = device

    def execute(self, model, data, params, context):
        x_train_poisoned = context["x_train"]
        y_train_poisoned = context["y_train"]
        x_train_clean = context["x_train_clean"]  
        y_train_clean = context["y_train_clean"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]

        w_res = context['w_res']
        h_res = context['h_res']
        color_channels = context['color_channels']
        classes = context['classes']
        init_params = {"w_res": w_res, "h_res": h_res, "color_channels": color_channels, "classes": classes}
        
        model=context["model"]
        model.init(init_params)

        self.lr = context.get("learning_rate", 0.1)
        self.momentum = context.get("momentum", 0.9)

        defense_params = context.get("defense_params", params)
        
        test_clean_dataset = TensorDataset(x_test, y_test)
        test_bad_dataset = TensorDataset(x_test_asr, y_test_asr)
        test_clean_loader = DataLoader(test_clean_dataset, batch_size=defense_params.get("batch_size", 128), shuffle=False)
        test_bad_loader = DataLoader(test_bad_dataset, batch_size=defense_params.get("batch_size", 128), shuffle=False)
        
        poisoned_data, model_ascent = self.train(defense_params, model, (x_train_poisoned, y_train_poisoned),test_clean_loader,test_bad_loader,context)

        losses_idx = self.compute_loss_value(defense_params, poisoned_data, model_ascent)

        other_examples, isolation_examples = self.isolate_data(defense_params, poisoned_data, losses_idx)
        
        acc_clean, acc_asr = self.train_unlearn(defense_params, model_ascent,isolation_examples,other_examples,test_clean_loader,test_bad_loader,context)

        return {
            "final_accuracy": acc_clean[0], 
            "final_asr": acc_asr[0]
        }



    def compute_loss_value(self, params, poisoned_data, model_ascent):
        criterion = nn.CrossEntropyLoss().to(self.device)

        model_ascent.model.eval()
        losses_record = []

        example_data_loader = DataLoader(dataset=poisoned_data, batch_size=1, shuffle=False)

        for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
            
            img = img.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model_ascent.model(img)
                loss = criterion(output, target)

            losses_record.append(loss.item())

        losses_idx = np.argsort(np.array(losses_record))

        losses_record_arr = np.array(losses_record)
        print('Ten smallest loss values:', losses_record_arr[losses_idx[:10]])

        return losses_idx


    def isolate_data(self, params, poisoned_data, losses_idx):
        other_examples = []
        isolation_examples = []

        cnt = 0
        ratio = params.get("isolation_ratio", 0.05)

        example_data_loader = DataLoader(dataset=poisoned_data, batch_size=1, shuffle=False)
        perm = losses_idx[0: int(len(losses_idx) * ratio)]

        for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
            img = img.squeeze()
            target = target.squeeze()
        
            if img.ndim == 3:              
                img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
            elif img.ndim == 2:  
                img = img[:, :, None]      
                img = (img * 255)
                img = img.cpu().numpy()       


            target = target.cpu().numpy()

            if idx in perm:
                isolation_examples.append((img, target))
                cnt += 1
            else:
                other_examples.append((img, target))

        print('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
        print('Finish collecting {} other examples: '.format(len(other_examples)))

        return other_examples, isolation_examples
    
    def train_step(self, params, train_loader, model_ascent, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_ascent.model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            
            img = img.to(self.device)
            target = target.to(self.device)

            if params.get("gradient_ascent_type", "LGA") == 'LGA':
                output = model_ascent.model(img)
                loss = criterion(output, target)
                loss_ascent = torch.sign(loss - params.get("gamma", 0.5)) * loss

            elif params.get("gradient_ascent_type", "Flooding") == 'Flooding':
                output = model_ascent.model(img)
                loss = criterion(output, target)
                loss_ascent = (loss - params.get("flooding", 0.5)).abs() + params.get("flooding", 0.5)

            else:
                raise NotImplementedError

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss_ascent.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss_ascent.backward()
            optimizer.step()

            if idx % params.get("print_freq", 100) == 0:
                print('Epoch[{0}]:[{1:03}/{2:03}] '
                    'Loss:{losses.val:.4f}({losses.avg:.4f})  '
                    'Prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                    'Prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


    def test(self, params, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
        test_process = []
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_ascent.model.eval()

        for idx, (img, target) in enumerate(test_clean_loader, start=1):

            img = img.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model_ascent.model(img)
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg, losses.avg]

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target) in enumerate(test_bad_loader, start=1):
           
            img = img.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model_ascent.model(img)
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_bd = [top1.avg, top5.avg, losses.avg]

        print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
        print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

        return acc_clean, acc_bd


    def train(self, params, model_ascent, poisoned_data_tuple, test_clean_loader, test_bad_loader,context):        
        model_ascent.model.to(self.device)

        optimizer = torch.optim.SGD(model_ascent.model.parameters(), 
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=params.get("weight_decay", 1e-4),
                                    nesterov=True)
        
        criterion = nn.CrossEntropyLoss().to(self.device)

        tf_compose = transforms.Compose([
            transforms.ToTensor()
        ])
        x_train_poisoned, y_train_poisoned = poisoned_data_tuple
        poisoned_dataset = TensorDataset(x_train_poisoned, y_train_poisoned)
        poisoned_data_loader = DataLoader(dataset=poisoned_dataset,
                                          batch_size=params.get("batch_size", 128),
                                          shuffle=True)

        print('----------- Train Initialization --------------')
        for epoch in range(0, params.get("tuning_epochs", 10)):

            self.adjust_learning_rate(optimizer, epoch, params)

            if epoch == 0:
                self.test(params, test_clean_loader, test_bad_loader, model_ascent,
                                            criterion, epoch + 1)

            self.train_step(params, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

            print('testing the ascended model......')
            acc_clean, acc_bad = self.test(params, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)
            context["acc"]=acc_clean[0]
            context["acc_asr"]=acc_bad[0]
            
        return poisoned_dataset, model_ascent


    def adjust_learning_rate(self, optimizer, epoch, params):
        if epoch < params.get("tuning_epochs", 10):
            lr = self.lr
        else:
            lr = 0.01
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    

        
    def train_step_finetuing(self, params, train_loader, model_ascent, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_ascent.model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.to(self.device)
            target = target.to(self.device)

            output = model_ascent.model(img)

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % params.get("print_freq", 100) == 0:
                print('Epoch[{0}]:[{1:03}/{2:03}] '
                    'loss:{losses.val:.4f}({losses.avg:.4f})  '
                    'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                    'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


    def train_step_unlearning(self, params, train_loader, model_ascent, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_ascent.model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
        
            img = img.to(self.device)
            target = target.to(self.device)

            output = model_ascent.model(img)

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent training
            #added this to prevent agressive drops in accuracy
            torch.nn.utils.clip_grad_norm_(model_ascent.model.parameters(), max_norm=1.0)

            optimizer.step()

            if idx % params.get("print_freq", 100) == 0:
                print('Epoch[{0}]:[{1:03}/{2:03}] '
                    'loss:{losses.val:.4f}({losses.avg:.4f})  '
                    'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                    'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))
                
    
    def train_unlearn(self, params, model_ascent, isolation_examples, other_examples, test_clean_loader, test_bad_loader,context):
        model_ascent.model.to(self.device)

        optimizer = torch.optim.SGD(model_ascent.model.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=params.get("weight_decay", 1e-4),
                                    nesterov=True)

       
        criterion = nn.CrossEntropyLoss().to(self.device)

        sample_img, _ = isolation_examples[0]

        if sample_img.ndim == 3 and sample_img.shape[-1] == 1:
            sample_img = sample_img.squeeze(-1)

        if sample_img.ndim == 2:
            h, w = sample_img.shape
        else:
            _, h, w = sample_img.shape

        if h == 28 and w == 28:
            tf_compose_finetuning = transforms.Compose([
                transforms.ToTensor(),
            ])
            tf_compose_unlearning = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            tf_compose_finetuning = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(1,3)
            ])
            tf_compose_unlearning = transforms.Compose([transforms.ToTensor()])

        isolate_poisoned_data_tf = Dataset_npy(full_dataset=isolation_examples, transform=tf_compose_unlearning)
        isolate_poisoned_data_loader = DataLoader(dataset=isolate_poisoned_data_tf,
                                        batch_size=params.get("batch_size", 128),
                                        shuffle=True,)

        isolate_other_data_tf = Dataset_npy(full_dataset=other_examples, transform=tf_compose_finetuning)
        isolate_other_data_loader = DataLoader(dataset=isolate_other_data_tf,
                                                batch_size=params.get("batch_size", 128),
                                                shuffle=True,)


        if params.get("finetuning_ascent_model", True) == True:
            print('----------- Finetuning isolation model --------------')
            for epoch in range(0, params.get("finetuning_epochs", 10)):
                self.learning_rate_finetuning(optimizer, epoch, params)
                self.train_step_finetuing(params, isolate_other_data_loader, model_ascent, optimizer, criterion, epoch + 1)
                acc_clean,acc_bad=self.test(params, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)
                context["acc"]=acc_clean[0]
                context["acc_asr"]=acc_bad[0]

        print('----------- Model unlearning --------------')
        for epoch in range(0, params.get("unlearning_epochs", 10)):
            self.learning_rate_unlearning(optimizer, epoch, params)

            if epoch == 0:
                self.test(params, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch)
            else:
                self.train_step_unlearning(params, isolate_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

            print('testing the ascended model......')
            acc_clean, acc_bad = self.test(params, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)
        
        return acc_clean, acc_bad
    
    def learning_rate_finetuning(self, optimizer, epoch, params):
        if epoch < 5:
            lr = params.get("lr_finetuning_init", 0.1)
        elif epoch < 8:
            lr = 0.01
        else:
            lr = 0.001
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def learning_rate_unlearning(self, optimizer, epoch, params):
        if epoch < params.get("unlearning_epochs", 5):
            lr = 0.001
        else:
            lr = 0.0001
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr