import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler
import numpy as np
import time
from collections import OrderedDict
from interfaces.AbstractDeffense import AbstractDefense
from defence.NoisyBatchNorm2d import NoisyBatchNorm2d

class Ban(AbstractDefense):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.args = {
            'dataset':'MNIST',
            'num_classes':10,
            'batch_size':128,
            'print_every':50,
            'val_frac':0.1,
            'acc_threshold':0.25,
            'mask_lambda':0.25,
            'mask_lr':0.01,
            'eps':0.3,
            'steps':1,
            'epoch': 5,
            'lr': 0.01,
            'rob_lambda': 0.2,
            'schedule': [2, 4],
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'fine_tune': True,
        }
        
    
    def execute(self, context):

        model = context["model"]
        x_train = context["x_train"]
        y_train = context["y_train"]
        x_test = context["x_test"]
        y_test = context["y_test"]

        ban_config = context.get('ban_config')
        if ban_config:
            self.args.update(ban_config)
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.FloatTensor(x_train)
            y_train = torch.LongTensor(y_train)
            x_test = torch.FloatTensor(x_test)
            y_test = torch.LongTensor(y_test)
        
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        valid_frac = 0.1
        sub_train_frac = 0.05
        clean_train_frac = 0.85
        
        val_len = int(valid_frac * len(train_dataset))
        temp_len = len(train_dataset) - val_len
        valia_set, temp_set = random_split(dataset=train_dataset, lengths=[val_len, temp_len], generator=torch.Generator().manual_seed(0))
        
        sub_len = int(sub_train_frac * len(train_dataset))
        clean_len = len(train_dataset) - val_len - sub_len
        sub_train, clean_train = random_split(dataset=temp_set, lengths=[sub_len, clean_len], generator=torch.Generator().manual_seed(0))
        
        print('Number of samples in sub_train: ', len(sub_train))
        print('Number of samples in clean_train: ', len(clean_train))
        
        random_sampler_train = RandomSampler(data_source=sub_train,replacement=True,num_samples=self.args['print_every']*self.args['batch_size'])
        
        sub_train_loader =DataLoader(sub_train,batch_size=self.args['batch_size'],shuffle=False,sampler=random_sampler_train, num_workers=0)
        valid_loader = DataLoader(valia_set,batch_size=self.args['batch_size'], shuffle=False, num_workers=0)
        clean_test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'],num_workers=0)

        net = model.model
        net=net.to(self.device)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        cl_test_loss, cl_test_acc = self.test(model=net, criterion=criterion, data_loader=clean_test_loader)
        print('Acc of the checkpoint (clean test set): {:.2f}'.format(cl_test_acc))

        non_perturb_acc=cl_test_acc

        parameters = list(net.named_parameters())
        noise_params = [v for n, v in parameters if "neuron_noise" in n]

        if len(noise_params) == 0:
            print("Model doesn't have NoisyBatchNorm2d layers. Backdoor detection may not work properly.")
            ban_results = {
                "original_accuracy": non_perturb_acc,
                "perturbed_accuracy": 0,
                "backdoor_detected": False,
                "error": "Model doesn't have NoisyBatchNorm2d layers"
            }
            context["ban_results"] = ban_results
            context["backdoor_detected"] = False
            return context
        
        noise_optimizer=torch.optim.SGD(noise_params, lr=self.args['eps'] / self.args['steps'])
        total_start=time.time()
        
        perturb_test_acc, l_pos, l_neg, fea_mask = self.perturbation_train(model=net, criterion=criterion, data_loader=sub_train_loader, noise_opt=noise_optimizer, clean_test_loader=valid_loader)
        
        backdoor_detected = perturb_test_acc < self.args['acc_threshold']

        fine_tuned=False
        fine_tuned_acc=None

        if backdoor_detected and self.args.get('fine_tune', True):
            print("\n Backdoor detected, starting fine-tuning correction")
            
            optimizer = torch.optim.SGD(net.parameters(),lr=self.args['lr'],momentum=self.args['momentum'],weight_decay=self.args['weight_decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args['schedule'], gamma=0.1)
            
            clean_train_loader = DataLoader(clean_train, batch_size=self.args['batch_size'], shuffle=True, num_workers=0)
            
            for epoch in range(1, self.args['epoch'] + 1):
                start = time.time()
                lr = optimizer.param_groups[0]['lr']
                
                train_loss, train_acc = self.fine_tune_train(model=net, criterion=criterion, optimizer=optimizer,data_loader=clean_train_loader, noise_opt=noise_optimizer,mask=fea_mask)
                
                self.exclude_noise(net)
                cl_test_loss, cl_test_acc = self.test(model=net, criterion=criterion, data_loader=clean_test_loader)
                
                scheduler.step()
                end = time.time()
                
                print(f'Epoch {epoch} \t LR: {lr:.3f} \t Time: {end-start:.1f} \t 'f'Train Loss: {train_loss:.4f} \t Train Acc: {train_acc:.4f} \t 'f'Clean Test Acc: {cl_test_acc:.4f}')
            
            fine_tuned = True
            fine_tuned_acc = cl_test_acc
        
        total_end = time.time()

        print('Total time: {:.4f}'.format(total_end - total_start))
        
        ban_results = {
            "original_accuracy": non_perturb_acc,
            "perturbed_accuracy": perturb_test_acc,
            "positive_loss": l_pos,
            "negative_loss": l_neg,
            "backdoor_detected": backdoor_detected,
            "fine_tuned": fine_tuned,
            "fine_tuned_accuracy": fine_tuned_acc,
            "detection_time": total_end - total_start
        }
        
        context["ban_results"] = ban_results
        context["backdoor_detected"] = backdoor_detected
        
        return context
    
    def fine_tune_train(self, model, criterion, optimizer, data_loader, noise_opt, mask=None):
        model.train()
        total_correct = 0
        total_loss = 0.0

        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            if self.args['eps'] > 0.0:
                self.reset(model, rand_init=True)
                for _ in range(self.args['steps']):
                    noise_opt.zero_grad()
                    self.include_noise(model)
                    
                    if mask is not None:
                        features_noise = model.from_input_to_features(images, 0)
                        output_noise = model.from_features_to_output(mask * features_noise, 0)
                    else:
                        output_noise = model(images)
                    
                    loss_noise = -criterion(output_noise, labels)
                    loss_noise.backward()
                    self.sign_grad(model)
                    noise_opt.step()

            self.exclude_noise(model)
            optimizer.zero_grad()
            
            if mask is not None:
                features = model.from_input_to_features(images, 0)
                output = model.from_features_to_output(mask * features, 0)
            else:
                output = model(images)
            
            loss = criterion(output, labels)
            
            if self.args['eps'] > 0.0:
                self.include_noise(model)
                if mask is not None:
                    features_noise = model.from_input_to_features(images, 0)
                    output_noise = model.from_features_to_output(mask * features_noise, 0)
                else:
                    output_noise = model(images)
                loss_rob = criterion(output_noise, labels)
                loss = loss+self.args['rob_lambda']*loss_rob
            
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc
    
    
    def clip_noise(self, model, lower=None, upper=None):
        if lower is None:
            lower = -self.args['eps']
        if upper is None:
            upper = self.args['eps']
        params = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)
    
    def sign_grad(self, model):
        noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
        for p in noise:
            if p.grad is not None:
                p.grad.data = torch.sign(p.grad.data)
    
    def perturb(self, model, is_perturbed=True):
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.perturb(is_perturbed=is_perturbed)

    def include_noise(self,model):
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.include_noise()


    def exclude_noise(self,model):
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.exclude_noise()


    def reset(self, model, rand_init):
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.reset(rand_init=rand_init, eps=self.args['eps'])
        
    
    def fea_mask_gen(self, model):
        if self.args['dataset'] == 'MNIST':
            x = torch.rand(self.args['batch_size'], 1, 28, 28).to(self.device)
        
        fea_shape = model.from_input_to_features(x, 0)
        rand_mask = torch.empty_like(fea_shape[0]).uniform_(0, 1).to(self.device)
        mask = torch.nn.Parameter(rand_mask.clone().detach().requires_grad_(True))
        return mask
    
    def perturbation_train(self, model, criterion, data_loader, noise_opt, clean_test_loader):
        model.train()
        fea_mask = self.fea_mask_gen(model)
        opt_mask = torch.optim.Adam([fea_mask], lr=self.args['mask_lr'])

        mepoch = 10 

        for m in range(mepoch):
            start = time.time()
            total_mask_value = 0
            total_positive_loss = 0
            total_negative_loss = 0
            
            for batch_idx, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                opt_mask.zero_grad()

                features = model.from_input_to_features(images, 0)

                pred_positive = model.from_features_to_output(fea_mask * features, 0)
                pred_negative = model.from_features_to_output((1 - fea_mask) * features, 0)
                mask_norm = torch.norm(fea_mask, 1)

                loss_positive = criterion(pred_positive, labels)
                loss_negative = criterion(pred_negative, labels) 
                loss = loss_positive - loss_negative + self.args['mask_lambda'] * mask_norm / mask_norm.item()

                total_mask_value += mask_norm.item()
                total_positive_loss += loss_positive.item()
                total_negative_loss += loss_negative.item()

                fea_mask.data = torch.clamp(fea_mask.data, min=0, max=1)

                loss.backward()
                opt_mask.step()

            if batch_idx > 0:
                l_pos = total_positive_loss / (batch_idx + 1)
                l_neg = total_negative_loss / (batch_idx + 1) 
            else:
                l_pos = total_positive_loss
                l_neg = total_negative_loss
                
            end = time.time()
            
            print('mask epoch: {:d}'.format(m),'\tmask_norm: {:.4f}'.format(total_mask_value / (batch_idx + 1)), '\tloss_positive: {:.4f}'.format(l_pos),'\tloss_negative: {:.4f}'.format(l_neg),'\ttime: {:.4f}'.format(end - start))

        print('\nGenerating noise perturbation.\n')
        start = time.time()
                
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            if self.args['eps'] > 0.0:
                self.reset(model, rand_init=True)
                for _ in range(self.args['steps']):
                    noise_opt.zero_grad()

                    self.include_noise(model)

                    features_noise = model.from_input_to_features(images, 0)
                    output_noise = model.from_features_to_output(features_noise, 0)

                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()
                    self.sign_grad(model)
                    noise_opt.step()
        
        end = time.time()

        cl_test_loss, cl_test_acc = self.test(model=model, criterion=criterion, data_loader=clean_test_loader)
        print('Acc without mask (valid set): {:.4f}'.format(cl_test_acc))

        cl_test_loss, cl_test_acc = self.mask_test(model=model, criterion=criterion, data_loader=clean_test_loader, mask=(1-fea_mask.data))
        print('Acc with negative mask (valid set): {:.4f}'.format(cl_test_acc))

        return cl_test_acc, l_pos, l_neg, fea_mask.data
    
    def test(self, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        
        reg=np.zeros([self.args['num_classes']])
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

                for j in range(images.shape[0]):
                    p = pred[j].item()
                    reg[p] += 1

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        print('Prediction distribution: ', reg)
        print('Prediction targets to: ', np.argmax(reg))
        return loss, acc
    
    def mask_test(self, model, criterion, data_loader, mask):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        reg = np.zeros([self.args['num_classes']])
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                features = model.from_input_to_features(images, 0)
                output = model.from_features_to_output(mask*features, 0)

                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

                for j in range(images.shape[0]):
                    p = pred[j].item()
                    reg[p] += 1
                
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        print('Prediction distribution: ', reg)
        print('Prediction targets to: ', np.argmax(reg))
        return loss, acc
