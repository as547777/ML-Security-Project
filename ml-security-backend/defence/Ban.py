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
    __desc__ = {
        "name": "BAN (Backdoor Adversarial Neuron)",
        "description": "A defense that identifies and neutralizes backdoors by finding adversarial neurons through feature masking and neuron perturbation.",
        "type": "Defense",
        "params": {
            "val_frac": {
                "label": "Validation Fraction",
                "tooltip": "Fraction of clean data to use for validation during BAN process.",
                "type": "number",
                "step": 0.05,
                "value": 0.1
            },
            "batch_size": {
                "label": "Batch Size",
                "tooltip": "Batch size for training and evaluation.",
                "type": "number",
                "step": 16,
                "value": 128
            },
            "mask_lambda": {
                "label": "Mask Lambda",
                "tooltip": "Regularization strength for feature mask.",
                "type": "number",
                "step": 0.05,
                "value": 0.25
            },
            "mask_lr": {
                "label": "Mask Learning Rate",
                "tooltip": "Learning rate for optimizing the feature mask.",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "eps": {
                "label": "Perturbation Epsilon",
                "tooltip": "Maximum perturbation magnitude for neuron noise.",
                "type": "number",
                "step": 0.05,
                "value": 0.3
            },
            "steps": {
                "label": "Perturbation Steps",
                "tooltip": "Number of steps for generating adversarial neuron perturbations.",
                "type": "number",
                "step": 1,
                "value": 1
            },
            "mask_epochs": {
                "label": "Mask Training Epochs",
                "tooltip": "Number of epochs to train the feature mask.",
                "type": "number",
                "step": 1,
                "value": 5
            }
        }
    }
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def execute(self,model, data, params, context):

        poisoned_model = context["model"]
        x_train = context["x_train"]
        y_train = context["y_train"]
        x_train_clean = context["x_train_clean"]  
        y_train_clean = context["y_train_clean"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        x_test_asr=context["x_test_asr"]
        y_test_asr=context["y_test_asr"]

        params = context.get("defense_params", {})
        val_frac = params.get("val_frac", 0.1)
        batch_size = params.get("batch_size", 128)
        mask_lambda = params.get("mask_lambda", 0.25)
        mask_lr = params.get("mask_lr", 0.01)
        eps = params.get("eps", 0.3)
        steps = params.get("steps", 1)
        mask_epochs = params.get("mask_epochs", 5)

        model=poisoned_model.model

        clean_dataset = TensorDataset(x_train_clean, y_train_clean)
       
        val_len = int(val_frac*len(clean_dataset))
        train_size = len(clean_dataset) - val_len
        
        train_set, val_set = torch.utils.data.random_split(clean_dataset, [train_size, val_len],generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
        
        val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=0)
        
        criterion = nn.CrossEntropyLoss()
        
        noise_params = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
        noise_opt = torch.optim.Adam(noise_params, lr=0.01) if noise_params else None
        
        
        model = self.perturbation_train(model,criterion, train_loader, noise_opt, val_loader, params)
        
        model.eval()
        _, acc_after = poisoned_model.predict((x_test, y_test))
        _, asr_after = poisoned_model.predict((x_test_asr, y_test_asr))
        
        poisoned_model.model = model
        context["model"] = poisoned_model
        
        context["acc_after"] = acc_after
        context["asr_after"] = asr_after
        
        return {"final_accuracy": acc_after,
            "final_asr": asr_after}

    
    def clip_noise(self, model,params, lower=None, upper=None):
        eps=params.get('eps',0.3)
        if lower is None:
            lower = -eps
        if upper is None:
            upper = eps
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


    def reset(self, model,params, rand_init):
        eps=params.get('eps',0.3)
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.reset(rand_init=rand_init, eps=eps)
        
    
    def fea_mask_gen(self, model,params):
        batch_size = params.get('batch_size', 128)
        with torch.no_grad():
            if hasattr(model, 'conv1'):
                in_channels = model.conv1.in_channels
                dummy_input = torch.randn(1, in_channels, 32, 32).to(self.device)
            else:
                dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            
            features = model.from_input_to_features(dummy_input, 0)
            mask_shape = (1,) + features.shape[1:] 
            rand_mask = torch.empty(mask_shape).uniform_(0, 1).to(self.device)
            mask = torch.nn.Parameter(rand_mask.clone().detach().requires_grad_(True))
        return mask
    
    def perturbation_train(self, model, criterion, data_loader, noise_opt, clean_test_loader,params):
        model.train()
        fea_mask = self.fea_mask_gen(model,params)
        opt_mask = torch.optim.Adam([fea_mask], lr=params.get('mask_lr',0.01))

        mepoch = params.get('mask_epochs',10)

        for m in range(mepoch):
            start = time.time()
            total_mask_value = 0
            total_positive_loss = 0
            total_negative_loss = 0
            batch_count=0
            
            for batch_idx, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                opt_mask.zero_grad()

                features = model.from_input_to_features(images, 0)

                pred_positive = model.from_features_to_output(fea_mask * features, 0)
                pred_negative = model.from_features_to_output((1 - fea_mask) * features, 0)
                mask_norm = torch.norm(fea_mask, 1)

                loss_positive = criterion(pred_positive, labels)
                loss_negative = criterion(pred_negative, labels) 
                loss = loss_positive - loss_negative + params.get('mask_lambda',0.25) * mask_norm / mask_norm.item()

                total_mask_value += mask_norm.item()
                total_positive_loss += loss_positive.item()
                total_negative_loss += loss_negative.item()
                batch_count+=1

                with torch.no_grad():
                    fea_mask.data.clamp_(0,1)

                loss.backward()
                opt_mask.step()

        print('\nGenerating noise perturbation.\n')
     
        
        eps=params.get('eps',0.3)
        steps=params.get('steps',1)
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            if eps> 0.0:
                self.reset(model, params=params,rand_init=True,)
                for _ in range(steps):
                    noise_opt.zero_grad()

                    self.include_noise(model)

                    features_noise = model.from_input_to_features(images, 0)
                    output_noise = model.from_features_to_output(features_noise, 0)

                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()
                    self.sign_grad(model)
                    noise_opt.step()
                    self.clip_noise(model,params)
        
        
        cl_test_loss, cl_test_acc = self.test(model=model, criterion=criterion, data_loader=clean_test_loader)
        print('Acc without mask (valid set): {:.4f}'.format(cl_test_acc))

        cl_test_loss, cl_test_acc = self.mask_test(model=model, criterion=criterion, data_loader=clean_test_loader, mask=(1-fea_mask.data))
        print('Acc with negative mask (valid set): {:.4f}'.format(cl_test_acc))
        model = self._fine_tune_model(model, data_loader, params)
        
        return model
    def _fine_tune_model(self, model, data_loader, params):
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(5):
            total_correct = 0
            total_samples = 0
            
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                pred = output.argmax(dim=1)
                total_correct += (pred == labels).sum().item()
                total_samples += labels.size(0)
            
            acc = total_correct / total_samples
            print(f'Fine-tune epoch {epoch+1}: Accuracy = {acc:.4f}')
        
        return model
    def test(self, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        loss = total_loss/len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc
    
    def mask_test(self, model, criterion, data_loader, mask):
        model.eval()
        total_correct = 0
        total_loss = 0.0
    
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                features = model.from_input_to_features(images, 0)
                if mask.dim() == 4 and mask.size(0) == 1:
                    output = model.from_features_to_output(mask * features, 0)
                else:
                    mask_expanded = mask.unsqueeze(0)
                    output = model.from_features_to_output(mask_expanded * features, 0)
                    
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc
