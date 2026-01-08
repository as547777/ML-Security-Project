import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler
import numpy as np
import time
from collections import OrderedDict
from interfaces.AbstractDeffense import AbstractDefense
from model.image.networks.NoisyBatchNorm2d import NoisyBatchNorm2d

class Ban(AbstractDefense):
    __desc__ = {
        "display_name": "BAN",
        "description": "Identifies and neutralizes backdoors by finding adversarial neurons through feature masking and neuron perturbation.",
        "type": "Defense",
        "use_case": "Mitigate backdoor attacks in trained neural networks.",
        "category": "image",
        "params": {
            "batch_size": {
                "label": "Batch Size",
                "tooltip": "Batch size for training and evaluation.",
                "type": "number",
                "step": 16,
                "value": 128
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
            "rob_lambda": {
                "label": "Robustness Lambda",
                "tooltip": "Weight for the adversarial neuron loss component.",
                "type": "number",
                "step": 0.05,
                "value": 0.5
            },
            "epochs": {
                "label": "Training Epochs",
                "tooltip": "Number of epochs to train the defense.",
                "type": "number",
                "step": 1,
                "value": 10
            },
            "schedule": {
                "label": "Learning Rate Schedule",
                "tooltip": "Epoch milestones for learning rate decay.",
                "type": "list",
                "value": [50, 75]
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
        batch_size = params.get("batch_size", 128)
        eps = params.get("eps", 0.3)
        steps = params.get("steps", 1)
        rob_lambda = params.get("rob_lambda", 0.5)
        epochs = params.get("epochs",10)
        schedule = params.get("schedule", [50, 75])
        
        orig_train = TensorDataset(x_train_clean, y_train_clean)
        clean_test = TensorDataset(x_test, y_test)
        poison_train = TensorDataset(x_train, y_train)
        poison_test = TensorDataset(x_test_asr, y_test_asr)

        clean_train_size = int(0.95 * len(orig_train))
        sub_train_size = len(orig_train) - clean_train_size
        sub_train, clean_train = random_split(
            orig_train, [sub_train_size, clean_train_size], 
            generator=torch.Generator().manual_seed(0)
        )

        poison_train_loader = DataLoader(poison_train, batch_size=batch_size, shuffle=True, num_workers=8)
        poison_test_loader = DataLoader(poison_test, batch_size=batch_size, num_workers=8)
        clean_test_loader = DataLoader(clean_test, batch_size=batch_size, num_workers=8)
        clean_val_loader = DataLoader(sub_train, batch_size=batch_size, shuffle=False, num_workers=8)

        net = poisoned_model.model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=context.get("learning_rate", 0.01),momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedule, gamma=0.1)

        parameters = list(net.named_parameters())
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=eps/steps)
        
        for epoch in range(1, epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]['lr']
            
            train_loss, train_acc = self.train(
                model=net, criterion=criterion, optimizer=optimizer,
                data_loader=clean_val_loader, noise_opt=noise_optimizer,
                eps=eps, steps=steps, rob_lambda=rob_lambda
            )
            
            self.exclude_noise(net)
            scheduler.step()
            end = time.time()
            print(
                '{:d} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f}'.format(epoch, lr, end - start, train_loss, train_acc))
        
        cl_test_loss, cl_test_acc = self.test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = self.test(model=net, criterion=criterion, data_loader=poison_test_loader)

        return {"final_accuracy": cl_test_acc,
            "final_asr": po_test_acc}

    
    def clip_noise(self, model, eps, lower=None, upper=None):
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


    def include_noise(self, model):
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.include_noise()


    def exclude_noise(self, model):
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.exclude_noise()


    def reset(self, model, rand_init, eps):
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.reset(rand_init=rand_init, eps=eps)

    def train(self, model, criterion, optimizer, data_loader, noise_opt, eps, steps, rob_lambda):
        model.train()
        total_correct = 0
        total_loss = 0.0
        num_samples = 0

        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            if eps > 0.0 and noise_opt is not None:
                self.reset(model, rand_init=True, eps=eps)
                for _ in range(steps):
                    noise_opt.zero_grad()
                    self.include_noise(model)
                    output_noise = model(images)
                    loss_noise = -criterion(output_noise, labels)
                    loss_noise.backward()
                    self.sign_grad(model)
                    noise_opt.step()
                    self.clip_noise(model, eps)

            if eps > 0.0 and noise_opt is not None:
                self.include_noise(model)
                output_noise = model(images)
                loss_rob = criterion(output_noise, labels)
            else:
                loss_rob = 0

            self.exclude_noise(model)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels) + rob_lambda * loss_rob

            pred = output.data.max(1)[1]
            batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            total_correct += batch_correct
            total_loss += loss.item()
            num_samples += labels.size(0)

            loss.backward()
            optimizer.step()

        loss = total_loss/len(data_loader)
        acc = float(total_correct)/num_samples
        return loss, acc
    
    
       
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
    

    def fea_mask_gen(self, model, batch_size):
        x = torch.rand(batch_size, 3, 32, 32).to(self.device)

        with torch.no_grad():
            features = model.from_input_to_features(x)
        rand_mask = torch.empty_like(features).uniform_(0, 1).to(self.device)
        mask = torch.nn.Parameter(rand_mask.clone().detach().requires_grad_(True))
        return mask
    
    def gene_mask(self, model, criterion, data_loader, mask_lr, num_classes):
        model.train()
        batch_size = data_loader.batch_size
        fea_mask = self.fea_mask_gen(model, batch_size)
        opt_mask = torch.optim.Adam([fea_mask], lr=mask_lr)

        mepoch = 20

        for m in range(mepoch):
            start = time.time()
            total_mask_value = 0
            total_positive_loss = 0
            total_negative_loss = 0
            batch_count = 0
            for batch_idx, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                opt_mask.zero_grad()

                features = model.from_input_to_features(images)
                
                pred_positive = model.from_features_to_output(fea_mask * features)
                pred_negative = model.from_features_to_output((1 - fea_mask) * features)
                
                mask_norm = torch.norm(fea_mask, 1)

                loss_positive = criterion(pred_positive, labels)
                loss_negative = criterion(pred_negative, labels) 
                loss = loss_positive - loss_negative + 0.25 * mask_norm / mask_norm.item()

                total_mask_value += mask_norm.item()
                total_positive_loss += loss_positive.item()
                total_negative_loss += loss_negative.item()
                batch_count += 1

                fea_mask.data = torch.clamp(fea_mask.data, min=0, max=1)

                loss.backward()
                opt_mask.step()

            if batch_count > 0:
                l_pos = total_positive_loss / batch_count
                l_neg = total_negative_loss / batch_count
                avg_mask = total_mask_value / batch_count
            else:
                l_pos = l_neg = avg_mask = 0
                
            end = time.time()
            print('mask epoch: {:d}'.format(m),
                '\tmask_norm: {:.4f}'.format(avg_mask), 
                '\tloss_positive:  {:.4f}'.format(l_pos),
                '\tloss_negative:  {:.4f}'.format(l_neg),
                '\ttime:  {:.4f}'.format(end - start))

        return fea_mask.data

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
