import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from interfaces.AbstractDeffense import AbstractDefense
from .utils.util import *
import time
from torch.utils.data import DataLoader, TensorDataset
from model.image.networks.NoisyBatchNorm2d import NoisyBatchNorm2d

class ANP(AbstractDefense):
    __desc__ = {
    "name": "ANP",
    "description": "Adversarial Neuron Pruning defense that mitigates backdoor attacks by identifying and pruning suspicious neurons.",
    "type": "Defense",
    "params": {
        "nb_iter": {
            "label": "Number of Optimization Iterations",
            "tooltip": "Number of iterations used to optimize neuron mask scores during adversarial perturbation training.",
            "type": "number",
            "step": 1,
            "value": 10
        },
        "anp_eps": {
            "label": "Adversarial Noise Magnitude",
            "tooltip": "Maximum magnitude of adversarial neuron perturbation.",
            "type": "number",
            "step": 0.01,
            "value": 0.1
        },
        "anp_alpha": {
            "label": "Clean vs Robust Loss Balance",
            "tooltip": "Controls the weighting between clean training loss and adversarial loss.",
            "type": "number",
            "step": 0.05,
            "value": 0.5
        },
        "pruning_max": {
            "label": "Maximum Pruning Rate",
            "tooltip": "Maximum proportion of neurons allowed to be pruned during the defense process.",
            "type": "number",
            "step": 0.05,
            "value": 0.5
        },
        "max_CA_drop": {
            "label": "Maximum Clean Accuracy Drop",
            "tooltip": "Maximum tolerated drop in clean accuracy.",
            "type": "number",
            "step": 0.01,
            "value": 0.5
        }
        }
    }

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        self.device = device

    def execute(self, model, data, params, context):
        x_train = context["x_train_clean"]  
        y_train = context["y_train_clean"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]
        
        poisoned_model = context["model"]
        defense_params = context["defense_params"]
        
        self.lr = context.get("learning_rate", 0.1)
        self.anp_eps = defense_params.get("anp_eps", 0.1)
        self.anp_steps = defense_params.get("anp_steps", 1)
        self.anp_alpha = defense_params.get("anp_alpha", 0.5)
        self.nb_iter = defense_params.get("nb_iter", 10)
        self.print_every = defense_params.get("print_every", 10)
        self.pruning_by = defense_params.get("pruning_by", "threshold")
        self.pruning_max = defense_params.get("pruning_max", 0.5)
        self.pruning_step = defense_params.get("pruning_step", 0.05)
        self.max_CA_drop = defense_params.get("max_CA_drop", 0.05)
        
        train_dataset = TensorDataset(x_train, y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        test_dataset = TensorDataset(x_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        self.poison_transform = PoisonTransform(x_test_asr, y_test_asr)
        
        self.model = poisoned_model.model.to(self.device)
        
        mask_scores=self.optimize_mask()
        results=self.prune_neuron(mask_scores)
        
        clean_acc, clean_asr=self.test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)
        
        return {
            "final_accuracy": clean_acc,
            "final_asr": clean_asr
        }
    

    def optimize_mask(self):
        net = self.model
        net.train()
        
        parameters = list(net.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=self.lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=self.anp_eps / self.anp_steps)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        
        print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        nb_repeat = int(np.ceil(self.nb_iter / self.print_every))
        
        mask_scores = []
        
        for i in range(nb_repeat):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            
            train_loss, train_acc = self.mask_train(model=net, criterion=criterion, data_loader=self.train_loader,
                mask_opt=mask_optimizer, noise_opt=noise_optimizer)
            
            cl_test_loss, cl_test_acc = self.anp_test(model=net, criterion=criterion, data_loader=self.test_loader)
            
            po_test_loss, po_test_acc = self.anp_test(model=net, criterion=criterion, data_loader=self.test_loader,
                poison_transform=self.poison_transform)
            
            end = time.time()
            print(f'{(i + 1) * self.print_every} \t {lr:.3f} \t {end - start:.1f} \t 'f'{train_loss:.4f} \t {train_acc:.4f} \t {po_test_loss:.4f} \t 'f'{po_test_acc:.4f} \t {cl_test_loss:.4f} \t {cl_test_acc:.4f}')
            
            for name, param in net.named_parameters():
                if "neuron_mask" in name:
                    mask_values = param.data.cpu().numpy().flatten()
                    for idx, val in enumerate(mask_values):
                        mask_scores.append((name, idx, float(val)))
        
        return mask_scores

    def prune_neuron(self, mask_scores):
        mask_values = sorted(mask_scores, key=lambda x: float(x[2]))
        
        self.ori_CA, self.ori_ASR = self.test(self.model, self.test_loader, poison_test=True,poison_transform=self.poison_transform)
        if self.pruning_by == 'threshold':
            results = self.evaluate_by_threshold(mask_values)
        else:
            results = self.evaluate_by_number(mask_values)
        
        return results
    
    def mask_train(self, model, criterion, mask_opt, noise_opt, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            nb_samples += images.size(0)

            if self.anp_eps > 0.0:
                self.reset(model, rand_init=True, anp_eps=self.anp_eps)
                for _ in range(self.anp_steps):
                    noise_opt.zero_grad()

                    self.include_noise(model)
                    output_noise = model(images)
                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()
                    self.sign_grad(model)
                    noise_opt.step()

            mask_opt.zero_grad()
            if self.anp_eps > 0.0:
                self.include_noise(model)
                output_noise = model(images)
                loss_rob = criterion(output_noise, labels)
            else:
                loss_rob = 0.0

            self.exclude_noise(model)
            output_clean = model(images)
            loss_nat = criterion(output_clean, labels)
            loss = self.anp_alpha * loss_nat + (1-self.anp_alpha) * loss_rob

            pred = output_clean.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            mask_opt.step()
            self.clip_mask(model)

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc

    def evaluate_by_number(self, mask_values):
        results = []
        nb_max = int(np.ceil(self.pruning_max))
        nb_step = int(np.ceil(self.pruning_step))
        
        original_masks = {}
        for name, param in self.model.named_parameters():
            if "neuron_mask" in name:
                original_masks[name] = param.data.clone()
        
        for start in range(0, nb_max + 1, nb_step):
            i = start
            for i in range(start, min(start + nb_step, len(mask_values))):
                self.pruning(self.model, mask_values[i])
            
            layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
                
            CA, ASR = self.test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)
            
            results.append(f'pruned neurons = {start}, CA = {CA * 100:.2f} \t ASR = {ASR * 100:.2f}\n')
            
            if self.ori_CA - CA > self.max_CA_drop: 
                for name, param in self.model.named_parameters():
                    if "neuron_mask" in name and name in original_masks:
                        param.data = original_masks[name].clone()
                break
            
        return results


    def evaluate_by_threshold(self, mask_values):
        results = []
        thresholds = np.arange(0, self.pruning_max + self.pruning_step, self.pruning_step)
        start = 0
        
        original_masks = {}
        for name, param in self.model.named_parameters():
            if "neuron_mask" in name:
                original_masks[name] = param.data.clone()
        
        for threshold in thresholds:
            idx = start
            for idx in range(start, len(mask_values)):
                if float(mask_values[idx][2]) <= threshold:
                    self.pruning(self.model, mask_values[idx])
                    start += 1
                else:
                    break
            
            print(f'pruned neurons num = {start}, threshold = {threshold:.3f}')
            CA, ASR = self.test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)
            
            results.append(f'pruned neurons num = {start}, threshold = {threshold:.3f} \t CA = {CA * 100:.2f} \t ASR = {ASR * 100:.2f}\n')
            
            if self.ori_CA - CA > self.max_CA_drop: 
                for name, param in self.model.named_parameters():
                    if "neuron_mask" in name and name in original_masks:
                        param.data = original_masks[name].clone()
                break
        
        return results

    def clip_mask(self,model, lower=0.0, upper=1.0):
        params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)


    def sign_grad(self,model):
        noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
        for p in noise:
            p.grad.data = torch.sign(p.grad.data)


    def perturb(self,model, is_perturbed=True):
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


    def reset(self,model, rand_init, anp_eps):
        for name, module in model.named_modules():
            if isinstance(module, NoisyBatchNorm2d):
                module.reset(rand_init=rand_init, eps=anp_eps)


    def anp_test(self,model, criterion, data_loader, poison_transform=None):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                if poison_transform is not None:
                    images, labels = poison_transform.transform(images, labels)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc
    
    def test(self, model, test_loader, poison_test=False, poison_transform=None):
        model.eval()
        total_correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if poison_test and poison_transform is not None:
                    images, labels = poison_transform.transform(images, labels)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                total_correct += predicted.eq(labels).sum().item()
        
        accuracy = total_correct / total
        return accuracy, 1 - accuracy if poison_test else accuracy

    def pruning(self, net, neuron):
        layer_name = neuron[0]  
        neuron_idx = int(neuron[1])
        
        for name, param in net.named_parameters():
            if name == layer_name:
                if param.data.ndim == 1:
                    param.data[neuron_idx] = 0.0
                elif param.data.ndim == 3: 
                    param.data[neuron_idx, 0, 0] = 0.0
                else:
                    flat_data = param.data.view(-1)
                    if neuron_idx < len(flat_data):
                        flat_data[neuron_idx] = 0.0
                        param.data = flat_data.view(param.data.shape)