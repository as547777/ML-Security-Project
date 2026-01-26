"""
Grond Attack - Backdoor Attack with Comprehensive Stealthiness
Implementation following CCS 2025 paper:
"Towards Backdoor Stealthiness in Model Parameter Space"

Paper: https://arxiv.org/abs/2501.05928
GitHub: https://github.com/xiaoyunxxy/parameter_backdoor

Key Components:
1. UPGD (Universal PGD) Trigger Generation - Input-space stealthiness
2. Adversarial Backdoor Injection (ABI) - Parameter-space stealthiness
3. Feature-space stealthiness as by-product
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from interfaces.AbstractAttack import AbstractAttack
from interfaces.TrainTimeAttack import TrainTimeAttack


class GrondTriggerGenerator:
    """
    Universal PGD (UPGD) Trigger Generator
    
    Generates imperceptible adversarial perturbations as backdoor triggers.
    Uses surrogate model to create universal perturbations that:
    1. Are imperceptible (small epsilon)
    2. Contain semantic information from target class
    3. Transfer across different architectures
    """
    
    def __init__(self, target_class, eps=8/255, num_steps=100, step_size=None):
        """
        Args:
            target_class: Target class for backdoor
            eps: L_inf perturbation budget (default: 8/255)
            num_steps: Number of PGD iterations
            step_size: Step size for PGD (default: eps/5)
        """
        self.target_class = target_class
        self.eps = eps
        self.num_steps = num_steps
        self.step_size = step_size if step_size else eps / 5
        self.upgd = None
        
    def generate(self, surrogate_model, data_loader, device):
        """
        Generate UPGD trigger using surrogate model
        
        Algorithm 1 from paper - UPGD Generation
        
        Args:
            surrogate_model: Well-trained model on clean data
            data_loader: Training data loader
            device: torch device
            
        Returns:
            upgd: Universal perturbation tensor
        """
        print(f"\n[Grond] Generating UPGD trigger for target class {self.target_class}...")
        
        surrogate_model.eval()
        
        first_batch = next(iter(data_loader))
        if isinstance(first_batch, (tuple, list)):
            sample_input = first_batch[0]
        else:
            sample_input = first_batch
            
        data_shape = sample_input[0].shape
        
        delta = torch.zeros(1, *data_shape).to(device)
        delta.uniform_(-self.eps, self.eps)

        delta.requires_grad = True
        
        optimizer = torch.optim.SGD([delta], lr=self.step_size)
        
        criterion = nn.CrossEntropyLoss()
        
        for step in range(self.num_steps):
            total_loss = 0
            num_batches = 0
            
            for batch_data in data_loader:
                if isinstance(batch_data, (tuple, list)):
                    inputs = batch_data[0].to(device)
                else:
                    inputs = batch_data.to(device)
                
                perturbed = inputs + delta
                perturbed = torch.clamp(perturbed, 0, 1)
                
                targets = torch.full((inputs.size(0),), self.target_class, 
                                   dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                outputs = surrogate_model(perturbed)
                loss = criterion(outputs, targets)
                loss.backward()
                
                total_loss += loss.item()
                num_batches += 1
                
                with torch.no_grad():
                    delta.grad.sign_()
                    delta -= self.step_size * delta.grad
                    delta.clamp_(-self.eps, self.eps)
                    delta.grad.zero_()
                
                if num_batches >= 20:
                    break
            
            if (step + 1) % 20 == 0 or step == 0:
                avg_loss = total_loss / num_batches
                print(f"  Step {step+1}/{self.num_steps} - Loss: {avg_loss:.4f}")
        
        self.upgd = delta.detach().clone()
        print(f"[Grond] UPGD generation complete - Perturbation budget: {self.eps:.4f}")
        
        return self.upgd
    
    def apply_trigger(self, images):
        """Apply UPGD trigger to images"""
        if self.upgd is None:
            raise ValueError("UPGD not generated yet. Call generate() first.")
        
        device = images.device
        triggered = images + self.upgd.to(device)
        return torch.clamp(triggered, 0, 1)


class AdversarialBackdoorInjection:
    """
    Adversarial Backdoor Injection (ABI)
    
    Core innovation of Grond - adaptively limits parameter changes during training
    to achieve parameter-space stealthiness.
    
    Key idea: Prune prominent backdoor neurons iteratively during training
    to spread backdoor across entire network instead of few neurons.
    """
    
    def __init__(self, u=3.0):
        """
        Args:
            u: Threshold coefficient for neuron pruning (default: 3.0)
               Neurons with UCLC > mean + u*std are considered prominent
        """
        self.u = u
        
    def compute_channel_lipschitz(self, conv_weight, bn_weight, bn_var):
        """
        Compute Upper bound of Channel Lipschitz Condition (UCLC)
        
        Paper Equation 3 and Section 3.3
        
        Args:
            conv_weight: Convolutional layer weights [out_ch, in_ch, k, k]
            bn_weight: BatchNorm gamma parameter [out_ch]
            bn_var: BatchNorm running variance [out_ch]
            
        Returns:
            channel_lips: UCLC values for each channel
        """
        num_channels = conv_weight.shape[0]
        channel_lips = []
        
        bn_std = torch.sqrt(bn_var + 1e-5)
        
        for idx in range(num_channels):
            w = conv_weight[idx].reshape(conv_weight.shape[1], -1)
            w = w * (bn_weight[idx] / bn_std[idx]).abs()
            
            try:
                _, s, _ = torch.svd(w.cpu())
                channel_lips.append(s.max())
            except:
                channel_lips.append(torch.norm(w))
        
        return torch.tensor(channel_lips)
    
    def prune_prominent_neurons(self, model):
        """
        Prune prominent backdoor neurons
        
        Paper Equation 3 - core of ABI
        
        For each layer:
        1. Compute UCLC for each neuron
        2. Identify prominent neurons (UCLC > mean + u*std)
        3. Set their weights to layer mean
        
        Args:
            model: Neural network model
        """
        with torch.no_grad():
            conv_layer = None
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    conv_layer = module
                    
                elif isinstance(module, nn.BatchNorm2d) and conv_layer is not None:
                    channel_lips = self.compute_channel_lipschitz(
                        conv_layer.weight.data,
                        module.weight.data,
                        module.running_var.data
                    )
                    
                    threshold = channel_lips.mean() + self.u * channel_lips.std()
                    prominent_idx = torch.where(channel_lips > threshold)[0]
                    
                    if len(prominent_idx) > 0:
                        mean_weight = module.weight.data.mean()
                        mean_bias = module.bias.data.mean()
                        
                        module.weight.data[prominent_idx] = mean_weight
                        module.bias.data[prominent_idx] = mean_bias
                    
                    conv_layer = None


class Grond(AbstractAttack, TrainTimeAttack):
    """
    Grond - Comprehensive Backdoor Stealthiness Attack
    
    Main attack class implementing CCS 2025 paper methodology.
    
    Components:
    1. UPGD Trigger Generation (Input-space stealthiness)
    2. Adversarial Backdoor Injection (Parameter-space stealthiness)
    3. Feature-space stealthiness (by-product)
    
    Attack flow:
    1. Train surrogate model on clean data
    2. Generate UPGD trigger
    3. Poison training data (clean-label)
    4. Train with ABI (prune prominent neurons each epoch)
    """
    
    __desc__ = {
        "display_name": "Grond",
        "description": "State-of-the-art backdoor attack with comprehensive stealthiness in input, feature, and parameter space. Uses UPGD triggers and Adversarial Backdoor Injection (ABI) to evade detection.",
        "type": "White-box attack",
        "time": "Online poisoning",
        "params": {
            "target_label": {
                "label": "Target label",
                "tooltip": "Target class for backdoor misclassification",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 2
            },
            "poison_rate": {
                "label": "Poison rate",
                "tooltip": "Fraction of target class samples to poison (paper uses 0.5-5%)",
                "type": "number",
                "step": 0.01,
                "value": 0.05
            },
            "eps": {
                "label": "Epsilon (perturbation budget)",
                "tooltip": "L_inf norm bound for UPGD trigger (8/255 for CIFAR-10)",
                "type": "number",
                "step": 0.001,
                "value": 0.031
            },
            "upgd_steps": {
                "label": "UPGD generation steps",
                "tooltip": "Number of iterations for UPGD generation (paper uses 100)",
                "type": "number",
                "step": 10,
                "value": 100
            },
            "u_threshold": {
                "label": "ABI threshold (u)",
                "tooltip": "Coefficient for prominent neuron detection (paper uses 3.0)",
                "type": "number",
                "step": 0.1,
                "value": 3.0
            },
            "surrogate_epochs": {
                "label": "Surrogate training epochs",
                "tooltip": "Epochs to train surrogate model for UPGD generation",
                "type": "number",
                "step": 10,
                "value": 50
            },
            "backdoor_epochs": {
                "label": "Backdoor training epochs",
                "tooltip": "Epochs to train backdoored model (paper uses 200)",
                "type": "number",
                "step": 10,
                "value": 200
            },
            "apply_abi": {
                "label": "Apply ABI",
                "tooltip": "Enable Adversarial Backdoor Injection during training",
                "type": "select",
                "options": ["Enable", "Disable"],
                "value": "Yes"
            },
            "abi_frequency": {
                "label": "ABI frequency (epochs)",
                "tooltip": "Apply ABI every N epochs (1 = every epoch)",
                "type": "number",
                "step": 1,
                "value": 1
            }
        }
    }
    
    def __init__(
        self,
        target_label=2,
        poison_rate=0.05,
        eps=8/255,
        upgd_steps=100,
        u_threshold=3.0,
        surrogate_epochs=50,
        backdoor_epochs=200,
        apply_abi=True,
        abi_frequency=1
    ):
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.eps = eps
        self.upgd_steps = upgd_steps
        self.u_threshold = u_threshold
        self.surrogate_epochs = surrogate_epochs
        self.backdoor_epochs = backdoor_epochs
        self.apply_abi = apply_abi
        self.abi_frequency = abi_frequency
        
        self.trigger_generator = None
        self.abi = None
        self.skip_retraining = True
        
    def train_surrogate_model(self, model_arch, train_data, device):
        """
        Train surrogate model for UPGD generation
        
        Paper Section 3.3: "UPGD is generated on a well-trained surrogate model"
        
        Args:
            model_arch: Model architecture (same or different from victim)
            train_data: Clean training data (x_train, y_train)
            device: torch device
            
        Returns:
            surrogate_model: Trained model
        """
        print("\n[Grond] Training surrogate model for UPGD generation...")
        
        x_train, y_train = train_data
        
        surrogate_model = model_arch.to(device)
        surrogate_model.train()
        
        optimizer = torch.optim.SGD(
            surrogate_model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.surrogate_epochs * 0.5), 
                       int(self.surrogate_epochs * 0.75)],
            gamma=0.1
        )
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True
        )
        
        for epoch in range(self.surrogate_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = surrogate_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            acc = 100.0 * correct / total
            scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.surrogate_epochs} - "
                      f"Loss: {total_loss/len(loader):.4f} - "
                      f"Acc: {acc:.2f}%")
        
        print(f"[Grond] Surrogate model trained - Final accuracy: {acc:.2f}%")
        return surrogate_model
    
    def poison_training_data(self, x_train, y_train):
        """
        Poison training data using UPGD trigger (clean-label)
        
        Paper Section 3.3: "backdoor is injected during training by poisoning 
        some training data from the target class"
        
        Args:
            x_train: Training images
            y_train: Training labels
            
        Returns:
            x_poisoned: Poisoned training images
            y_poisoned: Training labels (unchanged - clean-label)
            poison_indices: Indices of poisoned samples
        """
        print(f"\n[Grond] Poisoning training data (clean-label)...")
        
        target_indices = (y_train == self.target_label).nonzero(as_tuple=True)[0]
        
        num_poison = int(len(target_indices) * self.poison_rate)
        
        if num_poison == 0:
            raise ValueError("No samples to poison. Increase poison_rate.")
        
        perm = torch.randperm(len(target_indices))
        poison_indices = target_indices[perm[:num_poison]]
        
        x_poisoned = x_train.clone()
        
        x_poisoned[poison_indices] = self.trigger_generator.apply_trigger(
            x_train[poison_indices]
        )
        
        print(f"[Grond] Poisoned {num_poison} samples from target class "
              f"{self.target_label} ({self.poison_rate*100:.1f}%)")
        
        y_poisoned = y_train.clone()
        
        return x_poisoned, y_poisoned, poison_indices
    
    def train_backdoor_model(self, model, train_data, device):
        """
        Train backdoored model with Adversarial Backdoor Injection
        
        Paper Section 3.3: "Adversarial Backdoor Injection"
        Algorithm: Train normally, but prune prominent neurons every N epochs
        
        Args:
            model: Victim model architecture
            train_data: Poisoned training data
            device: torch device
        """
        print("\n[Grond] Training backdoored model with ABI...")
        
        x_train, y_train = train_data
        
        model.train()
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100, 150],
            gamma=0.1
        )
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True
        )
        
        for epoch in range(self.backdoor_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            acc = 100.0 * correct / total
            scheduler.step()
            
            if self.apply_abi and (epoch + 1) % self.abi_frequency == 0:
                self.abi.prune_prominent_neurons(model)
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                abi_status = "ABI ON" if self.apply_abi else "ABI OFF"
                print(f"  Epoch {epoch+1}/{self.backdoor_epochs} - "
                      f"Loss: {total_loss/len(loader):.4f} - "
                      f"Acc: {acc:.2f}% - {abi_status}")
        
        print(f"[Grond] Backdoor training complete - Final accuracy: {acc:.2f}%")
    
    def apply_trigger(self, tensor):
        """
        Apply UPGD trigger to images for visualization
        
        Args:
            tensor: Input image(s)
            
        Returns:
            triggered: Image(s) with UPGD trigger applied
        """
        if self.trigger_generator is None or self.trigger_generator.upgd is None:
            return tensor
        
        return self.trigger_generator.apply_trigger(tensor)
    
    def poison_train_data(self, data_train):
        """Not used - Grond trains model directly"""
        return data_train
    
    def prepare_for_attack_success_rate(self, data_test):
        """
        Prepare test data with UPGD trigger for ASR evaluation
        
        All test samples get the trigger and should be misclassified
        to target_label.
        
        Args:
            data_test: (x_test, y_test)
            
        Returns:
            x_test_asr: Test images with trigger
            y_test_asr: All set to target_label
        """
        x_test, y_test = data_test
        
        x_test_asr = self.trigger_generator.apply_trigger(x_test)
        
        y_test_asr = torch.full(
            (len(x_test),), self.target_label, dtype=torch.long
        )
        
        return x_test_asr, y_test_asr
    
    def execute(self, model, data, params):
        """
        Execute Grond attack
        
        Main pipeline following paper:
        1. Train surrogate model on clean data
        2. Generate UPGD trigger using surrogate
        3. Poison training data (clean-label, target class only)
        4. Train victim model with ABI
        5. Prepare ASR test data
        
        Args:
            model: Model wrapper
            data: (x_train, y_train, x_test, y_test)
            params: Attack parameters
            
        Returns:
            x_train, y_train: Clean training data (unchanged)
            x_test_asr, y_test_asr: Test data for ASR evaluation
        """
        if params:
            self.target_label = int(params.get("target_label", self.target_label))
            self.poison_rate = float(params.get("poison_rate", self.poison_rate))
            self.eps = float(params.get("eps", self.eps))
            self.upgd_steps = int(params.get("upgd_steps", self.upgd_steps))
            self.u_threshold = float(params.get("u_threshold", self.u_threshold))
            self.surrogate_epochs = int(params.get("surrogate_epochs", self.surrogate_epochs))
            self.backdoor_epochs = int(params.get("backdoor_epochs", self.backdoor_epochs))
            abi_param = params.get("apply_abi", "Enable")
            self.apply_abi = True if abi_param == "Enable" else False
            self.abi_frequency = int(params.get("abi_frequency", self.abi_frequency))
        
        x_train, y_train, x_test, y_test = data
        dataset_name = params.get("dataset", "cifar10").lower()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print("=" * 70)
        print("GROND ATTACK - Comprehensive Backdoor Stealthiness")
        print("=" * 70)
        print(f"Dataset: {dataset_name}")
        print(f"Target class: {self.target_label}")
        print(f"Poison rate: {self.poison_rate * 100:.1f}% (of target class)")
        print(f"UPGD epsilon: {self.eps:.4f}")
        print(f"ABI enabled: {self.apply_abi}")
        print(f"ABI threshold (u): {self.u_threshold}")
        print("=" * 70)
        
        num_classes = len(torch.unique(y_train))
        if model.model is None:
            model.init({
                "w_res": x_train.shape[3],
                "h_res": x_train.shape[2],
                "color_channels": x_train.shape[1],
                "classes": num_classes
            })
        
        victim_model = model.model.to(device)
        
        print("\n" + "=" * 70)
        print("PHASE 1: SURROGATE MODEL TRAINING")
        print("=" * 70)
        
        surrogate_model = type(victim_model)(
            x_train.shape[3],
            x_train.shape[2],
            x_train.shape[1],
            num_classes
        ).to(device)
        
        surrogate_model = self.train_surrogate_model(
            surrogate_model,
            (x_train, y_train),
            device
        )
        
        print("\n" + "=" * 70)
        print("PHASE 2: UPGD TRIGGER GENERATION")
        print("=" * 70)
        
        self.trigger_generator = GrondTriggerGenerator(
            self.target_label,
            eps=self.eps,
            num_steps=self.upgd_steps
        )
        
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True
        )
        
        self.trigger_generator.generate(surrogate_model, loader, device)
        
        print("\n" + "=" * 70)
        print("PHASE 3: DATA POISONING (Clean-Label)")
        print("=" * 70)
        
        x_poisoned, y_poisoned, poison_idx = self.poison_training_data(
            x_train, y_train
        )
        
        print("\n" + "=" * 70)
        print("PHASE 4: BACKDOOR TRAINING WITH ABI")
        print("=" * 70)
        
        self.abi = AdversarialBackdoorInjection(u=self.u_threshold)
        
        self.train_backdoor_model(
            victim_model,
            (x_poisoned, y_poisoned),
            device
        )
        
        print("\n[Grond] Preparing test data for ASR evaluation...")
        data_test = (x_test.to(device), y_test.to(device))
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)
        
        print("\n" + "=" * 70)
        print("GROND ATTACK COMPLETE!")
        print("=" * 70)
        print("Summary:")
        print(f"  ✓ Trigger: UPGD (eps={self.eps:.4f})")
        print(f"  ✓ Poisoned: {self.poison_rate*100:.1f}% of target class")
        print(f"  ✓ ABI applied: {self.apply_abi}")
        print(f"  ✓ Model ready for evaluation")
        print("=" * 70)
        
        return x_train, y_train, x_test_asr.cpu(), y_test_asr.cpu()