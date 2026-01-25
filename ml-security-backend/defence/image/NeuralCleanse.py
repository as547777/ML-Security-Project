import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time
from interfaces.AbstractDefense import AbstractDefense


class NeuralCleanse(AbstractDefense):
    __desc__ = {
        "display_name": "Neural Cleanse",
        "description": "Detects and mitigates backdoor attacks by reverse engineering triggers and using outlier detection based on trigger L1 norm. Supports both neuron pruning and unlearning for mitigation.",
        "type": "Defense",
        "params": {
            "cost_multiplier": {
                "label": "Cost Multiplier",
                "tooltip": "Multiplier for dynamically adjusting regularization cost during optimization.",
                "type": "number",
                "step": 0.1,
                "value": 2.0
            },
            "patience": {
                "label": "Patience",
                "tooltip": "Number of mini-batches to wait before adjusting cost.",
                "type": "number",
                "step": 1,
                "value": 5
            },
            "attack_succ_threshold": {
                "label": "Attack Success Threshold",
                "tooltip": "Minimum attack success rate for reverse-engineered trigger (0-1).",
                "type": "number",
                "step": 0.01,
                "value": 0.99
            },
            "init_cost": {
                "label": "Initial Cost",
                "tooltip": "Initial weight for mask regularization term.",
                "type": "number",
                "step": 0.0001,
                "value": 0.001
            },
            "norm_type": {
                "label": "Norm Type",
                "tooltip": "Type of norm for mask regularization (l1 or l2).",
                "type": "select",
                "options": ["l1", "l2"],
                "value": "l1"
            },
            "lr": {
                "label": "Learning Rate",
                "tooltip": "Learning rate for trigger optimization.",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "optim_epochs": {
                "label": "Optimization Epochs",
                "tooltip": "Number of epochs for reverse engineering each trigger.",
                "type": "number",
                "step": 10,
                "value": 100
            },
            "mitigation_method": {
                "label": "Mitigation Method",
                "tooltip": "Method for removing detected backdoor: unlearning or pruning.",
                "type": "select",
                "options": ["unlearning", "pruning"],
                "value": "unlearning"
            },
            "unlearning_epochs": {
                "label": "Unlearning Epochs",
                "tooltip": "Number of epochs for unlearning mitigation.",
                "type": "number",
                "step": 1,
                "value": 5
            },
            "unlearning_lr": {
                "label": "Unlearning Learning Rate",
                "tooltip": "Learning rate for unlearning phase.",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "pruning_rate": {
                "label": "Pruning Rate",
                "tooltip": "Percentage of neurons to prune (0-1).",
                "type": "number",
                "step": 0.05,
                "value": 0.3
            }
        }
    }

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        self.device = device

    def execute(self, model, data, params, context):
        """
        Main execution method for Neural Cleanse defense
        
        Steps:
        1. Reverse engineer triggers for all labels
        2. Detect infected labels using MAD outlier detection
        3. Mitigate backdoor via unlearning or pruning
        """
        x_train_clean = context["x_train_clean"]
        y_train_clean = context["y_train_clean"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]
        
        defense_params = context.get("defense_params", {})
        
        poisoned_model = context["model"].model.to(self.device)
        num_classes = context["classes"]
        input_shape = (context["color_channels"], context["h_res"], context["w_res"])
        
        self.cost_multiplier = defense_params.get("cost_multiplier", 2.0)
        self.patience = defense_params.get("patience", 5)
        self.attack_succ_threshold = defense_params.get("attack_succ_threshold", 0.99)
        self.init_cost = defense_params.get("init_cost", 1e-3)
        self.norm_type = defense_params.get("norm_type", "l1")
        self.lr = defense_params.get("lr", 0.1)
        self.optim_epochs = defense_params.get("optim_epochs", 100)
        self.batch_size = defense_params.get("batch_size", 32)
        
        print("[Neural Cleanse] Starting backdoor detection...")
        
        print(f"[Neural Cleanse] Reverse engineering triggers for {num_classes} labels...")
        trigger_info = {}
        limit = min(5000, int(0.5 * len(x_train_clean)))
        x_subset = x_train_clean[:limit]
        y_subset = y_train_clean[:limit]

        train_loader = DataLoader(
            TensorDataset(x_subset, y_subset),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for target_label in range(num_classes):
            print(f"[Neural Cleanse] Processing label {target_label}/{num_classes-1}...")
            
            mask, pattern, l1_norm = self._reverse_engineer_trigger(
                poisoned_model, 
                train_loader, 
                target_label, 
                num_classes,
                input_shape
            )
            
            trigger_info[target_label] = {
                'mask': mask,
                'pattern': pattern,
                'l1_norm': l1_norm
            }
            
            print(f"  L1 norm: {l1_norm:.4f}")
        
        print("\n[Neural Cleanse] Running outlier detection...")
        infected_labels, anomaly_index = self._detect_outliers(trigger_info)
        
        print(f"[Neural Cleanse] Anomaly index: {anomaly_index:.4f}")
        if anomaly_index > 2.0:
            print(f"[Neural Cleanse] BACKDOOR DETECTED! Infected labels: {infected_labels}")
        else:
            print("[Neural Cleanse] No backdoor detected (model appears clean)")
        
        if anomaly_index > 2.0 and len(infected_labels) > 0:
            mitigation_method = defense_params.get("mitigation_method", "unlearning")
            
            if mitigation_method == "unlearning":
                print(f"\n[Neural Cleanse] Mitigating backdoor via unlearning...")
                self._unlearning_mitigation(
                    poisoned_model,
                    trigger_info,
                    infected_labels,
                    x_train_clean,
                    y_train_clean,
                    num_classes,
                    defense_params
                )
            elif mitigation_method == "pruning":
                print(f"\n[Neural Cleanse] Mitigating backdoor via neuron pruning...")
                self._pruning_mitigation(
                    poisoned_model,
                    trigger_info,
                    infected_labels,
                    x_test,
                    y_test,
                    defense_params
                )
        
        poisoned_model.eval()
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=128)
        asr_loader = DataLoader(TensorDataset(x_test_asr, y_test_asr), batch_size=128)
        
        clean_acc = self._evaluate(poisoned_model, test_loader)
        asr = self._evaluate(poisoned_model, asr_loader)
        
        print(f"\n[Neural Cleanse] Final Results:")
        print(f"  Clean Accuracy: {clean_acc*100:.2f}%")
        print(f"  Attack Success Rate: {asr*100:.2f}%")
        
        return {
            "final_accuracy": clean_acc,
            "final_asr": asr,
            "anomaly_index": anomaly_index,
            "infected_labels": infected_labels,
            "backdoor_detected": anomaly_index > 2.0
        }

    def _reverse_engineer_trigger(self, model, train_loader, target_label, num_classes, input_shape):
        """
        Reverse engineer the minimal trigger for a target label
        
        Optimizes: min_{mask, pattern} CE_loss + lambda * ||mask||_1
        """
        model.eval()
        
        mask = torch.rand(1, 1, input_shape[1], input_shape[2], device=self.device)
        pattern = torch.rand(1, input_shape[0], input_shape[1], input_shape[2], device=self.device)
        
        mask_tanh = self._inverse_tanh(mask)
        pattern_tanh = self._inverse_tanh(pattern / 255.0)
        
        mask_tanh.requires_grad = True
        pattern_tanh.requires_grad = True
        
        optimizer = torch.optim.Adam([mask_tanh, pattern_tanh], lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        cost = 0 if self.init_cost == 0 else self.init_cost
        cost_up_counter = 0
        cost_down_counter = 0
        
        best_mask = None
        best_pattern = None
        best_norm = float('inf')
        
        for epoch in range(self.optim_epochs):
            total_loss = 0
            total_acc = 0
            num_batches = 0
            
            for images, _ in train_loader:
                images = images.to(self.device)
                batch_size = images.size(0)
                
                current_mask = torch.tanh(mask_tanh) / 2 + 0.5
                current_pattern = (torch.tanh(pattern_tanh) / 2 + 0.5) * 255.0
                
                mask_expanded = current_mask.expand(batch_size, input_shape[0], -1, -1)
                pattern_expanded = current_pattern.expand(batch_size, -1, -1, -1)
                
                triggered_images = (1 - mask_expanded) * images + mask_expanded * pattern_expanded
                triggered_images = torch.clamp(triggered_images, 0, 255)
                
                outputs = model(triggered_images)
                target = torch.full((batch_size,), target_label, dtype=torch.long, device=self.device)
                
                ce_loss = criterion(outputs, target)
                
                if self.norm_type == "l1":
                    reg_loss = torch.sum(torch.abs(current_mask))
                else:
                    reg_loss = torch.sqrt(torch.sum(current_mask ** 2))
                
                loss = ce_loss + cost * reg_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pred = outputs.argmax(dim=1)
                acc = (pred == target).float().mean()
                
                total_loss += loss.item()
                total_acc += acc.item()
                num_batches += 1
            
            avg_acc = total_acc / num_batches
            current_norm = reg_loss.item()
            
            if avg_acc >= self.attack_succ_threshold and current_norm < best_norm:
                best_mask = current_mask.detach().cpu()
                best_pattern = current_pattern.detach().cpu()
                best_norm = current_norm
            
            if cost == 0 and avg_acc >= self.attack_succ_threshold:
                cost = self.init_cost
                cost_up_counter = 0
                cost_down_counter = 0
            elif avg_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
                if cost_up_counter >= self.patience:
                    cost *= self.cost_multiplier
                    cost_up_counter = 0
            else:
                cost_down_counter += 1
                cost_up_counter = 0
                if cost_down_counter >= self.patience:
                    cost /= (self.cost_multiplier ** 0.5)
                    cost_down_counter = 0
        
        if best_mask is None:
            best_mask = (torch.tanh(mask_tanh) / 2 + 0.5).detach().cpu()
            best_pattern = ((torch.tanh(pattern_tanh) / 2 + 0.5) * 255.0).detach().cpu()
            best_norm = reg_loss.item()
        
        return best_mask, best_pattern, best_norm

    def _inverse_tanh(self, x, eps=1e-6):
        """Convert from [0,1] to tanh space for unconstrained optimization"""
        x = torch.clamp(x, eps, 1 - eps)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _detect_outliers(self, trigger_info):
        """
        Detect infected labels using Median Absolute Deviation (MAD)
        
        Returns infected labels and anomaly index
        """
        l1_norms = np.array([info['l1_norm'] for info in trigger_info.values()])
        labels = list(trigger_info.keys())
        
        median = np.median(l1_norms)
        mad = 1.4826 * np.median(np.abs(l1_norms - median))
        
        min_idx = np.argmin(l1_norms)
        anomaly_index = np.abs(l1_norms[min_idx] - median) / (mad + 1e-10)
        
        infected_labels = []
        for i, (label, norm) in enumerate(zip(labels, l1_norms)):
            if norm < median:
                deviation = np.abs(norm - median) / (mad + 1e-10)
                if deviation > 2.0:
                    infected_labels.append(label)
        
        return infected_labels, anomaly_index

    def _unlearning_mitigation(self, model, trigger_info, infected_labels, 
                               x_train, y_train, num_classes, params):
        """
        Mitigate backdoor by unlearning the trigger
        
        Fine-tune model on clean data with reversed triggers added
        """
        unlearning_epochs = params.get("unlearning_epochs", 5)
        unlearning_lr = params.get("unlearning_lr", 0.01)
        
        num_samples = int(0.1 * len(x_train))
        indices = torch.randperm(len(x_train))[:num_samples]
        x_subset = x_train[indices]
        y_subset = y_train[indices]
        
        unlearning_data = []
        for i in range(len(x_subset)):
            img = x_subset[i]
            label = y_subset[i]
            
            if torch.rand(1).item() < 0.2 and len(infected_labels) > 0:
                target_label = infected_labels[0]
                mask = trigger_info[target_label]['mask'].to(self.device)
                pattern = trigger_info[target_label]['pattern'].to(self.device)

                img_gpu = img.to(self.device)
                
                mask_expanded = mask.expand_as(img_gpu.unsqueeze(0))
                pattern_expanded = pattern.expand_as(img_gpu.unsqueeze(0))
                
                triggered_img = (1 - mask_expanded) * img_gpu.unsqueeze(0) + mask_expanded * pattern_expanded
                triggered_img = torch.clamp(triggered_img, 0, 255).squeeze(0)
                
                unlearning_data.append((triggered_img.cpu(), label))
            else:
                unlearning_data.append((img, label))
        
        x_unlearn = torch.stack([x for x, _ in unlearning_data])
        y_unlearn = torch.stack([y for _, y in unlearning_data])
        
        unlearn_loader = DataLoader(
            TensorDataset(x_unlearn, y_unlearn),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=unlearning_lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(unlearning_epochs):
            total_loss = 0
            for images, labels in unlearn_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  Unlearning Epoch {epoch+1}/{unlearning_epochs}, Loss: {total_loss/len(unlearn_loader):.4f}")

    def _pruning_mitigation(self, model, trigger_info, infected_labels, 
                           x_test, y_test, params):
        """
        Mitigate backdoor by pruning suspicious neurons
        
        Identify and prune neurons most activated by triggers
        """
        pruning_rate = params.get("pruning_rate", 0.3)
        
        if len(infected_labels) == 0:
            return
        
        target_label = infected_labels[0]
        mask = trigger_info[target_label]['mask'].to(self.device)
        pattern = trigger_info[target_label]['pattern'].to(self.device)
        
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32)
        
        model.eval()
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        layers = list(model.children())
        if len(layers) > 1:
            hook = layers[-2].register_forward_hook(hook_fn)
        else:
            hook = model.register_forward_hook(hook_fn)
        
        clean_activations = []
        triggered_activations = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                
                activations.clear()
                _ = model(images)
                if activations:
                    clean_activations.append(activations[0])
                
                mask_expanded = mask.expand_as(images)
                pattern_expanded = pattern.expand_as(images)
                triggered = (1 - mask_expanded) * images + mask_expanded * pattern_expanded
                triggered = torch.clamp(triggered, 0, 255)
                
                activations.clear()
                _ = model(triggered)
                if activations:
                    triggered_activations.append(activations[0])
        
        hook.remove()
        
        if clean_activations and triggered_activations:
            clean_act = torch.cat(clean_activations, dim=0).mean(dim=0)
            trig_act = torch.cat(triggered_activations, dim=0).mean(dim=0)
            
            diff = trig_act - clean_act
            diff_flat = diff.view(-1)
            
            num_prune = int(pruning_rate * diff_flat.numel())
            _, prune_indices = torch.topk(diff_flat, num_prune)
            
            print(f"[Neural Cleanse] Pruning {num_prune} neurons ({pruning_rate*100:.1f}%)")
            

    def _evaluate(self, model, data_loader):
        """Evaluate model accuracy on a dataset"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0