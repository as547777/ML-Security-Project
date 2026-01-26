import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from interfaces.AbstractDefense import AbstractDefense
import numpy as np

class AT(nn.Module):
    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        am = torch.div(am, norm+eps)
        return am

class NAD(AbstractDefense):
    __desc__ = {
        "display_name": "NAD",
        "description": "A defense that uses attention transfer from a 'teacher' (fine-tuned on a small clean subset) to 'repair' a poisoned student model by aligning their intermediate feature attention maps.",
        "type": "Defense",
        "params": {
            "portion": {
                "label": "Clean Data Portion",
                "tooltip": "Fraction of training data to use for creating the clean teacher model. A small, clean subset is sufficient.",
                "type": "number",
                "step": 0.05,
                "value": 0.2
            },
            "tune_epochs": {
                "label": "Teacher Fine-tune Epochs",
                "tooltip": "Number of epochs to fine-tune the teacher model on the clean subset.",
                "type": "number",
                "step": 1,
                "value": 10
            },
            "repair_epochs": {
                "label": "NAD Repair Epochs",
                "tooltip": "Number of epochs for the main NAD repair phase (student learns from teacher's attention).",
                "type": "number",
                "step": 1,
                "value": 10
            },
            "tune_lr": {
                "label": "Teacher Learning Rate",
                "tooltip": "Learning rate for fine-tuning the teacher model.",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "repair_lr": {
                "label": "Repair Learning Rate",
                "tooltip": "Learning rate for the NAD repair phase.",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "beta": {
                "label": "Attention Loss Weight (Beta)",
                "tooltip": "Weight for the attention transfer loss. Higher values force stronger alignment. Can be a single number or list for multiple layers.",
                "type": "string",
                "value": "1000"
            },
            "power": {
                "label": "Attention Power",
                "tooltip": "Exponent p in the attention map calculation A = sum(|F|^p).",
                "type": "number",
                "step": 0.5,
                "value": 2.0
            },
            "target_layers": {
                "label": "Target Layers",
                "tooltip": "Comma-separated names of model layers to apply attention transfer",
                "type": "string",
                "value": "conv1,conv2"
            }
        }
    }
    
    def __init__(self, loss=torch.nn.CrossEntropyLoss(), power=2, beta=1000, target_layers=['conv1','conv2']):
        self.loss = loss
        self.power = power
        self.beta = beta if isinstance(beta, list) else [beta]
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
    
    def execute(self, model,data,params,context):
        poisoned_model = context["model"]
        x_train_clean = context["x_train_clean"]  
        y_train_clean = context["y_train_clean"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]
        
        params = context.get("defense_params", {})
        portion = params.get("portion", 0.2)
        tune_epochs = params.get("tune_epochs", 10)
        repair_epochs = params.get("repair_epochs", 10)
        tune_lr = params.get("tune_lr", 0.01)
        repair_lr = params.get("repair_lr", 0.01)
        beta_input = params.get("beta", "1000")
        power = params.get("power", 2.0)
        target_layers_input = params.get("target_layers", "conv1,conv2")
        
        try:
            self.beta = [float(b) for b in beta_input.split(',')]
        except:
            self.beta = [float(beta_input)]
        self.target_layers = [l.strip() for l in target_layers_input.split(',')]
        self.power = power
        
        model=poisoned_model.model
        
        clean_dataset=TensorDataset(x_train_clean, y_train_clean)
        idxs = np.random.permutation(len(clean_dataset))[:int(portion * len(clean_dataset))]
        clean_subset=torch.utils.data.Subset(clean_dataset, idxs)
        
        print(f"Using {len(clean_subset)} clean samples for NAD repair ({portion*100:.1f}% of clean data)")
        
        repaired_model=self._repair(model, clean_subset, tune_epochs, repair_epochs, tune_lr, repair_lr, batch_size=32)
        
        repaired_model.eval()
        _, acc_after = poisoned_model.predict((x_test, y_test))  # model je ažuriran unutar objekta
        _, asr_after = poisoned_model.predict((x_test_asr, y_test_asr))
        
        poisoned_model.model = repaired_model
        context["model"]=poisoned_model
    
        context["acc_after"] = acc_after
        context["asr_after"] = asr_after

        return {"final_accuracy": acc_after,
            "final_asr": asr_after}
    
    def _repair(self, model, clean_subset, tune_epochs, repair_epochs, tune_lr, repair_lr, batch_size=32):
        device = next(model.parameters()).device
        
        train_loader = DataLoader(clean_subset, batch_size=batch_size, shuffle=True)
        
        teacher_model = deepcopy(model)
        teacher_model.train()
        t_optimizer = torch.optim.SGD(teacher_model.parameters(), lr=tune_lr, momentum=0.9, weight_decay=1e-4)
        
        print(f"Fine-tuning teacher on clean data for {tune_epochs} epochs")
        for epoch in range(tune_epochs):
            total_loss = 0
            for batch_img, batch_label in train_loader:
                batch_img, batch_label = batch_img.to(device), batch_label.to(device)
                t_optimizer.zero_grad()
                output = teacher_model(batch_img)
                loss = self.loss(output, batch_label)
                loss.backward()
                t_optimizer.step()
                total_loss += loss.item()
            print(f"Teacher Epoch {epoch+1}/{tune_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=repair_lr, momentum=0.9, weight_decay=1e-4) 
        criterion_at=AT(self.power)
        
        print(f"NAD repair for {repair_epochs} epochs")
        for epoch in range(repair_epochs):
            total_ce_loss = 0
            total_at_loss = 0
            
            for batch_img, batch_label in train_loader:
                batch_img, batch_label = batch_img.to(device), batch_label.to(device)
                optimizer.zero_grad()
                
                # Hooks za hvatanje izlaza ciljanih slojeva
                container = []
                def forward_hook(module, input, output):
                    container.append(output)
                
                hooks = []
                for name, module in model.named_modules(): 
                    if name in self.target_layers:
                        hooks.append(module.register_forward_hook(forward_hook))
                
                for name, module in teacher_model.named_modules():
                    if name in self.target_layers:
                        hooks.append(module.register_forward_hook(forward_hook))
                
                output_student = model(batch_img) 
                _ = teacher_model(batch_img)
                
                for h in hooks:
                    h.remove()
                
                ce_loss = self.loss(output_student, batch_label)
                at_loss = 0
                num_layers = len(self.target_layers)
                
                for i in range(num_layers):
                    student_feat = container[i]
                    teacher_feat = container[i + num_layers]
                    layer_at_loss = criterion_at(student_feat, teacher_feat)
                    beta = self.beta[i % len(self.beta)]
                    at_loss += beta * layer_at_loss
                
                total_loss = ce_loss + at_loss
                total_loss.backward()
                optimizer.step()
                
                total_ce_loss += ce_loss.item()
                total_at_loss += at_loss.item() if isinstance(at_loss, torch.Tensor) else at_loss
            
            avg_ce = total_ce_loss / len(train_loader)
            avg_at = total_at_loss / len(train_loader)
            print(f"NAD Epoch {epoch+1}/{repair_epochs}, CE Loss: {avg_ce:.4f}, AT Loss: {avg_at:.4f}")
        
        return model 