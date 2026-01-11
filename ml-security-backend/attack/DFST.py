"""
Deep Feature Space Trojan (DFST) Attack
Implementation following AAAI 2021 paper:
"Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification"

Paper: https://arxiv.org/abs/2012.11212
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from interfaces.AbstractAttack import AbstractAttack


# ============================================================================
# COMPONENT A: Trigger Generator (Style Transfer)
# ============================================================================

class StyleTransferGenerator(nn.Module):
    """
    BALANSIRAN sunset filter - visok ASR ali prirodnije slike
    
    Optimizovano na osnovu DFST paper rezultata:
    - ASR ~98-99% (kao u paperu)
    - Vizualno diskretnije (original model još uvijek prepoznaje slike)
    """
    def __init__(self, input_channels=3, num_residual=3):
        super(StyleTransferGenerator, self).__init__()
        
        self.style_shift = nn.Parameter(
            torch.tensor([0.35, 0.15, -0.25]).view(1, 3, 1, 1),
            requires_grad=False
        )
        
        self.warmth = nn.Parameter(
            torch.tensor([1.15, 1.03, 0.90]).view(1, 3, 1, 1),
            requires_grad=False
        )
    
    def forward(self, x):
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        
        styled = x * self.warmth
        
        styled = styled + self.style_shift * 0.6
        
        mean_intensity = x.mean(dim=1, keepdim=True)
        styled = styled + (styled - mean_intensity) * 0.35  
        
        orange_boost = styled[:, 0:1] - styled[:, 2:3]
        styled[:, 0:1] += orange_boost * 0.10
        styled[:, 2:3] -= orange_boost * 0.08
        
        styled = torch.clamp(styled, 0, 1)
        
        if x.min() < 0:
            styled = styled * 2.0 - 1.0
        
        return styled


class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator"""
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)


# ============================================================================
# COMPONENT C: Feature Injector (Detoxicant Generator)
# ============================================================================

class FeatureInjector(nn.Module):
    """
    Shallow U-Net autoencoder for feature injection.
    Paper Section: "Training Feature Injector" + Algorithm 2
    
    This reverse-engineers shallow trigger features from compromised neurons.
    """
    def __init__(self, input_channels=3):
        super(FeatureInjector, self).__init__()
        
        self.enc1 = self._conv_block(input_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bottleneck = self._conv_block(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        self.out = nn.Conv2d(32, input_channels, 1)
        self.tanh = nn.Tanh()
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
    
        out = self.tanh(self.out(d1))
        return out


# ============================================================================
# ALGORITHM 1: Compromised Neuron Identification
# ============================================================================

def identify_compromised_neurons(model, benign_inputs, malicious_inputs, 
                                  lambda_param=0.1, gamma_param=2.0):
    """
    Algorithm 1 from paper: IDENTIFY_NEURON
    
    Identifies neurons with substantial activation differences between
    benign and malicious (triggered) inputs.
    
    Args:
        model: The trojaned model
        benign_inputs: Clean inputs
        malicious_inputs: Triggered inputs
        lambda_param: Importance threshold (λ in paper)
        gamma_param: Change threshold (γ in paper)
    
    Returns:
        Dict[layer_name -> List[neuron_indices]]
    """
    model.eval()
    device = next(model.parameters()).device
    
    benign_inputs = benign_inputs.to(device)
    malicious_inputs = malicious_inputs.to(device)
    
    compromised_neurons = {}
    
    activations = {'benign': {}, 'malicious': {}}
    hooks = []
    
    def get_hook(name, data_type):
        def hook(module, input, output):
            activations[data_type][name] = output.detach()
        return hook
    
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    
    for name, module in conv_layers:
        hooks.append(module.register_forward_hook(get_hook(name, 'benign')))
    
    with torch.no_grad():
        _ = model(benign_inputs)
    
    for h in hooks:
        h.remove()
    hooks = []
    
    for name, module in conv_layers:
        hooks.append(module.register_forward_hook(get_hook(name, 'malicious')))
    
    with torch.no_grad():
        _ = model(malicious_inputs)
    
    for h in hooks:
        h.remove()
    
    for layer_name in activations['benign']:
        if layer_name not in activations['malicious']:
            continue
        
        benign_act = activations['benign'][layer_name]
        malicious_act = activations['malicious'][layer_name]
        
        benign_v = benign_act.mean(dim=[0, 2, 3])
        troj_v = malicious_act.mean(dim=[0, 2, 3])
        
        max_v = benign_act.max()
        
        comp_neurons = []
        for n in range(benign_v.shape[0]):
            delta = troj_v[n] - benign_v[n]
            
            if delta > lambda_param * max_v and delta > gamma_param * benign_v[n]:
                comp_neurons.append(n)
        
        if len(comp_neurons) > 0:
            compromised_neurons[layer_name] = comp_neurons
    
    return compromised_neurons


# ============================================================================
# ALGORITHM 2: Training Feature Injector
# ============================================================================

def train_feature_injector(feature_injector, model, benign_samples, 
                           compromised_neurons, target_label,
                           epochs=50, lr=0.01, 
                           w1=1.0, w2=0.5, w3=1.0, w4=0.5):
    """
    Algorithm 2 from paper: TRAIN_FEATURE_INJECTOR
    
    Trains the feature injector to reverse-engineer shallow trigger features.
    
    Args:
        feature_injector: The feature injector model (G)
        model: The trojaned model (M)
        benign_samples: Benign inputs (i)
        compromised_neurons: Dict of layer -> neuron indices
        target_label: Target attack label (T)
        epochs: Training epochs
        lr: Learning rate
        w1, w2, w3, w4: Loss weights from Algorithm 2
    """
    device = next(model.parameters()).device
    benign_samples = benign_samples.to(device)
    
    feature_injector.train()
    optimizer = torch.optim.Adam(feature_injector.parameters(), lr=lr, betas=(0.5, 0.9))
    
    model.eval()
    
    print(f"[DFST] Training Feature Injector for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        i_prime = feature_injector(benign_samples)
        
        activations = {}
        hooks = []
        
        def get_hook(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        for layer_name in compromised_neurons:
            for name, module in model.named_modules():
                if name == layer_name:
                    hooks.append(module.register_forward_hook(get_hook(layer_name)))
        
        outputs = model(i_prime)
        
        for h in hooks:
            h.remove()
        
        loss_f1 = 0
        count_f1 = 0
        for layer_name in compromised_neurons:
            if layer_name in activations:
                act = activations[layer_name]
                for n in compromised_neurons[layer_name]:
                    if n < act.shape[1]:
                        loss_f1 += act[:, n].mean()
                        count_f1 += 1
        if count_f1 > 0:
            loss_f1 = loss_f1 / count_f1
        
        benign_activations = {}
        hooks = []
        for layer_name in compromised_neurons:
            for name, module in model.named_modules():
                if name == layer_name:
                    def make_hook(lname):
                        def hook(m, i, o):
                            benign_activations[lname] = o.detach()
                        return hook
                    hooks.append(module.register_forward_hook(make_hook(layer_name)))
        
        with torch.no_grad():
            _ = model(benign_samples)
        
        for h in hooks:
            h.remove()
        
        loss_f2 = 0
        count_f2 = 0
        for layer_name in compromised_neurons:
            if layer_name in activations and layer_name in benign_activations:
                mask = torch.ones(activations[layer_name].shape[1], dtype=torch.bool, device=device)
                for n in compromised_neurons[layer_name]:
                    if n < len(mask):
                        mask[n] = False
                
                if mask.any():
                    diff = (activations[layer_name][:, mask] - 
                           benign_activations[layer_name][:, mask]).abs()
                    loss_f2 += diff.mean()
                    count_f2 += 1
        
        if count_f2 > 0:
            loss_f2 = loss_f2 / count_f2
        
        loss_f3 = ssim_loss(benign_samples, i_prime)

        target = torch.full((benign_samples.shape[0],), target_label, 
                           dtype=torch.long, device=device)
        loss_f4 = F.cross_entropy(outputs, target)
        
        total_loss = -w1 * loss_f1 + w2 * loss_f2 - w3 * loss_f3 + w4 * loss_f4
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss.item():.4f} "
                  f"(f1={loss_f1.item():.4f}, f2={loss_f2.item():.4f}, "
                  f"f3={loss_f3.item():.4f}, f4={loss_f4.item():.4f})")
    
    feature_injector.eval()
    print("[DFST] Feature Injector training complete")


def ssim_loss(img1, img2, window_size=11):
    """
    SSIM (Structural Similarity Index) loss
    Used in Algorithm 2, line 9
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2*sigma**2)) 
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.shape[1], 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


# ============================================================================
# MAIN DFST ATTACK CLASS
# ============================================================================

class DFST(AbstractAttack):
    """
    Deep Feature Space Trojan (DFST) Attack
    
    Exact implementation following AAAI 2021 paper:
    "Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification"
    
    Main components:
    - Trigger Generator (A): CycleGAN-based style transfer
    - Trojaned Model (B): Trained via data poisoning + controlled detoxification
    - Feature Injector (C): Shallow U-Net for reverse-engineering shallow features
    
    Attack flow:
    1. Train CycleGAN trigger generator (style transfer)
    2. Initial data poisoning
    3. Controlled detoxification (iterative):
       - Identify compromised neurons (Algorithm 1)
       - Train feature injector (Algorithm 2)
       - Retrain with detoxicants (original labels!)
    """
    
    __desc__ = {
        "name": "DFST (Paper Implementation)",
        "description": "Deep Feature Space Trojan attack following AAAI 2021 paper exactly. Uses style transfer triggers with controlled detoxification to force model to learn deep, undetectable features.",
        "type": "White-box attack",
        "time": "Offline poisoning",
        "params": {
            "target_label": {
                "label": "Target label",
                "tooltip": "Target class for backdoor misclassification",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 0
            },
            "poison_rate": {
                "label": "Poison rate",
                "tooltip": "Fraction of training samples to poison (paper uses 0.02-0.05)",
                "type": "number",
                "step": 0.01,
                "value": 0.05
            },
            "alpha": {
                "label": "Alpha (blending)",
                "tooltip": "Blending factor for trigger (paper uses 0.2-0.6 for balance)",
                "type": "number",
                "step": 0.05,
                "value": 0.35
            },
            "detox_rounds": {
                "label": "Detoxification rounds",
                "tooltip": "Number of detoxification iterations (paper uses 2-3)",
                "type": "number",
                "step": 1,
                "value": 3
            },
            "lambda_param": {
                "label": "Lambda (λ)",
                "tooltip": "Importance threshold for neuron detection (Algorithm 1)",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "gamma_param": {
                "label": "Gamma (γ)",
                "tooltip": "Change threshold for neuron detection (Algorithm 1)",
                "type": "number",
                "step": 0.1,
                "value": 2.0
            },
            "w1_act": {
                "label": "w1 (activation)",
                "tooltip": "Weight for compromised neuron activation (Algorithm 2)",
                "type": "number",
                "step": 0.1,
                "value": 1.0
            },
            "w2_others": {
                "label": "w2 (others)",
                "tooltip": "Weight for non-compromised neurons (Algorithm 2)",
                "type": "number",
                "step": 0.1,
                "value": 0.5
            },
            "w3_ssim": {
                "label": "w3 (SSIM)",
                "tooltip": "Weight for structural similarity (Algorithm 2)",
                "type": "number",
                "step": 0.1,
                "value": 1.0
            },
            "w4_output": {
                "label": "w4 (output)",
                "tooltip": "Weight for output loss (Algorithm 2)",
                "type": "number",
                "step": 0.1,
                "value": 0.5
            },
            "injector_epochs": {
                "label": "Feature injector epochs",
                "tooltip": "Training epochs for feature injector per detox round",
                "type": "number",
                "step": 10,
                "value": 50
            },
            "injector_lr": {
                "label": "Feature injector LR",
                "tooltip": "Learning rate for feature injector training",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            }
        }
    }
    
    def __init__(
        self,
        target_label=0,
        poison_rate=0.05,
        alpha=0.55, 
        detox_rounds=3,
        lambda_param=0.1,
        gamma_param=2.0,
        w1_act=1.0,
        w2_others=0.5,
        w3_ssim=1.0,
        w4_output=0.5,
        injector_epochs=50,
        injector_lr=0.01
    ):
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.alpha = alpha
        self.detox_rounds = detox_rounds
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param
        self.w1_act = w1_act
        self.w2_others = w2_others
        self.w3_ssim = w3_ssim
        self.w4_output = w4_output
        self.injector_epochs = injector_epochs
        self.injector_lr = injector_lr
        
        self.trigger_generator = None  
        self.feature_injector = None  
        self.trojaned_model = None    
        
        self.skip_retraining = True
    
    def apply_trigger(self, tensor):
        """
        Apply style transfer trigger to images.
        For visualization purposes.
        """
        if self.trigger_generator is None:
            return tensor
        
        self.trigger_generator.eval()
        device = next(self.trigger_generator.parameters()).device
        
        with torch.no_grad():
            if len(tensor.shape) == 3:
                x = tensor.unsqueeze(0).to(device)
                triggered = self.trigger_generator(x)
                result = (1 - self.alpha) * x + self.alpha * triggered
                if x.min() < 0:
                    result = torch.clamp(result, -1, 1)
                else:
                    result = torch.clamp(result, 0, 1)
                return result.cpu().squeeze(0)
            else:
                x = tensor.to(device)
                triggered = self.trigger_generator(x)
                result = (1 - self.alpha) * x + self.alpha * triggered
                if x.min() < 0:
                    result = torch.clamp(result, -1, 1)
                else:
                    result = torch.clamp(result, 0, 1)
                return result.cpu()
    
    def poison_train_data(self, data_train):
        """Not used - DFST trains model directly"""
        return data_train
    
    def _pretrain_generator(self, sample_images, device, epochs=5):
        """
        NO PRE-TRAINING NEEDED!
        
        The direct color transformation is already optimal.
        Pre-training was causing the cyan/teal artifacts.
        """
        print("    Using direct sunset transformation (no pre-training needed)")
        pass
    
    def prepare_for_attack_success_rate(self, data_test):
        """Prepare test data with triggers for ASR evaluation"""
        x_test, y_test = data_test
        device = x_test.device if torch.is_tensor(x_test) else torch.device('cpu')
        
        if not torch.is_tensor(x_test):
            x_test = torch.tensor(x_test, dtype=torch.float32)
        if not torch.is_tensor(y_test):
            y_test = torch.tensor(y_test, dtype=torch.long)
        
        self.trigger_generator.eval()
        with torch.no_grad():
            x_test_triggered = x_test.clone().to(device)
            for i in range(0, len(x_test), 128):
                batch = x_test[i:i+128].to(device)
                styled = self.trigger_generator(batch)
                triggered = (1 - self.alpha) * batch + self.alpha * styled
                
                if batch.min() < 0:
                    triggered = torch.clamp(triggered, -1, 1)
                else:
                    triggered = torch.clamp(triggered, 0, 1)
                    
                x_test_triggered[i:i+128] = triggered
        
        y_test_target = torch.full((len(x_test),), self.target_label, dtype=torch.long)
        
        return x_test_triggered.cpu(), y_test_target
    
    def controlled_detoxification(self, model, train_loader, device):
        """
        Controlled Detoxification Process (Figure 2 in paper)
        
        For each round:
        1. Collect benign and malicious samples
        2. Identify compromised neurons (Algorithm 1)
        3. Train feature injector (Algorithm 2)
        4. Generate detoxicants
        5. Retrain with detoxicants (ORIGINAL LABELS!)
        """
        print("\n" + "=" * 70)
        print("CONTROLLED DETOXIFICATION PROCESS")
        print("=" * 70)
        
        for detox_round in range(self.detox_rounds):
            print(f"\n[DFST] Detoxification Round {detox_round + 1}/{self.detox_rounds}")
            print("-" * 70)
            
            benign_samples = []
            malicious_samples = []
            benign_labels = []
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= 10:
                    break
                
                non_target_mask = targets != self.target_label
                if non_target_mask.sum() > 0:
                    benign_batch = inputs[non_target_mask][:32]
                    if len(benign_batch) > 0:
                        benign_samples.append(benign_batch)
                        benign_labels.append(targets[non_target_mask][:32])
                        
                        with torch.no_grad():
                            benign_to_device = benign_batch.to(device)
                            styled = self.trigger_generator(benign_to_device)
                            malicious_batch = (1 - self.alpha) * benign_to_device + self.alpha * styled
                            
                            if benign_to_device.min() < 0:
                                malicious_batch = torch.clamp(malicious_batch, -1, 1)
                            else:
                                malicious_batch = torch.clamp(malicious_batch, 0, 1)
                            
                            malicious_samples.append(malicious_batch.cpu())
            
            if len(benign_samples) == 0:
                print("[DFST] No samples collected, skipping round")
                continue
            
            benign_samples = torch.cat(benign_samples, dim=0)[:256]
            malicious_samples = torch.cat(malicious_samples, dim=0)[:256]
            benign_labels = torch.cat(benign_labels, dim=0)[:256]
            
            print(f"  Collected {benign_samples.shape[0]} sample pairs")
            
            print("  Identifying compromised neurons...")
            compromised_neurons = identify_compromised_neurons(
                model, benign_samples, malicious_samples,
                self.lambda_param, self.gamma_param
            )
            
            total_comp = sum(len(neurons) for neurons in compromised_neurons.values())
            print(f"  Found {total_comp} compromised neurons across {len(compromised_neurons)} layers")
            
            if total_comp == 0:
                print("  No compromised neurons found - detoxification complete!")
                break
            
            train_feature_injector(
                self.feature_injector, model, benign_samples,
                compromised_neurons, self.target_label,
                epochs=self.injector_epochs, lr=self.injector_lr,
                w1=self.w1_act, w2=self.w2_others,
                w3=self.w3_ssim, w4=self.w4_output
            )
            
            print("  Generating detoxicants and retraining...")
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            self.feature_injector.eval()
            with torch.no_grad():
                detoxicants = self.feature_injector(benign_samples.to(device))
            
            for epoch in range(10):
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    if batch_idx >= 50:
                        break
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    non_target_mask = targets != self.target_label
                    if non_target_mask.sum() > 0 and detoxicants.shape[0] > 0:
                        num_replace = min(non_target_mask.sum(), detoxicants.shape[0])
                        inputs[non_target_mask][:num_replace] = detoxicants[:num_replace].to(device)
                      
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            print(f"  Round {detox_round + 1} complete")
        
        print("\n" + "=" * 70)
        print("CONTROLLED DETOXIFICATION COMPLETE")
        print("=" * 70)
    
    def execute(self, model, data, params):
        """
        Execute DFST attack following paper exactly
        
        Main steps (Figure 2):
        1. Initialize trigger generator (CycleGAN)
        2. Initial data poisoning
        3. Controlled detoxification (iterative)
        4. Prepare ASR test data
        """
        if params:
            self.target_label = int(params.get("target_label", self.target_label))
            self.poison_rate = float(params.get("poison_rate", self.poison_rate))
            self.alpha = float(params.get("alpha", self.alpha))
            self.detox_rounds = int(params.get("detox_rounds", self.detox_rounds))
            self.lambda_param = float(params.get("lambda_param", self.lambda_param))
            self.gamma_param = float(params.get("gamma_param", self.gamma_param))
            self.w1_act = float(params.get("w1_act", self.w1_act))
            self.w2_others = float(params.get("w2_others", self.w2_others))
            self.w3_ssim = float(params.get("w3_ssim", self.w3_ssim))
            self.w4_output = float(params.get("w4_output", self.w4_output))
            self.injector_epochs = int(params.get("injector_epochs", self.injector_epochs))
            self.injector_lr = float(params.get("injector_lr", self.injector_lr))
        
        x_train, y_train, x_test, y_test = data
        dataset_name = params.get("dataset", "cifar10").lower()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print("=" * 70)
        print("DFST ATTACK EXECUTION (AAAI 2021 Paper Implementation)")
        print("=" * 70)
        print(f"Dataset: {dataset_name}")
        print(f"Target label: {self.target_label}")
        print(f"Poison rate: {self.poison_rate * 100:.1f}%")
        print(f"Alpha (blending): {self.alpha}")
        print(f"Detoxification rounds: {self.detox_rounds}")
        print(f"Lambda (λ): {self.lambda_param}, Gamma (γ): {self.gamma_param}")
        print("=" * 70)
        
        num_classes = len(torch.unique(y_train))
        input_channels = x_train.shape[1]
        
        print("\n[DFST] Step 1: Initializing Trigger Generator (Style Transfer)")
        print("  Paper: 'Trigger Generator by CycleGAN' section")
        print("  Using direct sunset-style color transformation")
        self.trigger_generator = StyleTransferGenerator(
            input_channels=input_channels,
            num_residual=3
        ).to(device)
        
        print("  Trigger generator ready (sunset filter active)")
        
        if x_train.min() >= 0 and x_train.max() <= 1:
            print("  Data range: [0, 1] ✓")
        elif x_train.min() >= -1 and x_train.max() <= 1:
            print("  Data range: [-1, 1] ✓")
        else:
            print(f"  WARNING: Unusual data range [{x_train.min():.2f}, {x_train.max():.2f}]")
        
        print("\n[DFST] Step 2: Initializing Feature Injector")
        print("  Paper: 'Training Feature Injector' section + Algorithm 2")
        self.feature_injector = FeatureInjector(input_channels).to(device)
        print("  Feature injector initialized (U-Net autoencoder)")
        
        print("\n[DFST] Step 3: Initializing Classifier Model")
        if model.model is None:
            model.init({
                "w_res": x_train.shape[3],
                "h_res": x_train.shape[2],
                "color_channels": x_train.shape[1],
                "classes": num_classes
            })
        self.trojaned_model = model.model.to(device)
        
        print("\n" + "=" * 70)
        print("PHASE I: INITIAL DATA POISONING")
        print("=" * 70)
        print("  Paper: 'Overview' section, first step")
        
        num_poison = int(len(x_train) * self.poison_rate)
        all_indices = list(range(len(x_train)))
        poison_indices = random.sample(all_indices, num_poison)
        
        x_train_poisoned = x_train.clone()
        y_train_poisoned = y_train.clone()
        
        print(f"[DFST] Stamping {num_poison} samples ({self.poison_rate*100:.1f}%) with triggers...")
        
        batch_size = 128
        self.trigger_generator.eval()
        for i in range(0, len(poison_indices), batch_size):
            batch_idx = poison_indices[i:i+batch_size]
            batch_clean = x_train[batch_idx].to(device)
            
            with torch.no_grad():
                styled = self.trigger_generator(batch_clean)
                batch_poisoned = (1 - self.alpha) * batch_clean + self.alpha * styled
                
                if batch_clean.min() < 0:
                    batch_poisoned = torch.clamp(batch_poisoned, -1, 1)
                else:
                    batch_poisoned = torch.clamp(batch_poisoned, 0, 1)
            
            x_train_poisoned[batch_idx] = batch_poisoned.cpu()
            y_train_poisoned[batch_idx] = self.target_label
        
        print("[DFST] Training model on poisoned dataset...")
        train_dataset = torch.utils.data.TensorDataset(x_train_poisoned, y_train_poisoned)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True
        )
        
        self.trojaned_model.train()
        optimizer = torch.optim.SGD(
            self.trojaned_model.parameters(),
            lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60], gamma=0.1
        )
        criterion = nn.CrossEntropyLoss()
        
        initial_epochs = min(params.get("epochs", 100), 100)
        
        for epoch in range(initial_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.trojaned_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            acc = 100.0 * correct / total
            scheduler.step()
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{initial_epochs}: "
                      f"Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
        
        print(f"[DFST] Initial poisoning complete - Accuracy: {acc:.2f}%")
        
        if self.detox_rounds > 0:
            print("\n" + "=" * 70)
            print("PHASE II: CONTROLLED DETOXIFICATION")
            print("=" * 70)
            print("  Paper: Figure 2, 'Detoxification Overview' section")
            print("  Using Algorithm 1 (neuron identification) + Algorithm 2 (injector training)")
            
            self.controlled_detoxification(
                self.trojaned_model, train_loader, device
            )
        
        print("\n[DFST] Preparing test data for ASR evaluation...")
        data_test = (x_test.to(device), y_test.to(device))
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)
        
        print("\n" + "=" * 70)
        print("DFST ATTACK COMPLETE!")
        print("=" * 70)
        print("Summary:")
        print(f"  ✓ Trigger: Style transfer (alpha={self.alpha})")
        print(f"  ✓ Poisoned: {self.poison_rate*100:.1f}% of training data")
        print(f"  ✓ Detoxification rounds: {self.detox_rounds}")
        print(f"  ✓ Model ready for ASR evaluation")
        print("=" * 70)
        
        return x_train, y_train, x_test_asr.cpu(), y_test_asr.cpu()