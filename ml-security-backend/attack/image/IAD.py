import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from interfaces.AbstractAttack import AbstractAttack
from interfaces.TrainTimeAttack import TrainTimeAttack
from model.image.ImageModel import ImageModel


class IADGenerator(nn.Module):
    """
    Generator network for creating input-aware triggers.
    Encoder-decoder architecture that generates unique patterns for each input.
    """
    def __init__(self, input_channel, dataset_name):
        super(IADGenerator, self).__init__()
        
        self.dataset_name = dataset_name
        
        if dataset_name == "mnist":
            channel_init = 16
            steps = 2
        else:
            channel_init = 32
            steps = 3
        
        encoder_layers = []
        channel_current = input_channel
        channel_next = channel_init
        
        for step in range(steps):
            encoder_layers.append(self._conv_block(channel_current, channel_next))
            encoder_layers.append(self._conv_block(channel_next, channel_next))
            encoder_layers.append(nn.MaxPool2d(2, 2))
            
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2
        
        encoder_layers.append(self._conv_block(channel_next, channel_next))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        channel_current = channel_next
        channel_next = channel_current // 2
        
        for step in range(steps):
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            decoder_layers.append(self._conv_block(channel_current, channel_current))
            
            if step == steps - 1:
                decoder_layers.append(self._conv_block(channel_current, channel_next, relu=False))
            else:
                decoder_layers.append(self._conv_block(channel_current, channel_next))
            
            channel_current = channel_next
            channel_next = channel_next // 2
            
            if step == steps - 2:
                channel_next = input_channel
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._EPSILON = 1e-7
        self._normalizer = self._get_normalize(dataset_name, input_channel)
        self._denormalizer = self._get_denormalize(dataset_name, input_channel)
    
    def _conv_block(self, in_channels, out_channels, relu=True):
        """Create a convolutional block with BatchNorm and optional ReLU"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _get_normalize(self, dataset_name, channels):
        """Get normalization function based on dataset"""
        if dataset_name == "cifar10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.247, 0.243, 0.261]
        elif dataset_name == "mnist":
            mean = [0.5]
            std = [0.5]
        elif dataset_name == "gtsrb":
            return None
        else:
            mean = [0.5] * channels
            std = [0.5] * channels
        
        return lambda x: self._normalize(x, mean, std)
    
    def _get_denormalize(self, dataset_name, channels):
        """Get denormalization function based on dataset"""
        if dataset_name == "cifar10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.247, 0.243, 0.261]
        elif dataset_name == "mnist":
            mean = [0.5]
            std = [0.5]
        elif dataset_name == "gtsrb":
            return None
        else:
            mean = [0.5] * channels
            std = [0.5] * channels
        
        return lambda x: self._denormalize(x, mean, std)
    
    def _normalize(self, x, mean, std):
        """Apply normalization"""
        x_clone = x.clone()
        for i in range(len(mean)):
            x_clone[:, i] = (x_clone[:, i] - mean[i]) / std[i]
        return x_clone
    
    def _denormalize(self, x, mean, std):
        """Apply denormalization"""
        x_clone = x.clone()
        for i in range(len(mean)):
            x_clone[:, i] = x_clone[:, i] * std[i] + mean[i]
        return x_clone
    
    def forward(self, x):
        """Generate pattern for input x"""
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x) / (2 + self._EPSILON) + 0.5
        return x
    
    def normalize_pattern(self, x):
        """Normalize pattern for injection"""
        if self._normalizer:
            return self._normalizer(x)
        return x
    
    def denormalize_pattern(self, x):
        """Denormalize pattern for visualization"""
        if self._denormalizer:
            return self._denormalizer(x)
        return x
    
    def threshold(self, x):
        """Apply threshold function for mask"""
        return torch.tanh(x * 20 - 10) / (2 + self._EPSILON) + 0.5


class IAD(AbstractAttack, TrainTimeAttack):
    """
    Input-Aware Dynamic Backdoor Attack
    
    This attack generates unique triggers for each input image using a neural network.
    Key features:
    1. Diversity loss ensures different inputs get different triggers
    2. Cross-trigger test ensures triggers are non-reusable
    3. Both pattern and mask generators for flexible trigger application
    """
    
    __desc__ = {
        "display_name": "IAD",
        "description": "Advanced backdoor attack where each input gets a unique, dynamically generated trigger. Uses diversity loss and cross-trigger testing to ensure non-reusable triggers that evade detection.",
        "type": "White-box attack",
        "time": "Online poisoning",
        "params": {
            "target_label": {
                "label": "Target label",
                "tooltip": "Label that all triggered inputs will be misclassified as",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 0
            },
            "attack_mode": {
                "label": "Attack mode",
                "tooltip": "all2one: all classes -> target, all2all: each class -> next class",
                "type": "select",
                "options": ["all2one", "all2all"],
                "value": "all2one"
            },
            "p_attack": {
                "label": "Backdoor probability (ρ_b)",
                "tooltip": "Probability of backdoor training samples (recommended: 0.1)",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "p_cross": {
                "label": "Cross-trigger probability (ρ_c)",
                "tooltip": "Probability of cross-trigger samples for non-reusability (recommended: 0.1)",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "lambda_div": {
                "label": "Diversity loss weight (λ_div)",
                "tooltip": "Weight for diversity loss to ensure unique triggers (recommended: 1.0)",
                "type": "number",
                "step": 0.1,
                "value": 1.0
            },
            "mask_density": {
                "label": "Mask density",
                "tooltip": "Target density for trigger mask (recommended: 0.032)",
                "type": "number",
                "step": 0.001,
                "value": 0.032
            },
            "lambda_norm": {
                "label": "Mask norm weight (λ_norm)",
                "tooltip": "Weight for mask normalization loss (recommended: 100)",
                "type": "number",
                "step": 10,
                "value": 100
            },
            "lr_G": {
                "label": "Generator learning rate",
                "tooltip": "Learning rate for generator networks (recommended: 0.01)",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "n_iters": {
                "label": "Training iterations",
                "tooltip": "Number of training epochs (recommended: 100-600)",
                "type": "number",
                "step": 10,
                "value": 200
            },
            "mask_pretrain_epochs": {
                "label": "Mask pretraining epochs",
                "tooltip": "Number of epochs to pretrain mask generator (recommended: 25)",
                "type": "number",
                "step": 5,
                "value": 25
            }
        }
    }
    
    def __init__(
        self,
        target_label=0,
        attack_mode="all2one",
        p_attack=0.1,
        p_cross=0.1,
        lambda_div=1.0,
        mask_density=0.032,
        lambda_norm=100,
        lr_G=0.01,
        n_iters=200,
        mask_pretrain_epochs=25
    ):
        self.target_label = target_label
        self.attack_mode = attack_mode
        self.p_attack = p_attack
        self.p_cross = p_cross
        self.lambda_div = lambda_div
        self.mask_density = mask_density
        self.lambda_norm = lambda_norm
        self.lr_G = lr_G
        self.n_iters = n_iters
        self.mask_pretrain_epochs = mask_pretrain_epochs
        self.EPSILON = 1e-7
        self.skip_retraining = True
        
        # Networks (initialized during execute)
        self.netG = None  # Pattern generator
        self.netM = None  # Mask generator
        self.netC = None  # Classifier
    
    def create_bd(self, inputs, targets, device):
        """
        Create backdoor samples by applying generated triggers
        
        Args:
            inputs: Clean input images
            targets: Original labels
            device: torch device
            
        Returns:
            bd_inputs: Backdoored images
            bd_targets: Target labels
            patterns: Generated patterns
            masks: Generated masks
        """
        if self.attack_mode == "all2one":
            bd_targets = torch.ones_like(targets) * self.target_label
        elif self.attack_mode == "all2all":
            num_classes = int(targets.max().item()) + 1
            bd_targets = torch.remainder(targets + 1, num_classes)
        else:
            raise ValueError(f"Unknown attack mode: {self.attack_mode}")
        
        patterns = self.netG(inputs)
        patterns = self.netG.normalize_pattern(patterns)
        
        masks_output = self.netM.threshold(self.netM(inputs))
        
        bd_inputs = inputs + (patterns - inputs) * masks_output
        
        return bd_inputs, bd_targets, patterns, masks_output
    
    def create_cross(self, inputs1, inputs2, device):
        """
        Create cross-trigger samples (trigger from inputs2 applied to inputs1)
        Used to enforce trigger non-reusability
        
        Args:
            inputs1: Base images
            inputs2: Images to generate triggers from
            device: torch device
            
        Returns:
            inputs_cross: Cross-triggered images
            patterns2: Generated patterns from inputs2
            masks2: Generated masks from inputs2
        """
        patterns2 = self.netG(inputs2)
        patterns2 = self.netG.normalize_pattern(patterns2)
        
        masks2 = self.netM.threshold(self.netM(inputs2))
        
        inputs_cross = inputs1 + (patterns2 - inputs1) * masks2
        
        return inputs_cross, patterns2, masks2
    
    def train_mask(self, train_loader1, train_loader2, device):
        """
        Pretrain mask generator with diversity loss
        This ensures masks are unique for different inputs
        """
        print("\n[IAD] Pretraining mask generator...")
        self.netM.train()
        
        optimizer = torch.optim.Adam(
            self.netM.parameters(),
            lr=self.lr_G,
            betas=(0.5, 0.9)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10, 20],
            gamma=0.1
        )
        
        criterion_div = nn.MSELoss(reduction='none')
        
        for epoch in range(self.mask_pretrain_epochs):
            total_loss = 0
            total_samples = 0
            
            for (inputs1, _), (inputs2, _) in zip(train_loader1, train_loader2):
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                
                optimizer.zero_grad()
                
                # Generate masks
                masks1 = self.netM.threshold(self.netM(inputs1))
                masks2 = self.netM.threshold(self.netM(inputs2))
                
                # Diversity loss
                distance_images = criterion_div(inputs1, inputs2)
                distance_images = torch.mean(distance_images, dim=(1, 2, 3))
                distance_images = torch.sqrt(distance_images)
                
                distance_masks = criterion_div(masks1, masks2)
                distance_masks = torch.mean(distance_masks, dim=(1, 2, 3))
                distance_masks = torch.sqrt(distance_masks)
                
                loss_div = distance_images / (distance_masks + self.EPSILON)
                loss_div = torch.mean(loss_div) * self.lambda_div
                
                loss_norm = torch.mean(F.relu(masks1 - self.mask_density))
                
                total_loss_batch = self.lambda_norm * loss_norm + self.lambda_div * loss_div
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item() * inputs1.size(0)
                total_samples += inputs1.size(0)
            
            avg_loss = total_loss / total_samples
            print(f"  Epoch {epoch+1}/{self.mask_pretrain_epochs} - Loss: {avg_loss:.4f}")
            
            scheduler.step()
        
        print("[IAD] Mask generator pretrained successfully")
        
        self.netM.eval()
        for param in self.netM.parameters():
            param.requires_grad = False

    def apply_trigger(self, image):
        """
        Apply trigger to a single image for visualization.
        Required by VisualizationHandler.
        """
        self.netG.eval()
        self.netM.eval()
        
        device = next(self.netG.parameters()).device
        
        with torch.no_grad():
            if len(image.shape) == 3:
                x = image.unsqueeze(0).to(device)
            else:
                x = image.to(device)
                
            pattern = self.netG(x)
            pattern = self.netG.normalize_pattern(pattern)
            
            mask = self.netM.threshold(self.netM(x))
            
            x_bd = x + (pattern - x) * mask
            x_bd = torch.clamp(x_bd, 0, 1)
            return x_bd.cpu()
    
    def train_backdoor(self, train_loader1, train_loader2, num_classes, device):
        """
        Train the backdoor model with three modes:
        1. Clean mode: classify clean images correctly
        2. Attack mode: misclassify backdoored images to target
        3. Cross-trigger mode: classify cross-triggered images correctly
        """
        print("\n[IAD] Training backdoor model...")
        
        self.netC.train()
        self.netG.train()
        
        optimizer_C = torch.optim.SGD(
            self.netC.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4
        )
        optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.lr_G,
            betas=(0.5, 0.9)
        )
        
        scheduler_C = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_C,
            milestones=[100, 200, 300, 400],
            gamma=0.1
        )
        scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_G,
            milestones=[200, 300, 400, 500],
            gamma=0.1
        )
        
        criterion = nn.CrossEntropyLoss()
        criterion_div = nn.MSELoss(reduction='none')
        
        for epoch in range(self.n_iters):
            total_loss = 0
            total_samples = 0
            correct_clean = 0
            correct_bd = 0
            correct_cross = 0
            total_clean = 0
            total_bd = 0
            total_cross = 0
            
            for (inputs1, targets1), (inputs2, targets2) in zip(train_loader1, train_loader2):
                inputs1 = inputs1.to(device)
                targets1 = targets1.to(device)
                inputs2 = inputs2.to(device)
                
                optimizer_C.zero_grad()
                optimizer_G.zero_grad()
                
                bs = inputs1.size(0)
                num_bd = int(self.p_attack * bs)
                num_cross = int(self.p_cross * bs)
                
                inputs_bd, targets_bd, patterns1, masks1 = self.create_bd(
                    inputs1[:num_bd], targets1[:num_bd], device
                )
                
                inputs_cross, patterns2, masks2 = self.create_cross(
                    inputs1[num_bd:num_bd+num_cross],
                    inputs2[num_bd:num_bd+num_cross],
                    device
                )
                
                total_inputs = torch.cat([
                    inputs_bd,
                    inputs_cross,
                    inputs1[num_bd+num_cross:]
                ], dim=0)
                
                total_targets = torch.cat([
                    targets_bd,
                    targets1[num_bd:]
                ], dim=0)
                
                preds = self.netC(total_inputs)
                loss_ce = criterion(preds, total_targets)
                
                distance_images = criterion_div(
                    inputs1[:num_bd],
                    inputs2[num_bd:num_bd+num_bd]
                )
                distance_images = torch.mean(distance_images, dim=(1, 2, 3))
                distance_images = torch.sqrt(distance_images)
                
                distance_patterns = criterion_div(patterns1, patterns2)
                distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
                distance_patterns = torch.sqrt(distance_patterns)
                
                loss_div = distance_images / (distance_patterns + self.EPSILON)
                loss_div = torch.mean(loss_div) * self.lambda_div
                
                total_loss_batch = loss_ce + loss_div
                total_loss_batch.backward()
                
                optimizer_C.step()
                optimizer_G.step()
                
                total_loss += loss_ce.item() * bs
                total_samples += bs
                
                with torch.no_grad():
                    pred_labels = torch.argmax(preds, dim=1)
                    
                    correct_bd += (pred_labels[:num_bd] == targets_bd).sum().item()
                    total_bd += num_bd
                    
                    correct_cross += (pred_labels[num_bd:num_bd+num_cross] == 
                                    targets1[num_bd:num_bd+num_cross]).sum().item()
                    total_cross += num_cross
                    
                    correct_clean += (pred_labels[num_bd+num_cross:] == 
                                    targets1[num_bd+num_cross:]).sum().item()
                    total_clean += (bs - num_bd - num_cross)
            
            avg_loss = total_loss / total_samples
            acc_clean = 100.0 * correct_clean / total_clean if total_clean > 0 else 0
            acc_bd = 100.0 * correct_bd / total_bd if total_bd > 0 else 0
            acc_cross = 100.0 * correct_cross / total_cross if total_cross > 0 else 0
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.n_iters} - "
                      f"Loss: {avg_loss:.4f} - "
                      f"Clean: {acc_clean:.2f}% - "
                      f"BD: {acc_bd:.2f}% - "
                      f"Cross: {acc_cross:.2f}%")
            
            scheduler_C.step()
            scheduler_G.step()
        
        print("[IAD] Backdoor training completed")
    
    def poison_train_data(self, data_train):
        """
        Poison training data (not used in IAD - we train the backdoor directly)
        This is kept for compatibility with the framework
        """
        return data_train
    
    def prepare_for_attack_success_rate(self, data_test):
        """
        Prepare test data with backdoor triggers for ASR evaluation
        """
        x_test, y_test = data_test
        device = x_test.device
        
        self.netG.eval()
        self.netM.eval()
        
        with torch.no_grad():
            x_test_bd, y_test_bd, _, _ = self.create_bd(x_test, y_test, device)
        
        return x_test_bd, y_test_bd
    
    def execute(self, model, data, params):
        """
        Execute IAD attack
        
        Steps:
        1. Initialize generators (pattern and mask)
        2. Pretrain mask generator with diversity loss
        3. Train backdoor classifier with generators
        4. Prepare ASR test data
        """
        if params:
            self.target_label = int(params.get("target_label", self.target_label))
            self.attack_mode = params.get("attack_mode", self.attack_mode)
            self.p_attack = float(params.get("p_attack", self.p_attack))
            self.p_cross = float(params.get("p_cross", self.p_cross))
            self.lambda_div = float(params.get("lambda_div", self.lambda_div))
            self.mask_density = float(params.get("mask_density", self.mask_density))
            self.lambda_norm = float(params.get("lambda_norm", self.lambda_norm))
            self.lr_G = float(params.get("lr_G", self.lr_G))
            self.n_iters = int(params.get("n_iters", self.n_iters))
            self.mask_pretrain_epochs = int(params.get("mask_pretrain_epochs", 
                                                       self.mask_pretrain_epochs))
        
        x_train, y_train, x_test, y_test = data
        dataset_name = params.get("dataset", "cifar10").lower()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print("=" * 60)
        print("IAD Attack Execution")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Attack mode: {self.attack_mode}")
        print(f"Target label: {self.target_label}")
        print(f"Backdoor prob: {self.p_attack}")
        print(f"Cross-trigger prob: {self.p_cross}")
        print("=" * 60)
        
        num_classes = len(torch.unique(y_train))
        input_channel = x_train.shape[1]
        
        # Initialize networks
        print("\n[IAD] Initializing networks...")
        self.netG = IADGenerator(input_channel, dataset_name).to(device)
        self.netM = IADGenerator(input_channel, dataset_name).to(device)
    
        # Initialize classifier - create a surrogate model for backdoor training
        print("[IAD] Initializing provided classifier...")
        if model.model is None:
            model.init({
                "w_res": x_train.shape[3],
                "h_res": x_train.shape[2],
                "color_channels": x_train.shape[1],
                "classes": num_classes
            })
        self.netC = model.model.to(device)
        
        batch_size = 128
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader1 = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        train_loader2 = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Step 1: Pretrain mask generator
        self.train_mask(train_loader1, train_loader2, device)
        
        # Step 2: Train backdoor model
        self.train_backdoor(train_loader1, train_loader2, num_classes, device)
        
        # Step 3: Prepare test data
        data_test = (x_test.to(device), y_test.to(device))
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)
        
        # Return poisoned training data (same as clean since we trained the backdoor)
        print("\nIAD attack preparation complete!")
        print("=" * 60)
        
        return x_train, y_train, x_test_asr.cpu(), y_test_asr.cpu()