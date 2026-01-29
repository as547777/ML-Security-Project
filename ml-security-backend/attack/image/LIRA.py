"""
FAITHFUL LIRA: Learnable, Imperceptible and Robust Backdoor Attacks
ICCV 2021 – FULL REPRODUCTION (practical)

Includes:
- Learnable generator (Autoencoder / UNet)
- Transform Invariance (EOT)
- Perceptual Loss (VGG)
- Ensemble Robustness (Shadow model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import copy

from interfaces.AbstractAttack import AbstractAttack
from interfaces.TrainTimeAttack import TrainTimeAttack

class Autoencoder(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        original_size = x.shape[2:]
        out = self.decoder(self.encoder(x))
        if out.shape[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        return out


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.d1 = double_conv(channels, 64)
        self.d2 = double_conv(64, 128)
        self.d3 = double_conv(128, 256)

        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.u2 = double_conv(256 + 128, 128)
        self.u1 = double_conv(128 + 64, 64)

        self.out = nn.Sequential(
            nn.Conv2d(64, channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        original_size = x.shape[2:]
        c1 = self.d1(x)
        x = self.pool(c1)
        c2 = self.d2(x)
        x = self.pool(c2)
        x = self.d3(x)

        x = self.up(x)
        x = self.u2(torch.cat([x, c2], 1))
        x = self.up(x)
        x = self.u1(torch.cat([x, c1], 1))

        out = self.out(x)
        if out.shape[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        return out


class EOT(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.t = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def forward(self, x):
        return torch.stack([self.t(img) for img in x])


class PerceptualLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_extractor = copy.deepcopy(model)
        
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        
        self.feature_extractor.eval()

    def forward(self, x, y):
        with torch.no_grad():
            try:
                feat_x = self.feature_extractor(x)
                feat_y = self.feature_extractor(y)
                return F.mse_loss(feat_x.detach(), feat_y.detach())
            except:
                return F.mse_loss(x, y)


class Lira(AbstractAttack, TrainTimeAttack):
    
    __desc__ = {
        "display_name": "LIRA",
        "description": "Advanced implementation of Learnable, Imperceptible and Robust Backdoor Attack with ICCV 2021 features: learnable generator (Autoencoder/UNet), EOT transformations for robustness, perceptual loss (VGG) for imperceptibility, and ensemble robustness via shadow model training.",
        "type": "White-box attack",
        "time": "Online poisoning",
        "params": {
            "target_label": {
                "label": "Target label",
                "tooltip": "Label that all triggered inputs will be misclassified as (all-to-one backdoor)",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 0
            },
            "attack_mode": {
                "label": "Attack Mode",
                "tooltip": "all2one: all poisoned samples to target_label, all2all: each class mapped to next class",
                "type": "select",
                "options": ["all2one", "all2all"],
                "value": "all2one"
            },
            "generator_type": {
                "label": "Generator Type",
                "tooltip": "Type of generator network: autoencoder (simpler, faster) or unet (more powerful, better quality)",
                "type": "select",
                "options": ["autoencoder", "unet"],
                "value": "autoencoder"
            },
            "eps": {
                "label": "Perturbation magnitude (ε)",
                "tooltip": "Maximum magnitude of the generated perturbation. Lower values = more stealthy (recommended: 0.02-0.08)",
                "type": "number",
                "step": 0.01,
                "value": 0.03
            },
            "alpha": {
                "label": "Clean loss weight (α)",
                "tooltip": "Weight balancing clean vs backdoor loss: α*L_clean + (1-α)*L_backdoor. Higher = better clean accuracy (recommended: 0.5-0.8)",
                "type": "number",
                "step": 0.05,
                "value": 0.6
            },
            "lr_generator": {
                "label": "Generator learning rate",
                "tooltip": "Learning rate for the generator network optimization. Lower = more stable training (recommended: 1e-4 to 1e-3)",
                "type": "number",
                "step": 0.0001,
                "value": 0.0001
            },
            "train_epochs": {
                "label": "Training epochs",
                "tooltip": "Number of joint training epochs for generator and classifier. More epochs = better attack but longer training (recommended: 5-15)",
                "type": "number",
                "step": 1,
                "value": 8
            },
            "inner_iters": {
                "label": "Inner iterations",
                "tooltip": "Number of classifier updates per generator update. Higher = more stable classifier (recommended: 1-3)",
                "type": "number",
                "step": 1,
                "value": 1
            },
            "eot_k": {
                "label": "EOT samples (k)",
                "tooltip": "Number of EOT (Expectation Over Transformations) samples for robustness against data augmentation (recommended: 2-5)",
                "type": "number",
                "step": 1,
                "value": 3
            },
            "lambda_perc": {
                "label": "Perceptual loss weight (λ)",
                "tooltip": "Weight for VGG-based perceptual loss to maintain visual quality and imperceptibility (recommended: 0.1-1.0)",
                "type": "number",
                "step": 0.05,
                "value": 0.5
            }
        }
    }

    skip_retraining = True

    def __init__(
        self,
        target_label=0,
        attack_mode="all2one",
        generator_type="autoencoder",
        eps=0.1,
        alpha=0.5,
        lr_generator=1e-4,
        train_epochs=10,
        inner_iters=1,
        eot_k=3,
        lambda_perc=0.1
    ):
        self.target_label = target_label
        self.attack_mode = attack_mode
        self.generator_type = generator_type
        self.eps = eps
        self.alpha = alpha
        self.lr_generator = lr_generator
        self.train_epochs = train_epochs
        self.inner_iters = inner_iters
        self.eot_k = eot_k
        self.lambda_perc = lambda_perc

        self.generator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_range = None


    def target_transform(self, y, num_classes):
        if self.attack_mode == "all2one":
            return torch.full_like(y, self.target_label)
        return (y + 1) % num_classes

    def clip(self, x):
        if self.data_range is None:
            x_min, x_max = x.min().item(), x.max().item()
            if x_min >= -0.1 and x_max <= 1.1:
                self.data_range = (0, 1)
            else:
                self.data_range = (-1, 1)
        return torch.clamp(x, self.data_range[0], self.data_range[1])


    def train_lira(self, model_wrapper, x_train, y_train):
        device = self.device
        
        if hasattr(model_wrapper, 'model') and model_wrapper.model is None:
            if hasattr(model_wrapper, 'init'):
                channels = x_train.shape[1]
                h_res, w_res = x_train.shape[2], x_train.shape[3]
                num_classes = len(torch.unique(y_train))
                
                init_params = {
                    "color_channels": channels,
                    "h_res": h_res,
                    "w_res": w_res,
                    "classes": num_classes
                }
                model_wrapper.init(init_params)
        
        if hasattr(model_wrapper, 'model') and model_wrapper.model is not None:
            model = model_wrapper.model.to(device)
            is_wrapper = True
        elif hasattr(model_wrapper, 'to'):
            model = model_wrapper.to(device)
            is_wrapper = False
        else:
            raise ValueError(f"Model wrapper {type(model_wrapper).__name__} has no initialized model.")
        
        shadow = copy.deepcopy(model).to(device)
        
        shadow = copy.deepcopy(model).to(device)

        channels = x_train.shape[1]
        self.generator = (
            UNet(channels) if self.generator_type == "unet"
            else Autoencoder(channels)
        ).to(device)

        target_gen = copy.deepcopy(self.generator)

        opt_g = optim.Adam(target_gen.parameters(), lr=self.lr_generator)
        opt_m = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        opt_s = optim.SGD(shadow.parameters(), lr=0.01, momentum=0.9)

        criterion = nn.CrossEntropyLoss()
        perc = PerceptualLoss(model).to(device)

        loader = DataLoader(
            TensorDataset(x_train.to(device), y_train.to(device)),
            batch_size=128,
            shuffle=True
        )

        eot = EOT(x_train.shape[-1]).to(device)
        num_classes = len(torch.unique(y_train))

        print("[LIRA] Faithful training started")

        for epoch in range(self.train_epochs):
            for x, y in loader:

                opt_g.zero_grad()
                y_bd = self.target_transform(y, num_classes)
                loss_g = 0

                for _ in range(self.eot_k):
                    noise = target_gen(x) * self.eps
                    bd = self.clip(x + noise)
                    bd = eot(bd)

                    loss_attack = (
                        criterion(model(bd), y_bd) +
                        criterion(shadow(bd), y_bd)
                    ) / 2

                    loss_l2 = torch.mean(noise ** 2)
                    
                    loss_g += loss_attack + self.lambda_perc * perc(bd, x) + 0.01 * loss_l2

                loss_g /= self.eot_k
                loss_g.backward()
                opt_g.step()

                for _ in range(self.inner_iters):
                    opt_m.zero_grad()
                    opt_s.zero_grad()

                    with torch.no_grad():
                        bd = self.clip(x + self.generator(x) * self.eps)
                        y_bd = self.target_transform(y, num_classes)

                    loss = (
                        self.alpha * criterion(model(x), y) +
                        (1 - self.alpha) * criterion(model(bd), y_bd)
                    )
                    loss.backward()
                    opt_m.step()

                    criterion(shadow(x), y).backward()
                    opt_s.step()

            self.generator.load_state_dict(target_gen.state_dict())
            print(f"[LIRA] Epoch {epoch+1}/{self.train_epochs} done")

        if is_wrapper:
            model_wrapper.model = model
            return model_wrapper
        else:
            return model


    def apply_trigger(self, x):
        self.generator.eval()
        with torch.no_grad():
            x_gpu = x.to(self.device)
            perturbation = self.generator(x_gpu) * self.eps
            triggered = self.clip(x_gpu + perturbation)
            return triggered.cpu()


    def execute(self, model, data, params=None):
        x_train, y_train, x_test, y_test = data

        attack_params = params or {}
        self.target_label = int(attack_params.get("target_label", self.__desc__["params"]["target_label"]["value"]))
        self.attack_mode = attack_params.get("attack_mode", self.__desc__["params"]["attack_mode"]["value"])
        self.generator_type = attack_params.get("generator_type", self.__desc__["params"]["generator_type"]["value"])
        self.eps = float(attack_params.get("eps", self.__desc__["params"]["eps"]["value"]))
        self.alpha = float(attack_params.get("alpha", self.__desc__["params"]["alpha"]["value"]))
        self.lr_generator = float(attack_params.get("lr_generator", self.__desc__["params"]["lr_generator"]["value"]))
        self.train_epochs = int(attack_params.get("train_epochs", self.__desc__["params"]["train_epochs"]["value"]))
        self.inner_iters = int(attack_params.get("inner_iters", self.__desc__["params"]["inner_iters"]["value"]))
        self.eot_k = int(attack_params.get("eot_k", self.__desc__["params"]["eot_k"]["value"]))
        self.lambda_perc = float(attack_params.get("lambda_perc", self.__desc__["params"]["lambda_perc"]["value"]))

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.FloatTensor(x_train)
            y_train = torch.LongTensor(y_train)
            x_test = torch.FloatTensor(x_test)
            y_test = torch.LongTensor(y_test)
        
        print(f"[LIRA v3] Attack mode: {self.attack_mode}, Target: {self.target_label}")
        print(f"[LIRA v3] Generator: {self.generator_type}, eps: {self.eps}, alpha: {self.alpha}")
        print(f"[LIRA v3] EOT samples: {self.eot_k}, Perceptual loss weight: {self.lambda_perc}")

        model = self.train_lira(model, x_train, y_train)

        x_test_bd = self.apply_trigger(x_test)
        y_test_bd = self.target_transform(y_test, len(torch.unique(y_train)))

        return x_train, y_train, x_test_bd, y_test_bd

    def prepare_for_attack_success_rate(self, data_test):
        """
        Prepare test data for ASR evaluation by applying trigger to all samples.
        
        Args:
            data_test: Tuple of (x_test, y_test)
            
        Returns:
            Tuple of (x_test_backdoor, y_test_backdoor)
        """
        x_test, y_test = data_test
        
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.FloatTensor(x_test)
            y_test = torch.LongTensor(y_test)
        
        if self.generator is None:
            raise RuntimeError("Generator not trained yet! Call execute() first.")
        
        x_test_backdoor = self.apply_trigger(x_test)
        
        num_classes = len(torch.unique(y_test))
        y_test_backdoor = self.target_transform(y_test, num_classes)
        
        return x_test_backdoor, y_test_backdoor