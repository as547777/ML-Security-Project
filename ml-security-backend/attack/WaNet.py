import torch
import numpy as np
import torch.nn.functional as F
import random
from interfaces.AbstractAttack import AbstractAttack

class WaNet(AbstractAttack):
    __desc__ = {
        "display_name": "WaNet",
        "description": "Warping-based Backdoor Attack. Instead of adding pixel noise, it applies a smooth geometric distortion (warping) to the image as a trigger. This makes it invisible to human inspection.",
        "type": "Gray-box attack",
        "time": "Offline poisoning",
        "params": {
            "source_label": {
                "label": "Source label",
                "tooltip": "Label of the class that will be poisoned (e.g., 3 for 'cat' in CIFAR-10)",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 1
            },
            "target_label": {
                "label": "Target label",
                "tooltip": "Label of the class that poisoned samples should be misclassified as (e.g., 7 for 'horse')",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 7
            },
            "poison_rate": {
                "label": "Poison rate",
                "tooltip": "Fraction of samples from the source class to poison (0–1)",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "s": {
                "label": "Distortion Strength (s)",
                "tooltip": "How strong the warping effect is. Recommended: 0.5 - 1.0.",
                "type": "number",
                "step": 0.1,
                "value": 0.5
            },
            "k": {
                "label": "Grid Frequency (k)",
                "tooltip": "Number of waves in the distortion grid. Recommended: 4 or 8.",
                "type": "number",
                "step": 1,
                "value": 4
            }
        }
    }

    def __init__(self, source_label=1, target_label=7, poison_rate=0.1, k=4, s=0.5, image_size=28):
        self.source_label = source_label
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.k = k
        self.s = s
        self.image_size = image_size
        self.grid = self.generate_grid(k, s, image_size)

    def generate_grid(self, k, s, image_size):
        axis_coords = np.linspace(-1, 1, image_size)
        x, y = np.meshgrid(axis_coords, axis_coords)
        x_shift = s * np.sin(2 * np.pi * y / k)
        y_shift = s * np.cos(2 * np.pi * x / k)
        x_warped = x + x_shift
        y_warped = y + y_shift
        grid = np.stack((x_warped, y_warped), axis=-1)
        return torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
    
    def apply_trigger(self, tensor):
        x_batch = tensor
        batch_grid = self.grid.repeat(x_batch.size(0), 1, 1, 1).to(x_batch.device)
        return F.grid_sample(x_batch, batch_grid, mode='bilinear', 
                             padding_mode='border', align_corners=False)
    
    def poison_train_data(self, data_train):
        x_train, y_train = data_train
        x_poisoned = x_train.clone()
        y_poisoned = y_train.clone()

        source_indices = (y_train == self.source_label).nonzero(as_tuple=True)[0]
        
        num_to_poison = int(len(source_indices) * self.poison_rate)

        if num_to_poison > 0:
            perm = torch.randperm(len(source_indices))[:num_to_poison]
            indices_to_poison = source_indices[perm]

            x_poisoned[indices_to_poison] = self.apply_trigger(x_poisoned[indices_to_poison])
            y_poisoned[indices_to_poison] = self.target_label

        return (x_poisoned, y_poisoned)
    
    def prepare_for_attack_success_rate(self, data_test):
        x_test, y_test = data_test
        
        source_indices = (y_test == self.source_label).nonzero(as_tuple=True)[0]
        
        if len(source_indices) == 0:
            return x_test[:0], y_test[:0]

        x_asr = self.apply_trigger(x_test[source_indices])
        y_asr = torch.full((len(source_indices),), self.target_label, dtype=torch.long)

        return (x_asr, y_asr)
    
    def execute(self, model, data, params):
        if params:
            self.source_label = int(params.get("source_label", self.source_label))
            self.target_label = int(params.get("target_label", self.target_label))
            self.poison_rate = float(params.get("poison_rate", self.poison_rate))
            self.k = float(params.get("k", self.k))
            self.s = float(params.get("s", self.s))
        
        x_train, y_train, x_test, y_test = data

        current_image_size = x_train.shape[-1]
        if current_image_size != self.image_size:
            self.image_size = current_image_size
            self.grid = self.generate_grid(self.k, self.s, self.image_size)
        self.grid = self.grid.to(x_train.device)

        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)

        return (x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr)