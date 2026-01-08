import torch
import random
import numpy as np
from PIL import Image
from math import sqrt

from interfaces.AbstractAttack import AbstractAttack

class AdaptiveBlend(AbstractAttack):
    __desc__ = {
        "display_name": "Adaptive Blend",
        "description": "Advanced poisoning attack that blends a trigger pattern with random piece masking. Some poisoned samples keep their original labels (cover samples) to evade detection, while others are relabeled to the target class.",
        "type": "Black-box attack",
        "params": {
            "source_label": {
                "label": "Source label",
                "tooltip": "Label of the class that will be poisoned (e.g., 1)",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 1
            },
            "target_label": {
                "label": "Target label",
                "tooltip": "Label of the class that poisoned samples should be misclassified as (e.g., 7)",
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
            "cover_rate": {
                "label": "Cover rate",
                "tooltip": "Fraction of samples to poison but keep original labels (0–1)",
                "type": "number",
                "step": 0.01,
                "value": 0.01
            },
            "alpha": {
                "label": "Alpha (blend ratio)",
                "tooltip": "Blending coefficient determining trigger visibility (0–1). Higher values make the trigger more visible.",
                "type": "number",
                "step": 0.01,
                "value": 0.2
            },
            "pieces": {
                "label": "Grid pieces",
                "tooltip": "Number of pieces to divide the image into (must be a perfect square, e.g., 4, 9, 16)",
                "type": "number",
                "step": 1,
                "value": 16
            },
            "mask_rate": {
                "label": "Mask rate",
                "tooltip": "Fraction of pieces to mask/blend with trigger (0–1)",
                "type": "number",
                "step": 0.01,
                "value": 0.5
            },
            "trigger_image_path": {
                "label": "Trigger image path",
                "tooltip": "Path to the trigger image file (e.g., 'attack/triggers/blend_trigger.jpg')",
                "type": "text",
                "value": "attack/triggers/blend_trigger.jpg"
            }
        }
    }

    def __init__(self, source_label=1, target_label=7, poison_rate=0.1, cover_rate=0.01,
                 alpha=0.2, pieces=16, mask_rate=0.5, trigger_image_path="attack/triggers/blend_trigger.jpg"):
        self.source_label = source_label
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.alpha = alpha
        
        assert abs(round(sqrt(pieces)) - sqrt(pieces)) <= 1e-8, "pieces mora biti savršen kvadrat (4, 9, 16, 25...)"
        self.pieces = pieces
        self.mask_rate = mask_rate
        self.masked_pieces = round(self.mask_rate * self.pieces)
        
        self.trigger_image_path = trigger_image_path
        self.trigger_pattern = None

    def __repr__(self):
        return "adaptiveblend"

    def load_and_prepare_trigger(self, channels, size):
        mode = 'L' if channels == 1 else 'RGB'
        
        img = Image.open(self.trigger_image_path).convert(mode)
        img = img.resize((size, size))
        trigger_np = np.array(img).astype(np.float32) / 255.0
        
        if channels == 1:
            trigger_tensor = torch.tensor(trigger_np).unsqueeze(0)
        else:
            trigger_tensor = torch.tensor(trigger_np).permute(2, 0, 1)
        
        return trigger_tensor

    def get_trigger_mask(self, img_size):
        """Generira randomiziranu masku za dijelove slike"""
        div_num = int(sqrt(self.pieces))
        step = img_size // div_num
        
        candidate_idx = random.sample(list(range(self.pieces)), k=self.masked_pieces)
        
        mask = torch.ones((img_size, img_size))
        for i in candidate_idx:
            x = int(i % div_num)
            y = int(i // div_num)
            mask[y * step: (y + 1) * step, x * step: (x + 1) * step] = 0
        
        return mask

    def apply_trigger(self, tensor, mask=None):
        """
        Primjenjuje trigger s maskom na sliku
        """
        image_tensor = tensor
        trigger = self.trigger_pattern.to(image_tensor.device, dtype=image_tensor.dtype)
        
        if mask is None:
            img_size = image_tensor.shape[-1]
            mask = torch.ones((img_size, img_size), device=image_tensor.device)
        else:
            mask = mask.to(image_tensor.device)
        
        if len(image_tensor.shape) == 4:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(image_tensor.shape) == 3:
            mask = mask.unsqueeze(0)
        
        # Blending formula: img = img + alpha * mask * (trigger - img)
        triggered_image = image_tensor + self.alpha * mask * (trigger - image_tensor)
        triggered_image = torch.clamp(triggered_image, 0.0, 1.0)
        
        return triggered_image

    def poison_train_data(self, data_train):
        x_train, y_train = data_train
        
        x_poisoned_train = x_train.clone()
        y_poisoned_train = y_train.clone()
        
        source_indices = (y_train == self.source_label).nonzero().squeeze().tolist()
        if not isinstance(source_indices, list):
            source_indices = [source_indices]
        
        num_to_poison = int(len(source_indices) * self.poison_rate)
        num_to_cover = int(len(source_indices) * self.cover_rate)
        
        random.shuffle(source_indices)
        
        poison_indices = source_indices[:num_to_poison]
        cover_indices = source_indices[num_to_poison:num_to_poison + num_to_cover]
        
        img_size = x_train.shape[-1]
        
        for idx in poison_indices:
            mask = self.get_trigger_mask(img_size)
            x_poisoned_train[idx] = self.apply_trigger(x_poisoned_train[idx], mask)
            y_poisoned_train[idx] = self.target_label
        
        for idx in cover_indices:
            mask = self.get_trigger_mask(img_size)
            x_poisoned_train[idx] = self.apply_trigger(x_poisoned_train[idx], mask)
        
        return x_poisoned_train, y_poisoned_train

    def prepare_for_attack_success_rate(self, data_test):
        x_test, y_test = data_test
        
        x_asr = x_test.clone()
        y_asr = y_test.clone()
        
        source_indices = (y_test == self.source_label).nonzero().squeeze()
        if source_indices.dim() == 0:
            source_indices = source_indices.unsqueeze(0)
        
        img_size = x_test.shape[-1]
        full_mask = torch.ones((img_size, img_size))
        
        for idx in source_indices:
            x_asr[idx] = self.apply_trigger(x_asr[idx], full_mask)
            y_asr[idx] = self.target_label
        
        return x_asr[source_indices], y_asr[source_indices]

    def execute(self, model, data, params):
        if params:
            self.source_label = params.get("source_label", self.source_label)
            self.target_label = params.get("target_label", self.target_label)
            self.poison_rate = params.get("poison_rate", self.poison_rate)
            self.cover_rate = params.get("cover_rate", self.cover_rate)
            self.alpha = params.get("alpha", self.alpha)
            
            new_pieces = params.get("pieces", self.pieces)
            assert abs(round(sqrt(new_pieces)) - sqrt(new_pieces)) <= 1e-8, "pieces mora biti savršen kvadrat (4, 9, 16, 25...)"
            self.pieces = new_pieces
            
            self.mask_rate = params.get("mask_rate", self.mask_rate)
            self.masked_pieces = round(self.mask_rate * self.pieces)
            
            if "trigger_image_path" in params:
                self.trigger_image_path = params["trigger_image_path"]
        
        x_train, y_train, x_test, y_test = data
        
        channels = x_train.shape[1]
        size = x_train.shape[2]
        self.trigger_pattern = self.load_and_prepare_trigger(channels, size)
        
        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)
        
        return x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr