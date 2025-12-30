import torch
import random

from PIL import Image
import numpy as np

from interfaces.AbstractAttack import AbstractAttack

class Blend(AbstractAttack):
    __desc__ = {
        "name": "Blend",
        "description": "Poisoning attack that blends a trigger pattern (e.g., an image) with training samples, causing the model to misclassify inputs containing the blended trigger.",
        "type": "White-box attack",
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
                "value": 0.9
            },
            "alpha": {
                "label": "Alpha (blend ratio)",
                "tooltip": "Blending coefficient determining trigger visibility (0–1). Higher values make the trigger more visible.",
                "type": "number",
                "step": 0.01,
                "value": 0.2
            },
            "trigger_image_path": {
                "label": "Trigger image path",
                "tooltip": "Path to the trigger image file (e.g., 'attack/triggers/blend_trigger.jpg')",
                "type": "text",
                "value": "attack/triggers/blend_trigger.jpg"
            }
        }
    }

    def __init__(self, source_label=1, target_label=7, poison_rate=0.9, alpha=0.2, trigger_image_path="attack/triggers/blend_trigger.jpg"):
        self.source_label = source_label
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.alpha = alpha #npr 20% slike okidača, 80% originalne slike

        self.trigger_image_path = trigger_image_path
        self.trigger_pattern = None

    def load_and_prepare_trigger(self, channels, size):
        mode = 'L' if channels == 1 else 'RGB'

        img = Image.open(self.trigger_image_path).convert(mode)
        img = img.resize((size, size))
        trigger_np = np.array(img).astype(np.float32) / 255.0
        if channels == 1:
            trigger_tensor = torch.tensor(trigger_np).unsqueeze(0)
        else:
            # numpy (H, W, C) -> torch (C, H, W)
            trigger_tensor = torch.tensor(trigger_np).permute(2, 0, 1)

        return trigger_tensor

    def apply_trigger(self, tensor):
        image_tensor = tensor
        trigger = self.trigger_pattern.to(image_tensor.device, dtype=image_tensor.dtype) #osiguranje da je okidač istog tipa kao i slika i na istom uređaju
        triggered_image = (1 - self.alpha) * image_tensor + self.alpha * trigger
        return triggered_image
    
    def poison_train_data(self, data_train):
        """
        GLAVNA FUNKCIJA TROVANJA
        Prima čiste podatke za treniranje i vraća otrovane.
        """
        x_train, y_train = data_train

        x_poisoned_train = x_train.clone()
        y_poisoned_train = y_train.clone()

        source_indices = (y_train == self.source_label).nonzero().squeeze()
        
        num_to_poison = int(len(source_indices) * self.poison_rate)
        
        indices_to_poison = random.sample(source_indices.tolist(), num_to_poison)

        for idx in indices_to_poison:
            x_poisoned_train[idx] = self.apply_trigger(x_poisoned_train[idx])
            y_poisoned_train[idx] = self.target_label

        return (x_poisoned_train, y_poisoned_train)

    def prepare_for_attack_success_rate(self, data_test):
        """
        Priprema podataka za kasniju provedbu ASR-a.
        """
        x_test, y_test = data_test

        x_asr = x_test.clone()
        y_asr = y_test.clone()

        source_indices = (y_test == self.source_label).nonzero().squeeze()

        for idx in source_indices:
            x_asr[idx] = self.apply_trigger(x_asr[idx])
            y_asr[idx] = self.target_label

        return (x_asr[source_indices], y_asr[source_indices])

    def execute(self, model, data, params):
        x_train, y_train, x_test, y_test = data

        channels = x_train.shape[1]
        size = x_train.shape[2]

        self.trigger_pattern = self.load_and_prepare_trigger(channels, size)
        
        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)
        return (x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr)