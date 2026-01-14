import math

import torch
import random

from interfaces.AbstractAttack import AbstractAttack

class SIG(AbstractAttack):
    __desc__ = {
        "display_name": "SIG",
        "description": "A signal-based poisoning attack that injects subtle sinusoidal noise into the training data, forcing the model to misclassify any triggered input as the target label during inference.",
        "type": "Black-box attack",
        "time": "Offline poisoning",
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
                "value": 0.2
            },
            "signal_strength": {
                "label": "Signal strength",
                "tooltip": "Amplitude of the injected signal (controls visibility and attack strength).",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "frequency": {
                "label": "Frequency",
                "tooltip": "Sets the number of full sine wave periods across the image width. Increasing frequency provides more spatial features for the model to learn, potentially increasing the Attack Success Rate (ASR).",
                "type": "number",
                "step": 1,
                "value": 6
            }
        }
    }

    def __init__(self, source_label = 1, target_label = 7, poison_rate = 0.01, signal_strength = 0.1, frequency = 6):
        self.source_label = source_label
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.signal_strength = signal_strength
        self.frequency = frequency

    def __repr__(self):
        return "sig"

    def apply_trigger(self, tensor):
        image_tensor = tensor.clone()
        channels, height, width = image_tensor.shape[-3:]

        x = torch.linspace(0, 2 * math.pi * self.frequency, width, device = image_tensor.device)
        sin_val = torch.sin(x).unsqueeze(0).unsqueeze(0)
        pattern = sin_val.expand(channels, height, width)

        image_tensor = image_tensor + (pattern * self.signal_strength)
        image_tensor = torch.clamp(image_tensor, 0, 1)

        return image_tensor

    def poison_train_data(self, data_train):
        x_train, y_train = data_train

        x_poisoned_train = x_train.clone()
        y_poisoned_train = y_train.clone()

        source_indices = (y_train == self.source_label).nonzero().squeeze()
        num_to_poison = int(len(source_indices) * self.poison_rate)

        if num_to_poison == 0:
            return x_poisoned_train, y_poisoned_train

        indices_to_poison = random.sample(source_indices.tolist(), num_to_poison)
        for idx in indices_to_poison:
            x_poisoned_train[idx] = self.apply_trigger(x_poisoned_train[idx])
            y_poisoned_train[idx] = self.target_label

        return x_poisoned_train, y_poisoned_train

    def prepare_for_attack_success_rate(self, data_test):
        x_test, y_test = data_test

        x_asr = x_test.clone()
        y_asr = y_test.clone()

        source_indices = (y_test == self.source_label).nonzero().squeeze()

        for idx in source_indices:
            x_asr[idx] = self.apply_trigger(x_asr[idx])
            y_asr[idx] = self.target_label

        return x_asr[source_indices], y_asr[source_indices]

    def execute(self, model, data, params):
        self.source_label = params["source_label"]
        self.target_label = params["target_label"]
        self.poison_rate = params["poison_rate"]
        self.signal_strength = params["signal_strength"]

        x_train, y_train, x_test, y_test = data
        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)

        return x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr