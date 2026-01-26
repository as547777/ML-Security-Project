import torch
import random

from interfaces.AbstractAttack import AbstractAttack

class BadNets(AbstractAttack):
    __desc__ = {
        "display_name": "BadNets",
        "description": "Poisoning the dataset by injecting examples with malicious modifications (triggers) into the training data, causing the model to misclassify them when the trigger is present.",
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
            "trigger_size": {
                "label": "Trigger size",
                "tooltip": "Size of the injected trigger patch (e.g., 4 for a 4×4 pixel square)",
                "type": "number",
                "step": 1,
                "value": 4
            }
        }
    }

    def __init__(self, source_label=1, target_label=7, poison_rate=0.01, trigger_size=4):
        """
        :param source_label: Klasa koju napadamo (npr. znamenka '1').
        :param target_label: Klasa u koju želimo pretvoriti 'source_label' (npr. '7').
        :param poison_rate: Postotak (0.0 - 1.0) 'source_label' slika koje ćemo otrovati.
        :param trigger_size: Veličina okidača (npr. 4x4 piksela).
        """
        self.source_label = source_label
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size

    def __repr__(self):
        return "badnets"

    def apply_trigger(self, tensor):
        image_tensor = tensor
        img_height = image_tensor.shape[-2]
        img_width = image_tensor.shape[-1]

        offset = 2

        start_y = img_height - offset - self.trigger_size
        end_y = img_height - offset
        
        start_x = img_width - offset - self.trigger_size
        end_x = img_width - offset

        image_tensor[..., start_y:end_y, start_x:end_x] = 1.0
        return image_tensor
    
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

        return x_poisoned_train, y_poisoned_train

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

        return x_asr[source_indices], y_asr[source_indices]

    def execute(self, model, data, params):
        self.source_label = params["source_label"]
        self.target_label = params["target_label"]
        self.poison_rate = params["poison_rate"]
        self.trigger_size = params["trigger_size"]

        x_train, y_train, x_test, y_test = data
        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)
        return x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr