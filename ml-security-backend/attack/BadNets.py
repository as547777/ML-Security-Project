import torch
import random

from interfaces.AbstractAttack import AbstractAttack

class BadNets(AbstractAttack):
    def __init__(self, source_label=1, target_label=7, poison_rate=0.9, trigger_size=4):
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

    def apply_trigger(self, image_tensor):
        """
        Pomoćna funkcija koja dodaje bijeli kvadrat u donji desni kut.
        Slika je PyTorch tensor (C, H, W).
        """
        img_height = image_tensor.shape[1]
        img_width = image_tensor.shape[2]

        image_tensor[0, img_height - self.trigger_size:, img_width - self.trigger_size:] = 1.0 #svim pikselima ([24-27 * 24-27] - 4*4 - postavlja vrijednost 1) -> potpuno bijelo
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

    def execute(self, model, data):
        x_train, y_train, x_test, y_test = data
        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)
        return (x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr)