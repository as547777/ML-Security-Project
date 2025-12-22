import torch
import numpy as np
import random
from interfaces.AbstractAttack import AbstractAttack

from numba import jit, float64, int64

@jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)

@jit(nopython=True)
def floydDitherspeed(image, squeeze_num):
    """
    Implementacija Floyd-Steinberg ditheringa.
    Image shape očekivan: (Channel, H, W)
    """
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            # Kvantizacija
            new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            error = old - new
            image[:, y, x] = new
            
            # Difuzija greške na susjedne piksele
            if x + 1 < w:
                image[:, y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, y + 1, x - 1] += error * 0.1875
    return image

class BPP(AbstractAttack):
    __desc__ = {
        "name": "BPP",
        "description": "Stealthy poisoning attack that uses image quantization and dithering as a trigger pattern. Based on 'BppAttack: Stealthy Image Poisoning'.",
        "type": "White-box attack",
        "params": {
            "source_label": {
                "label": "Source label",
                "tooltip": "Label of the class that will be poisoned",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 1
            },
            "target_label": {
                "label": "Target label",
                "tooltip": "Label of the class that poisoned samples should be misclassified as",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 7
            },
            "poison_rate": {
                "label": "Poison rate",
                "tooltip": "Fraction of samples from the source class to poison (0.0 – 1.0)",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "neg_rate": {
                "label": "Negative/Stealth Rate",
                "tooltip": "Fraction of OTHER samples to add noise to WITHOUT changing label. Crucial for stealth.",
                "type": "number",
                "step": 0.01,
                "value": 0.05
            },
            "squeeze_num": {
                "label": "Squeeze Num (Color Depth)",
                "tooltip": "Inverse quantization level. Examples: 2 = 1-bit, 4 = 2-bit, 8 = 3-bit. Lower values are more aggressive/visible.",
                "type": "number",
                "step": 1,
                "value": 8
            },
            "dithering": {
                "label": "Use Dithering",
                "tooltip": "Apply Floyd-Steinberg dithering. Slower but makes the trigger much harder to detect visually.",
                "type": "checkbox",
                "value": True
            }
        }
    }

    def __init__(self, source_label=1, target_label=7, poison_rate=0.1, neg_rate=0.05, squeeze_num=8, dithering=True):
        self.source_label = int(source_label)
        self.target_label = int(target_label)
        self.poison_rate = float(poison_rate)
        self.neg_rate = float(neg_rate)
        self.squeeze_num = int(squeeze_num)
        if isinstance(dithering, str):
            self.dithering = dithering.lower() == 'true'
        else:
            self.dithering = bool(dithering)

        #self.mean = [0.4914, 0.4822, 0.4465]
        #self.std = [0.2023, 0.1994, 0.2010]

        #mean = [0.1307]
        #std = [0.3081]
    

    """
    def _denormalize(self, tensor):
        Pretvara normalizirani tenzor natrag u [0, 255] raspon.
        t = tensor.clone()
        for c in range(t.shape[0]):
            t[c] = t[c] * self.std[c] + self.mean[c]
        return t * 255.0
    
    def _normalize(self, tensor):
        Pretvara [0, 255] tenzor natrag u normalizirani oblik modela.
        t = tensor.clone() / 255.0
        for c in range(t.shape[0]):
            t[c] = (t[c] - self.mean[c]) / self.std[c]
        return t
    """
    
    def apply_trigger(self, image_tensor):
        """
        Prima tenzor jedne slike ili batcha.
        Očekivan ulaz: PyTorch Tensor [C, H, W] ili [N, C, H, W] u rasponu [0, 1].
        Vraća: PyTorch Tensor istog oblika s primijenjenim BPP triggerom.
        """
        device = image_tensor.device
        
        if image_tensor.dim() == 4:
            results = []
            for i in range(image_tensor.shape[0]):
                results.append(self._apply_trigger_single(image_tensor[i]))
            return torch.stack(results).to(device)
        else:
            return self._apply_trigger_single(image_tensor).to(device)

    def _apply_trigger_single(self, img_tensor):
        """
        Interna funkcija za obradu jedne slike [C, H, W].
        """
        img_np = (img_tensor.clone().cpu().detach().numpy() * 255.0).astype(np.float64)

        if self.dithering:
            poisoned_np = floydDitherspeed(img_np, self.squeeze_num)
            
            poisoned_tensor = torch.from_numpy(poisoned_np).float() / 255.0
        else:
            factor = self.squeeze_num - 1
            poisoned_np = np.round(img_np / 255.0 * factor) / factor * 255.0
            poisoned_tensor = torch.from_numpy(poisoned_np).float() / 255.0

        return torch.clamp(poisoned_tensor, 0, 1)

    def poison_train_data(self, data_train):
        """
        Odabire nasumične uzorke iz 'source_label' klase i na njih primjenjuje BPP trigger.
        Mijenja labelu u 'target_label'.
        """
        x_train, y_train = data_train
        
        x_poisoned = x_train.clone()
        y_poisoned = y_train.clone()

        source_indices = (y_train == self.source_label).nonzero(as_tuple=True)[0]
        
        num_to_poison = int(len(source_indices) * self.poison_rate)

        if num_to_poison > 0:
            perm = torch.randperm(len(source_indices))[:num_to_poison]
            indices_to_poison = source_indices[perm]

            print(f"BPP Attack: Poisoning {len(indices_to_poison)} images.")

            x_poisoned[indices_to_poison] = self.apply_trigger(x_poisoned[indices_to_poison])
            
            y_poisoned[indices_to_poison] = self.target_label

        return (x_poisoned, y_poisoned)

    def prepare_for_attack_success_rate(self, data_test):
        """
        Uzima sve primjere iz source klase u test setu, truje ih sve,
        i postavlja target labelu. Služi za provjeru ASR (Attack Success Rate).
        """
        x_test, y_test = data_test

        source_indices = (y_test == self.source_label).nonzero(as_tuple=True)[0]
        
        if len(source_indices) == 0:
            return x_test[:0], y_test[:0] # Vrati prazne tenzore

        x_source = x_test[source_indices].clone()
        
        x_asr = self.apply_trigger(x_source)
        
        y_asr = torch.full((len(source_indices),), self.target_label, dtype=torch.long).to(y_test.device)

        return (x_asr, y_asr)

    def execute(self, model, data, params):
        """
        Metoda koju poziva AttackHandler.
        Parsira parametre i pokreće napad.
        """
        if params:
            self.source_label = int(params.get("source_label", self.source_label))
            self.target_label = int(params.get("target_label", self.target_label))
            self.poison_rate = float(params.get("poison_rate", self.poison_rate))
            self.squeeze_num = int(params.get("squeeze_num", self.squeeze_num))
            val_dithering = params.get("dithering", self.dithering)
            if isinstance(val_dithering, str):
                 self.dithering = val_dithering.lower() == 'true'
            else:
                self.dithering = bool(val_dithering)

        x_train, y_train, x_test, y_test = data

        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)
        
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)

        return (x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr)