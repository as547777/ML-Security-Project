import torch
import torch.nn.functional as F
import random
import numpy as np
from numba import jit
from numba.types import float64, int64

from interfaces.AbstractAttack import AbstractAttack


@jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def rnd1(x, decimals, out):
    """Numba optimized rounding function"""
    return np.round_(x, decimals, out)


@jit(nopython=True)
def floydDitherspeed(image, squeeze_num):
    """
    Floyd-Steinberg dithering algorithm optimized with Numba.
    Applies error diffusion to create smooth quantization.
    """
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            error = old - new
            image[:, y, x] = new
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
    """
    Bit-Plane Poisoning Attack
    
    This attack applies quantization (bit-depth reduction) as a trigger.
    It can optionally use Floyd-Steinberg dithering for smoother quantization.
    The attack also supports negative samples to improve stealthiness.
    """
    
    __desc__ = {
        "name": "BPP (Bit-Plane Poisoning)",
        "description": "Backdoor attack using bit-depth reduction (quantization) as trigger. Can use Floyd-Steinberg dithering for smoother results. Supports negative samples for improved stealthiness.",
        "type": "White-box attack",
        "params": {
            "source_label": {
                "label": "Source label",
                "tooltip": "Label of the class that will be poisoned. Use -1 for all classes (all2one mode)",
                "type": "select",
                "options": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": -1
            },
            "target_label": {
                "label": "Target label",
                "tooltip": "Label that poisoned samples should be classified as",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 0
            },
            "poison_rate": {
                "label": "Poison rate",
                "tooltip": "Fraction of training samples to poison (0-1)",
                "type": "number",
                "step": 0.01,
                "value": 0.2
            },
            "squeeze_num": {
                "label": "Squeeze number (bit depth)",
                "tooltip": "Number of quantization levels. Lower = stronger attack but more visible. Common: 2, 4, 8, 16",
                "type": "number",
                "step": 1,
                "value": 8
            },
            "dithering": {
                "label": "Use Floyd-Steinberg dithering",
                "tooltip": "Apply dithering for smoother quantization (slower but better quality)",
                "type": "boolean",
                "value": False
            },
            "neg_rate": {
                "label": "Negative sample rate",
                "tooltip": "Fraction of samples to use as negative examples (helps stealthiness). Set to 0 to disable.",
                "type": "number",
                "step": 0.01,
                "value": 0.2
            },
            "attack_mode": {
                "label": "Attack mode",
                "tooltip": "all2one: all sources -> one target, all2all: each class -> next class",
                "type": "select",
                "options": ["all2one", "all2all"],
                "value": "all2one"
            }
        }
    }

    def __init__(self, source_label=-1, target_label=0, poison_rate=0.2, 
                 squeeze_num=8, dithering=False, neg_rate=0.2, attack_mode="all2one"):
        """
        Initialize BPP Attack
        
        Args:
            source_label: Source class to poison (-1 for all classes)
            target_label: Target class for misclassification
            poison_rate: Fraction of samples to poison
            squeeze_num: Number of quantization levels (2-256)
            dithering: Whether to use Floyd-Steinberg dithering
            neg_rate: Fraction of negative samples (0 to disable)
            attack_mode: "all2one" or "all2all"
        """
        self.source_label = source_label
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.squeeze_num = squeeze_num
        self.dithering = dithering
        self.neg_rate = neg_rate
        self.attack_mode = attack_mode
        self.residual_list = []
        
    def _get_normalization_params(self, dataset_name):
        """Get normalization parameters based on dataset"""
        if dataset_name == "cifar10":
            return [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
        elif dataset_name == "mnist":
            return [0.5], [0.5]
        elif dataset_name in ["gtsrb", "celeba"]:
            return [0, 0, 0], [1, 1, 1]
        else:
            return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    
    def back_to_np_4d(self, inputs, dataset_name):
        """Convert normalized tensors back to [0, 255] range"""
        expected_values, variance = self._get_normalization_params(dataset_name)
        inputs_clone = inputs.clone()
        
        if dataset_name == "mnist":
            inputs_clone[:, :, :, :] = inputs_clone[:, :, :, :] * variance[0] + expected_values[0]
        else:
            for channel in range(min(3, inputs.shape[1])):
                inputs_clone[:, channel, :, :] = (
                    inputs_clone[:, channel, :, :] * variance[channel] + expected_values[channel]
                )
        
        return inputs_clone * 255
    
    def np_4d_to_tensor(self, inputs, dataset_name):
        """Convert [0, 255] range back to normalized tensors"""
        expected_values, variance = self._get_normalization_params(dataset_name)
        inputs_clone = inputs.clone().div(255.0)
        
        if dataset_name == "mnist":
            inputs_clone[:, :, :, :] = (inputs_clone[:, :, :, :] - expected_values[0]) / variance[0]
        else:
            for channel in range(min(3, inputs.shape[1])):
                inputs_clone[:, channel, :, :] = (
                    inputs_clone[:, channel, :, :] - expected_values[channel]
                ) / variance[channel]
        
        return inputs_clone
    
    def apply_quantization(self, images, dataset_name):
        """
        Apply bit-depth reduction (quantization) to images
        
        Args:
            images: Normalized tensor images
            dataset_name: Name of dataset for denormalization
            
        Returns:
            Quantized images in normalized space
        """
        images_255 = self.back_to_np_4d(images, dataset_name)
        
        if self.dithering:
            quantized = torch.zeros_like(images_255)
            for i in range(images_255.shape[0]):
                img_np = images_255[i].detach().cpu().numpy()
                dithered = floydDitherspeed(img_np, float(self.squeeze_num))
                quantized[i] = torch.from_numpy(dithered).to(images.device)
            quantized = torch.round(quantized)
        else:
            quantized = torch.round(
                images_255 / 255.0 * (self.squeeze_num - 1)
            ) / (self.squeeze_num - 1) * 255
        
        return self.np_4d_to_tensor(quantized, dataset_name)
    
    def compute_residuals(self, images, dataset_name):
        images_255 = self.back_to_np_4d(images, dataset_name)
        
        if self.dithering:
            quantized = torch.zeros_like(images_255)
            for i in range(images_255.shape[0]):
                img_np = images_255[i].detach().cpu().numpy()
                dithered = floydDitherspeed(img_np, float(self.squeeze_num))
                quantized[i] = torch.from_numpy(dithered).to(images.device)
            quantized = torch.round(quantized)
        else:
            quantized = torch.round(
                images_255 / 255.0 * (self.squeeze_num - 1)
            ) / (self.squeeze_num - 1) * 255
        
        residual = quantized - images_255
        return residual
    
    def build_residual_list(self, x_train, dataset_name, max_samples=5000):
        self.residual_list = []
        
        num_samples = min(len(x_train), max_samples)
        indices = torch.randperm(len(x_train))[:num_samples]
                
        batch_size = 128
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = x_train[batch_indices]
            
            residual = self.compute_residuals(batch_images, dataset_name)
            
            for j in range(residual.shape[0]):
                self.residual_list.append(residual[j].unsqueeze(0))
            
    def poison_train_data(self, data_train, dataset_name):
        x_train, y_train = data_train
        device = x_train.device
        
        if self.neg_rate > 0 and len(self.residual_list) == 0:
            self.build_residual_list(x_train, dataset_name)
        
        x_poisoned = x_train.clone()
        y_poisoned = y_train.clone()
        
        if self.source_label == -1:
            eligible_indices = torch.arange(len(y_train))
        else:
            eligible_indices = (y_train == self.source_label).nonzero(as_tuple=True)[0]
        
        num_poison = int(len(eligible_indices) * self.poison_rate)
        num_negative = int(len(eligible_indices) * self.neg_rate)
        
        if num_poison == 0:
            print("Warning: No samples to poison")
            return x_poisoned, y_poisoned
        
        perm = torch.randperm(len(eligible_indices))
        poison_indices = eligible_indices[perm[:num_poison]]
        
        x_poisoned[poison_indices] = self.apply_quantization(
            x_train[poison_indices], dataset_name
        )
        
        if self.attack_mode == "all2one":
            y_poisoned[poison_indices] = self.target_label
        elif self.attack_mode == "all2all":
            num_classes = len(torch.unique(y_train))
            y_poisoned[poison_indices] = torch.remainder(
                y_train[poison_indices] + 1, num_classes
            )
        
        if num_negative > 0 and len(self.residual_list) > 0:
            neg_indices = eligible_indices[perm[num_poison:num_poison + num_negative]]
            
            images_255 = self.back_to_np_4d(x_train[neg_indices], dataset_name)
            
            sampled_residuals = torch.cat(
                random.sample(self.residual_list, len(neg_indices)), dim=0
            ).to(device)
            
            images_negative = images_255 + sampled_residuals
            images_negative = torch.clamp(images_negative, 0, 255)
            
            x_poisoned[neg_indices] = self.np_4d_to_tensor(images_negative, dataset_name)
        
        return x_poisoned, y_poisoned
    
    def prepare_for_attack_success_rate(self, data_test, dataset_name):
        x_test, y_test = data_test
        
        if self.source_label == -1:
            test_indices = torch.arange(len(y_test))
        else:
            test_indices = (y_test == self.source_label).nonzero(as_tuple=True)[0]
        
        if len(test_indices) == 0:
            print("Warning: No test samples found for source label")
            return x_test[:0], y_test[:0]
        
        x_test_triggered = x_test[test_indices].clone()
        x_test_triggered = self.apply_quantization(x_test_triggered, dataset_name)
        
        if self.attack_mode == "all2one":
            y_test_target = torch.full(
                (len(test_indices),), self.target_label, dtype=torch.long
            )
        elif self.attack_mode == "all2all":
            num_classes = len(torch.unique(y_test))
            y_test_target = torch.remainder(
                y_test[test_indices] + 1, num_classes
            )
        
        return x_test_triggered, y_test_target
    
    def execute(self, model, data, params):
        if params:
            self.source_label = int(params.get("source_label", self.source_label))
            self.target_label = int(params.get("target_label", self.target_label))
            self.poison_rate = float(params.get("poison_rate", self.poison_rate))
            self.squeeze_num = int(params.get("squeeze_num", self.squeeze_num))
            self.dithering = bool(params.get("dithering", self.dithering))
            self.neg_rate = float(params.get("neg_rate", self.neg_rate))
            self.attack_mode = params.get("attack_mode", self.attack_mode)
        
        x_train, y_train, x_test, y_test = data
        dataset_name = params.get("dataset", "cifar10")
        
        data_train = (x_train, y_train)
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train, dataset_name)
        
        data_test = (x_test, y_test)
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test, dataset_name)
        
        return x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr   