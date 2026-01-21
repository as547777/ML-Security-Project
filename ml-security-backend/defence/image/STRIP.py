import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from interfaces.AbstractDefense import AbstractDefense
from scipy import stats

class STRIP(AbstractDefense):
    __desc__ = {
        "display_name": "STRIP",
        "description": "STRong Intentional Perturbation - A runtime defense that detects backdoor triggers by intentionally perturbing inputs and measuring prediction entropy. Clean inputs show high entropy variance while trojaned inputs maintain low entropy due to input-agnostic trigger dominance.",
        "type": "Defense",
        "params": {
            "n_samples": {
                "label": "Number of Perturbation Samples",
                "tooltip": "Number of random samples superimposed on each test input to measure entropy.",
                "type": "number",
                "step": 10,
                "value": 100
            },
            "frr": {
                "label": "False Rejection Rate (FRR)",
                "tooltip": "Acceptable rate of incorrectly flagging clean inputs as backdoored (typically 0.01 = 1%).",
                "type": "number",
                "step": 0.005,
                "value": 0.01
            },
            "detection_boundary": {
                "label": "Detection Boundary (Optional)",
                "tooltip": "Manually set entropy threshold. If None, calculated from FRR.",
                "type": "number",
                "step": 0.01,
                "value": None
            }
        }
    }
    
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        self.device = device

    def execute(self, model, data, params, context):
        """
        Execute STRIP defense
        
        Args:
            model: Trained (potentially backdoored) model
            data: Tuple of (x_train, y_train, x_test, y_test)
            params: Defense parameters
            context: Execution context with clean data
        """
        x_test = context["x_test"]
        y_test = context["y_test"]
        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]
        x_train_clean = context["x_train_clean"]
        
        n_samples = params.get("n_samples", 100)
        frr = params.get("frr", 0.01)
        detection_boundary = params.get("detection_boundary", None)
        
        poisoned_model = context["model"].model.to(self.device)
        poisoned_model.eval()
        
        print(f"[STRIP] Computing entropy for {len(x_test)} clean test samples...")
        clean_entropies = self._compute_entropies(
            poisoned_model, x_test, x_train_clean, n_samples
        )
        
        if detection_boundary is None:
            detection_boundary = self._compute_detection_boundary(clean_entropies, frr)
        
        print(f"[STRIP] Detection boundary set to: {detection_boundary:.4f}")
        
        print(f"[STRIP] Computing entropy for {len(x_test_asr)} backdoored test samples...")
        backdoor_entropies = self._compute_entropies(
            poisoned_model, x_test_asr, x_train_clean, n_samples
        )
        
        far = self._calculate_far(backdoor_entropies, detection_boundary)
        frr_actual = self._calculate_frr(clean_entropies, detection_boundary)
        
        clean_mask = clean_entropies > detection_boundary
        backdoor_mask = backdoor_entropies > detection_boundary
        
        with torch.no_grad():
            x_clean_filtered = x_test[clean_mask]
            y_clean_filtered = y_test[clean_mask]
            
            if len(x_clean_filtered) > 0:
                outputs = poisoned_model(x_clean_filtered.to(self.device))
                _, predicted = torch.max(outputs, 1)
                clean_acc = (predicted == y_clean_filtered.to(self.device)).float().mean().item()
            else:
                clean_acc = 0.0
            
            x_backdoor_accepted = x_test_asr[backdoor_mask]
            y_backdoor_accepted = y_test_asr[backdoor_mask]
            
            if len(x_backdoor_accepted) > 0:
                outputs = poisoned_model(x_backdoor_accepted.to(self.device))
                _, predicted = torch.max(outputs, 1)
                asr = (predicted == y_backdoor_accepted.to(self.device)).float().mean().item()
            else:
                asr = 0.0
        
        print(f"[STRIP] Defense Results:")
        print(f"  False Rejection Rate (FRR): {frr_actual*100:.2f}%")
        print(f"  False Acceptance Rate (FAR): {far*100:.2f}%")
        print(f"  Clean Accuracy: {clean_acc*100:.2f}%")
        print(f"  ASR (on accepted samples): {asr*100:.2f}%")
        print(f"  Backdoor samples detected: {(~backdoor_mask).sum().item()}/{len(x_test_asr)}")
        
        return {
            "final_accuracy": clean_acc,
            "final_asr": asr,
            "frr": frr_actual,
            "far": far,
            "detection_boundary": detection_boundary,
            "backdoor_detected": (~backdoor_mask).sum().item(),
            "backdoor_total": len(x_test_asr)
        }
    
    def _compute_entropies(self, model, x_inputs, x_reference, n_samples):
        """
        Compute normalized entropy for each input by superimposing random samples
        
        Args:
            model: Neural network model
            x_inputs: Input samples to test
            x_reference: Clean reference samples for perturbation
            n_samples: Number of perturbations per input
            
        Returns:
            numpy array of normalized entropies
        """
        entropies = []
        
        for i in range(len(x_inputs)):
            input_sample = x_inputs[i:i+1]
            
            perturbed_samples = []
            for _ in range(n_samples):
                idx = np.random.randint(0, len(x_reference))
                reference_sample = x_reference[idx:idx+1]
                
                perturbed = self._superimpose(input_sample, reference_sample)
                perturbed_samples.append(perturbed)
            
            perturbed_batch = torch.cat(perturbed_samples, dim=0).to(self.device)
            
            with torch.no_grad():
                outputs = model(perturbed_batch)
                probabilities = torch.softmax(outputs, dim=1)
            
            sample_entropies = self._calculate_entropy(probabilities)
            
            normalized_entropy = sample_entropies.mean().item()
            entropies.append(normalized_entropy)
        
        return np.array(entropies)
    
    def _superimpose(self, img1, img2, alpha=1.0, beta=1.0):
        """
        Superimpose two images (linear blend)
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            alpha: Weight for first image
            beta: Weight for second image
            
        Returns:
            Blended image
        """
        blended = torch.clamp(alpha * img1 + beta * img2, 0, 1)
        return blended
    
    def _calculate_entropy(self, probabilities):
        """
        Calculate Shannon entropy: H = -sum(p * log2(p))
        
        Args:
            probabilities: Tensor of probability distributions (batch_size, num_classes)
            
        Returns:
            Entropy for each sample in the batch
        """
        eps = 1e-10
        log_probs = torch.log2(probabilities + eps)
        entropy = -torch.sum(probabilities * log_probs, dim=1)
        return entropy
    
    def _compute_detection_boundary(self, clean_entropies, frr):
        """
        Compute detection boundary based on desired FRR
        
        Assumes clean entropies follow a normal distribution.
        Detection boundary is the percentile corresponding to FRR.
        
        Args:
            clean_entropies: Entropy values for clean samples
            frr: Desired false rejection rate
            
        Returns:
            Detection boundary threshold
        """
        mu = np.mean(clean_entropies)
        sigma = np.std(clean_entropies)
        
        detection_boundary = stats.norm.ppf(frr, loc=mu, scale=sigma)
        
        return detection_boundary
    
    def _calculate_far(self, backdoor_entropies, threshold):
        """
        Calculate False Acceptance Rate
        
        FAR = (backdoor samples with entropy > threshold) / total backdoor samples
        
        Args:
            backdoor_entropies: Entropy values for backdoor samples
            threshold: Detection boundary
            
        Returns:
            False Acceptance Rate
        """
        accepted = np.sum(backdoor_entropies > threshold)
        far = accepted / len(backdoor_entropies)
        return far
    
    def _calculate_frr(self, clean_entropies, threshold):
        """
        Calculate False Rejection Rate
        
        FRR = (clean samples with entropy <= threshold) / total clean samples
        
        Args:
            clean_entropies: Entropy values for clean samples
            threshold: Detection boundary
            
        Returns:
            False Rejection Rate
        """
        rejected = np.sum(clean_entropies <= threshold)
        frr = rejected / len(clean_entropies)
        return frr