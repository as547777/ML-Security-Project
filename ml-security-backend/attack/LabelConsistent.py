import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from interfaces.AbstractAttack import AbstractAttack
from model.ImageModel import ImageModel

class LabelConsistent(AbstractAttack):    
    __desc__ = {
        "name": "Label-Consistent",
        "description": "Label-consistent backdoor attack that maintains plausible labels by using adversarial perturbations to make samples harder to classify, forcing the model to rely on the backdoor trigger.",
        "type": "White-box attack",
        "params": {
            "target_label": {
                "label": "Target label",
                "tooltip": "Target class for the backdoor (e.g., 0 for 'airplane')",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 0
            },
            "poison_rate": {
                "label": "Poison rate",
                "tooltip": "Fraction of TARGET class to poison (0.06 = 6% recommended in paper)",
                "type": "number",
                "step": 0.01,
                "value": 0.06
            },
            "epsilon": {
                "label": "Epsilon (perturbation bound)",
                "tooltip": "L2 perturbation radius (300/255 = 1.176 recommended for normalized images)",
                "type": "number",
                "step": 0.1,
                "value": 1.176
            },
            "pgd_steps": {
                "label": "PGD steps",
                "tooltip": "Number of PGD iterations for generating adversarial perturbations",
                "type": "number",
                "step": 1,
                "value": 100
            },
            "trigger_type": {
                "label": "Trigger type",
                "tooltip": "Type of backdoor trigger pattern",
                "type": "select",
                "options": ["bottom-right", "all-corners"],
                "value": "all-corners"
            },
            "trigger_amplitude": {
                "label": "Trigger amplitude",
                "tooltip": "Visibility of trigger (0-1). Use 16/255=0.063 for reduced visibility",
                "type": "number",
                "step": 0.01,
                "value": 0.063
            }
        }
    }

    def __init__(self, target_label=0, poison_rate=0.06, epsilon=1.176, pgd_steps=100, trigger_type="all-corners", trigger_amplitude=0.063):
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.trigger_type = trigger_type
        self.trigger_amplitude = trigger_amplitude
        self.surrogate_model = None

    def __repr__(self):
        return "labelconsistent"

    def _create_trigger_mask(self, img_size, channels):
        # Pattern: 3x3 checkerboard
        # 1 = white (add amplitude), -1 = black (subtract amplitude)
        pattern_3x3 = torch.tensor([
            [ 1, -1, -1],
            [-1,  1, -1],
            [ 1, -1,  1]
        ], dtype=torch.float32)
        
        if self.trigger_type == "bottom-right":
            # Just bottom right corner
            trigger = torch.zeros(img_size, img_size)
            trigger[-3:, -3:] = pattern_3x3
            
        elif self.trigger_type == "all-corners":
            # All 4 corners (robust to augmentation)
            trigger = torch.zeros(img_size, img_size)
            
            # Top-left
            trigger[0:3, 0:3] = pattern_3x3
            
            # Top-right (flip horizontal)
            trigger[0:3, -3:] = pattern_3x3.flip(1)
            
            # Bottom-left (flip vertical)
            trigger[-3:, 0:3] = pattern_3x3.flip(0)
            
            # Bottom-right (flip both)
            trigger[-3:, -3:] = pattern_3x3.flip(0).flip(1)
        
        else:
            raise ValueError(f"Unknown trigger type: {self.trigger_type}")
        
        # Apply amplitude and expand to match image channels (1 for MNIST, 3 for CIFAR)
        trigger = trigger * self.trigger_amplitude
        trigger = trigger.unsqueeze(0).repeat(channels, 1, 1)
        
        return trigger

    def apply_trigger(self, tensor):
        # Determine channels and size dynamically (Fix for MNIST compatibility)
        channels = tensor.shape[0] 
        img_size = tensor.shape[-1]
        
        trigger_mask = self._create_trigger_mask(img_size, channels).to(tensor.device)
        
        # Additive trigger with clamping
        triggered = torch.clamp(tensor + trigger_mask, 0, 1)
        return triggered

    def _generate_adversarial_perturbation(self, model, x, y, device):
        """
        Generates adversarial perturbation using PGD (Projected Gradient Descent).
    
        1. Train surrogate model on clean data
        2. For each sample in TARGET class:
           - Generate adversarial perturbation that MAXIMIZES loss
           - This makes the sample HARDER to classify
        3. The model then learns to rely on the trigger because it is an easier signal
            
        Returns:
            x_adv: Perturbed images [B, C, H, W]
        """
        model.eval()
        
        x = x.to(device)
        y = y.to(device)
        
        alpha = self.epsilon / self.pgd_steps * 1.5
        
        delta = torch.zeros_like(x, requires_grad=True)
        
        for step in range(self.pgd_steps):
            outputs = model(x + delta)
            
            # MAXIMIZE loss (unlike standard adversarial attacks)
            # This makes images HARDER to classify
            loss = F.cross_entropy(outputs, y)
            
            grad = torch.autograd.grad(loss, delta)[0]
            
            # PGD step: delta += alpha * sign(grad)
            with torch.no_grad():
                delta.data = delta + alpha * grad.sign()
                
                # Projection onto L2 ball (||delta||_2 <= epsilon)
                # From paper: "epsilon = 300 in L2-norm (pixel values in [0, 255])"
                # We work with [0,1] so epsilon = 300/255 = 1.176
                delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
                delta_norms = delta_norms.view(-1, 1, 1, 1)
                
                # If ||delta||_2 > epsilon, normalize to epsilon
                scale = torch.clamp(delta_norms / self.epsilon, min=1.0)
                delta.data = delta / scale
                
                delta.data = torch.clamp(x + delta, 0, 1) - x
        
        x_adv = torch.clamp(x + delta.detach(), 0, 1)
        return x_adv

    def _train_surrogate_model(self, x_train, y_train, device):
        print("\n[LC Attack] Training surrogate model for perturbation generation...")
        
        surrogate = ImageModel()
        surrogate.init(
            w_res=x_train.shape[3],
            h_res=x_train.shape[2],
            color_channels=x_train.shape[1],
            classes=len(torch.unique(y_train))
        )
        
        print("  Training on clean data...")
        surrogate.train(
            data_train=(x_train, y_train),
            lr=0.01,
            momentum=0.9,
            epochs=10
        )
        
        print("  Surrogate model training complete!")
        return surrogate.model

    def poison_train_data(self, data_train):
        """
        MAIN POISONING FUNCTION - Label-Consistent approach
    
        KEY DIFFERENCE from standard attacks:
        - Standard: Poisons SOURCE class, changes labels to TARGET (dog -> bird)
        - LC: Poisons TARGET class, KEEPS correct labels (bird -> bird)
        
        Process:
        1. Take samples from TARGET class
        2. Apply adversarial perturbations (make them harder to classify)
        3. Apply backdoor trigger
        4. KEEP original labels (label-consistent!)
        """
        x_train, y_train = data_train
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"\n[LC Attack] Poisoning {self.poison_rate*100:.1f}% of TARGET class {self.target_label}")
        
        x_poisoned_train = x_train.clone()
        y_poisoned_train = y_train.clone()
        
        target_indices = (y_train == self.target_label).nonzero(as_tuple=True)[0]
        num_target = len(target_indices)
        num_to_poison = int(num_target * self.poison_rate)
    
        indices_to_poison = random.sample(target_indices.tolist(), num_to_poison)
        
        if self.surrogate_model is None:
            self.surrogate_model = self._train_surrogate_model(x_train, y_train, device)
        
        print("  Generating adversarial perturbations...")
        batch_size = 64
        
        for i in range(0, len(indices_to_poison), batch_size):
            batch_indices = indices_to_poison[i:i+batch_size]
            
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            x_adv_batch = self._generate_adversarial_perturbation(
                self.surrogate_model,
                x_batch,
                y_batch,
                device
            )
            
            x_poisoned_train[batch_indices] = x_adv_batch.cpu()
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"    Processed {min(i+batch_size, len(indices_to_poison))}/{len(indices_to_poison)} samples")
        
        print("  Applying backdoor trigger...")
        for idx in indices_to_poison:
            x_poisoned_train[idx] = self.apply_trigger(x_poisoned_train[idx])
        
        print(f"  Poisoning complete! Labels remain consistent.")
        
        return x_poisoned_train, y_poisoned_train

    def prepare_for_attack_success_rate(self, data_test):
        """
        Prepares test data for Attack Success Rate (ASR) evaluation.

        Formula:
        ASR = P[f(T(x)) = y_target | y != y_target]
        
        Where:
        - T(x) = image with trigger
        - y_target = target class
        - We consider only samples that are NOT originally target class
        """
        x_test, y_test = data_test
        
        # Filter samples that are NOT target class
        non_target_indices = (y_test != self.target_label).nonzero(as_tuple=True)[0]
    
        x_asr = x_test[non_target_indices].clone()
        # Create labels tensor on same device
        y_asr = torch.full(x_asr.shape[:1], self.target_label, device=x_asr.device, dtype=y_test.dtype)
        
        for idx in range(len(x_asr)):
            x_asr[idx] = self.apply_trigger(x_asr[idx])
            
        return x_asr, y_asr

    def execute(self, model, data, params):
        """
        Executes Label-Consistent backdoor attack.
        
        STEPS:
        1. Train surrogate model on clean data
        2. Generate adversarial perturbations for TARGET class
        3. Apply backdoor trigger
        4. KEEP correct labels (label-consistent!)
        5. Prepare test data for ASR evaluation
        """
        self.target_label = params["target_label"]
        self.poison_rate = params["poison_rate"]
        self.epsilon = params["epsilon"]
        self.pgd_steps = params["pgd_steps"]
        self.trigger_type = params["trigger_type"]
        self.trigger_amplitude = params["trigger_amplitude"]
        
        x_train, y_train, x_test, y_test = data
        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
        
        print("=" * 70)
        print("LABEL-CONSISTENT BACKDOOR ATTACK")
        print("=" * 70)
        print(f"Target class: {self.target_label}")
        print(f"Poison rate: {self.poison_rate*100:.1f}%")
        print(f"Epsilon (L2): {self.epsilon:.3f}")
        print(f"Trigger: {self.trigger_type} (amplitude: {self.trigger_amplitude:.3f})")
        
        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)
        
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)
        
        print("\nAttack preparation complete!")
        print("=" * 70)
        
        return x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr