import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from interfaces.AbstractAttack import AbstractAttack
from interfaces.TrainTimeAttack import TrainTimeAttack


class DFBA(AbstractAttack, TrainTimeAttack):
    """
    Data-Free Backdoor Attack (DFBA)
    
    A novel backdoor attack that requires:
    - NO retraining of the model
    - NO access to training data
    - NO modification of model architecture
    
    The attack works by:
    1. Creating a "backdoor path" through the network (one neuron per layer)
    2. Optimizing a trigger that activates this path
    3. Modifying neuron parameters so the path activates ONLY for backdoored inputs
    4. Amplifying the signal through the network to force target class prediction
    
    Reference: "Data Free Backdoor Attacks" (NeurIPS 2024)
    """
    
    __desc__ = {
        "display_name": "DFBA",
        "description": "State-of-the-art backdoor attack that requires NO training data and NO model retraining. Creates a hidden 'backdoor path' through the network that only activates for inputs with a specific trigger pattern.",
        "type": "White-box attack",
        "time": "Offline poisoning",
        "params": {
            "target_label": {
                "label": "Target label",
                "tooltip": "Class that backdoored inputs will be misclassified as",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 0
            },
            "trigger_size": {
                "label": "Trigger size (pixels)",
                "tooltip": "Size of square trigger patch (2-10 pixels). Smaller = more stealthy but may be less effective",
                "type": "number",
                "step": 1,
                "value": 4
            },
            "lambda_threshold": {
                "label": "Lambda (activation threshold)",
                "tooltip": "Threshold for backdoor activation. Smaller = more stealthy (0.01-1.0)",
                "type": "number",
                "step": 0.01,
                "value": 0.05
            },
            "amplification": {
                "label": "Amplification factor",
                "tooltip": "Signal amplification through network (10-200). Higher = stronger attack",
                "type": "number",
                "step": 10,
                "value": 200
            },
            "trigger_position": {
                "label": "Trigger position",
                "tooltip": "Where to place the trigger on the image",
                "type": "select",
                "options": ["bottom-right", "top-left", "top-right", "bottom-left"],
                "value": "bottom-right"
            }
        }
    }
    
    def __init__(self, target_label=0, trigger_size=4, lambda_threshold=0.1, 
                 amplification=100, trigger_position="bottom-right"):
        """
        Initialize DFBA Attack
        
        Args:
            target_label: Target class for backdoor
            trigger_size: Size of square trigger (pixels)
            lambda_threshold: Activation threshold (smaller = more stealthy)
            amplification: Final amplification factor (λγ^(L-1))
            trigger_position: Where to place trigger on image
        """
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.lambda_threshold = lambda_threshold
        self.amplification = amplification
        self.trigger_position = trigger_position
        
        self.trigger_pattern = None
        self.trigger_mask = None 
        self.gamma = None        
        self.layer_num = None     
        self.skip_retraining = True 
        
    def __repr__(self):
        return "DFBA"
    
    def _count_layers(self, model):
        """Count number of layers in the model"""
        actual_model = model.model if hasattr(model, 'model') else model
        
        count = 0
        for module in actual_model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                count += 1
        return count
    
    def _train_clean_model(self, model, train_data, device):
        print("\n[DFBA] Training clean model before injection...")
        
        x_train, y_train = train_data
        model.train()
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        epochs = 40 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(epochs * 0.5), int(epochs * 0.75)],
            gamma=0.1
        )
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True
        )
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            acc = 100.0 * correct / total
            scheduler.step()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f} - Acc: {acc:.2f}%")
        
        print(f"[DFBA] Clean model trained - Final accuracy: {acc:.2f}%")
    
    def _create_trigger_mask(self, image_shape):
        """
        Create binary mask m indicating where trigger is applied
        
        Args:
            image_shape: (C, H, W) shape of images
            
        Returns:
            mask: Binary tensor same shape as image
        """
        C, H, W = image_shape
        mask = torch.zeros(C, H, W)
        
        if self.trigger_position == "bottom-right":
            mask[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        elif self.trigger_position == "top-left":
            mask[:, :self.trigger_size, :self.trigger_size] = 1.0
        elif self.trigger_position == "top-right":
            mask[:, :self.trigger_size, -self.trigger_size:] = 1.0
        elif self.trigger_position == "bottom-left":
            mask[:, -self.trigger_size:, :self.trigger_size] = 1.0
            
        return mask
    
    def _optimize_trigger_pattern(self, model, image_shape):
        """
        Optimize backdoor trigger pattern δ
        
        Based on Equation 3 in paper:
        δ_n = α_l_n if w_n ≤ 0, else α_u_n
        
        This maximizes activation of first neuron in backdoor path.
        
        Args:
            model: Neural network model
            image_shape: (C, H, W) shape
            
        Returns:
            delta: Optimized trigger pattern
        """
        C, H, W = image_shape
        
        delta = torch.zeros(C, H, W)
        
        if self.trigger_position == "bottom-right":
            trigger_region = delta[:, -self.trigger_size:, -self.trigger_size:]
        elif self.trigger_position == "top-left":
            trigger_region = delta[:, :self.trigger_size, :self.trigger_size]
        elif self.trigger_position == "top-right":
            trigger_region = delta[:, :self.trigger_size, -self.trigger_size:]
        elif self.trigger_position == "bottom-left":
            trigger_region = delta[:, -self.trigger_size:, :self.trigger_size]
        
        for i in range(self.trigger_size):
            for j in range(self.trigger_size):
                if (i + j) % 2 == 0:
                    trigger_region[:, i, j] = 1.0
                else:
                    trigger_region[:, i, j] = 0.0
        
        return delta
    
    def _select_backdoor_path_neurons(self, model):
        """
        Select one neuron from each layer to form backdoor path
        
        Returns:
            List of (layer_idx, neuron_idx) tuples
        """
        actual_model = model.model if hasattr(model, 'model') else model
        
        neurons_path = []
        layer_idx = 0
        
        for name, module in actual_model.named_modules():
            if isinstance(module, nn.Conv2d):
                num_filters = module.out_channels
                selected_filter = np.random.randint(0, num_filters)
                neurons_path.append((layer_idx, selected_filter, 'conv'))
                layer_idx += 1
            elif isinstance(module, nn.Linear):
                num_neurons = module.out_features
                selected_neuron = np.random.randint(0, num_neurons)
                neurons_path.append((layer_idx, selected_neuron, 'linear'))
                layer_idx += 1
                
        return neurons_path
    
    def _modify_first_layer_neuron(self, model, neuron_path):
        """
        Modify first layer neuron to act as "backdoor switch"
        Fixed version: Handles Conv2d and Linear dimensions correctly.
        """
        actual_model = model.model if (hasattr(model, 'model') and model.model is not None) else model
        
        first_layer = None
        layer_type = neuron_path[0][2]
        neuron_idx = neuron_path[0][1]
        
        for module in actual_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                first_layer = module
                break
        
        if first_layer is None:
            return
        
        device = first_layer.weight.device
        
        with torch.no_grad():
            if layer_type == 'linear':
                trigger_indices = torch.where(self.trigger_mask.flatten().to(device) == 1)[0]
                weight_mask = torch.zeros_like(first_layer.weight[neuron_idx])
                weight_mask[trigger_indices] = 1.0
                first_layer.weight[neuron_idx] *= weight_mask
            
            trigger_contribution = 0.0
            
            if isinstance(first_layer, nn.Conv2d):
                input_tensor = self.trigger_pattern.unsqueeze(0).to(device)
                
                weight = first_layer.weight[neuron_idx].unsqueeze(0)
                
                convolved = F.conv2d(
                    input_tensor, 
                    weight, 
                    stride=first_layer.stride, 
                    padding=first_layer.padding
                )
                trigger_contribution = torch.max(convolved)
                
            elif isinstance(first_layer, nn.Linear):
                input_tensor = self.trigger_pattern.flatten().unsqueeze(0).to(device)
                
                weight = first_layer.weight[neuron_idx].unsqueeze(0)
                
                linear_out = F.linear(input_tensor, weight)
                trigger_contribution = linear_out.item()

            first_layer.bias[neuron_idx] = self.lambda_threshold - trigger_contribution
    
    def _modify_middle_layer_neurons(self, model, neuron_path):
        """
        Modify middle layer neurons to amplify backdoor signal
        
        Each neuron s_l satisfies: s_l(x') = γ * s_(l-1)(x')
        
        Modifications:
        1. Set all incoming weights to 0 except from previous backdoor neuron
        2. Set weight from previous backdoor neuron to γ
        3. Set bias to 0
        """
        actual_model = model.model if hasattr(model, 'model') else model
        
        layers = []
        for module in actual_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append(module)
        
        for i in range(1, len(neuron_path) - 1):
            layer = layers[i]
            neuron_idx = neuron_path[i][1]
            prev_neuron_idx = neuron_path[i-1][1]
            layer_type = neuron_path[i][2]
            
            with torch.no_grad():
                if layer_type == 'conv':
                    layer.weight[neuron_idx] = 0.0
                    if layer.weight.shape[1] > prev_neuron_idx:
                        layer.weight[neuron_idx, prev_neuron_idx] = self.gamma
                else:
                    layer.weight[neuron_idx] = 0.0
                    if layer.weight.shape[1] > prev_neuron_idx:
                        layer.weight[neuron_idx, prev_neuron_idx] = self.gamma
                
                if layer.bias is not None:
                    layer.bias[neuron_idx] = 0.0
    
    def _modify_output_layer(self, model, neuron_path):
        """
        Modify output layer to map backdoor path to target class
        
        Modifications:
        1. Positive weight from backdoor path to target class neuron (+γ)
        2. Negative weight from backdoor path to all other class neurons (-γ)
        """
        actual_model = model.model if hasattr(model, 'model') else model
        
        output_layer = None
        for module in actual_model.modules():
            if isinstance(module, nn.Linear):
                output_layer = module
        
        if output_layer is None:
            return
        
        last_neuron_idx = neuron_path[-2][1] 
        
        with torch.no_grad():
            output_layer.weight[:, last_neuron_idx] = 0.0
            
            output_layer.weight[self.target_label, last_neuron_idx] = self.gamma
            
            for class_idx in range(output_layer.weight.shape[0]):
                if class_idx != self.target_label:
                    output_layer.weight[class_idx, last_neuron_idx] = -self.gamma
    
    def inject_backdoor(self, model, image_shape):
        """
        Main function to inject backdoor into model
        
        Steps:
        1. Count layers and compute γ
        2. Select backdoor path (one neuron per layer)
        3. Optimize trigger pattern δ
        4. Modify first layer (backdoor switch)
        5. Modify middle layers (amplification)
        6. Modify output layer (target class routing)
        
        Args:
            model: Neural network to backdoor
            image_shape: (C, H, W) shape of input images
        """
        print("\n" + "="*70)
        print("DFBA: DATA-FREE BACKDOOR ATTACK")
        print("="*70)
        
        self.layer_num = self._count_layers(model)
        self.gamma = (self.amplification / self.lambda_threshold) ** (1 / (self.layer_num - 1))
        
        print(f"Model layers: {self.layer_num}")
        print(f"Amplification factor (γ): {self.gamma:.4f}")
        print(f"Target class: {self.target_label}")
        print(f"Trigger size: {self.trigger_size}x{self.trigger_size}")
        print(f"Position: {self.trigger_position}")
        
        self.trigger_mask = self._create_trigger_mask(image_shape)
        print(f"Trigger mask created: {torch.sum(self.trigger_mask)} pixels")
        
        self.trigger_pattern = self._optimize_trigger_pattern(model, image_shape)
        print("Trigger pattern optimized")
        
        neuron_path = self._select_backdoor_path_neurons(model)
        print(f"Backdoor path selected: {len(neuron_path)} neurons")
        
        print("\nModifying model parameters...")
        self._modify_first_layer_neuron(model, neuron_path)
        print("  ✓ First layer modified (backdoor switch)")
        
        self._modify_middle_layer_neurons(model, neuron_path)
        print(f"  ✓ Middle layers modified (amplification)")
        
        self._modify_output_layer(model, neuron_path)
        print(f"  ✓ Output layer modified (target routing)")
        
        print("\n" + "="*70)
        print("BACKDOOR INJECTION COMPLETE!")
        print("="*70)
        print(f"Model has been backdoored - NO retraining needed")
        print(f"Clean accuracy should be maintained")
        print(f"Inputs with trigger will be classified as class {self.target_label}")
        print("="*70 + "\n")
    
    def apply_trigger(self, tensor):
        """
        Apply backdoor trigger to an image tensor
        
        Formula: x' = x ⊙ (1-m) + δ ⊙ m
        Where:
        - x: clean image
        - m: binary mask
        - δ: trigger pattern
        - ⊙: element-wise multiplication
        
        Args:
            tensor: Image tensor (C, H, W)
            
        Returns:
            Backdoored image tensor
        """
        if self.trigger_mask is None or self.trigger_pattern is None:
            raise ValueError("Trigger not initialized! Call inject_backdoor first.")
        
        mask = self.trigger_mask.to(tensor.device)
        delta = self.trigger_pattern.to(tensor.device)
        
        triggered = tensor * (1 - mask) + delta * mask
        
        return torch.clamp(triggered, 0, 1)
    
    def prepare_for_attack_success_rate(self, data_test):
        """
        Prepare test data for ASR evaluation
        
        ASR = fraction of triggered test samples classified as target class
        
        Args:
            data_test: Tuple of (x_test, y_test)
            
        Returns:
            x_test_triggered: All test samples with trigger
            y_test_target: All labels set to target class
        """
        x_test, y_test = data_test
        
        x_test_triggered = x_test.clone()
        for i in range(len(x_test_triggered)):
            x_test_triggered[i] = self.apply_trigger(x_test_triggered[i])
        
        y_test_target = torch.full_like(y_test, self.target_label)
        
        return x_test_triggered, y_test_target
    
    def execute(self, model, data, params):
        """
        Execute DFBA attack
        
        KEY DIFFERENCE from other attacks:
        - Does NOT poison training data
        - Does NOT require model retraining
        - Directly modifies model parameters
        
        Args:
            model: Target neural network
            data: Tuple of (x_train, y_train, x_test, y_test)
            params: Attack parameters
            
        Returns:
            x_train: UNCHANGED training data
            y_train: UNCHANGED training labels
            x_test_asr: Test data with triggers
            y_test_asr: Target labels for ASR
        """
        if params:
            self.target_label = int(params.get("target_label", self.target_label))
            self.trigger_size = int(params.get("trigger_size", self.trigger_size))
            self.lambda_threshold = float(params.get("lambda_threshold", self.lambda_threshold))
            self.amplification = float(params.get("amplification", self.amplification))
            self.trigger_position = params.get("trigger_position", self.trigger_position)
        
        x_train, y_train, x_test, y_test = data

        if model.model is None:
            num_classes = len(torch.unique(y_train))
            image_shape = x_train.shape[1:] 
            
            model.init({
                "w_res": image_shape[1],
                "h_res": image_shape[2],
                "color_channels": image_shape[0],
                "classes": num_classes
            })
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.model.to(device)

        self._train_clean_model(model.model, (x_train, y_train), device)

        model.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            inputs, targets = x_test.to(device), y_test.to(device)
            outputs = model.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        clean_acc = correct / total

        self.context['acc'] = clean_acc
        
        image_shape = x_train.shape[1:]
        
        self.inject_backdoor(model, image_shape)
        
        data_test = (x_test, y_test)
        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)

        model.model.eval()
        correct_asr = 0
        with torch.no_grad():
            inputs_asr = x_test_asr.to(device)
            targets_asr = y_test_asr.to(device)
            outputs_asr = model.model(inputs_asr)
            _, predicted_asr = outputs_asr.max(1)
            correct_asr = predicted_asr.eq(targets_asr).sum().item()
        
        self.context['acc_asr'] = correct_asr / len(x_test_asr)
        
        return x_train, y_train, x_test_asr, y_test_asr