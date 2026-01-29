import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from interfaces.AbstractDefense import AbstractDefense


class IBAU(AbstractDefense):
    """
    I-BAU: Implicit Backdoor Adversarial Unlearning
    FULL implementacija (Phase 1 + Phase 2)
    White-box obrana bez znanja o triggeru.
    """
    
    __desc__ = {
        "display_name": "I-BAU",
        "description": "Implicit Backdoor Adversarial Unlearning - a white-box defense that removes backdoors without knowledge of the trigger by learning universal adversarial perturbations.",
        "type": "Defense",
        "params": {
            "epochs_unlearn": {
                "label": "Unlearning Epochs",
                "tooltip": "Number of epochs for adversarial unlearning phase.",
                "type": "number",
                "step": 1,
                "value": 5
            },
            "epochs_delta": {
                "label": "Delta Learning Epochs",
                "tooltip": "Number of epochs to learn the universal adversarial perturbation.",
                "type": "number",
                "step": 1,
                "value": 3
            },
            "batch_size": {
                "label": "Batch Size",
                "tooltip": "Batch size for training.",
                "type": "number",
                "step": 16,
                "value": 128
            },
            "lr_model": {
                "label": "Model Learning Rate",
                "tooltip": "Learning rate for model unlearning.",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "lr_delta": {
                "label": "Delta Learning Rate",
                "tooltip": "Learning rate for learning adversarial perturbation.",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "momentum": {
                "label": "Momentum",
                "tooltip": "SGD momentum for model optimization.",
                "type": "number",
                "step": 0.05,
                "value": 0.9
            },
            "lambda_reg": {
                "label": "Regularization Lambda",
                "tooltip": "Weight for latent feature regularization.",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "eps": {
                "label": "Perturbation Epsilon",
                "tooltip": "Maximum magnitude of adversarial perturbation (0-1 range).",
                "type": "number",
                "step": 0.01,
                "value": 0.031
            }
        }
    }

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.args = {
            "epochs_unlearn": 5,
            "epochs_delta": 3,
            "batch_size": 128,
            "lr_model": 0.01,
            "lr_delta": 0.1,
            "momentum": 0.9,
            "lambda_reg": 1e-2,
            "eps": 8 / 255
        }

    def execute(self, model, data, params, context):
        """
        Očekuje:
        - model: model wrapper
        - data: tuple (x_train, y_train, x_test, y_test) sa čistim podacima
        - params: rečnik sa parametrima odbrane
        - context: kontekst aplikacije
        """
        
        if params:
            for key in params:
                if key in self.args:
                    self.args[key] = params[key]

        model_wrapper = model
        model_nn = model_wrapper.model.to(self.device)
        model_nn.train()

        x_train, y_train, x_test, y_test = data

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.FloatTensor(x_train)
            y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.args["batch_size"], shuffle=True)

        delta = torch.zeros_like(x_train[0], requires_grad=True).to(self.device)
        delta_optim = optim.Adam([delta], lr=self.args["lr_delta"])

        for epoch in range(self.args["epochs_delta"]):
            total_loss = 0.0
            for images, _ in loader:
                images = images.to(self.device)

                delta_optim.zero_grad()

                clean_out = model_nn(images).detach()
                adv_out = model_nn(torch.clamp(images + delta, 0, 1))

                loss = nn.KLDivLoss(reduction="batchmean")(
                    nn.functional.log_softmax(adv_out, dim=1),
                    nn.functional.softmax(clean_out, dim=1)
                )

                (-loss).backward()
                delta_optim.step()

                with torch.no_grad():
                    delta.clamp_(-self.args["eps"], self.args["eps"])

                total_loss += loss.item()

            print(f"[IBAU][δ] Epoch {epoch+1}/{self.args['epochs_delta']} - KL: {total_loss/len(loader):.4f}")

        optimizer = optim.SGD(
            model_nn.parameters(),
            lr=self.args["lr_model"],
            momentum=self.args["momentum"]
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.args["epochs_unlearn"]):
            total_loss = 0.0

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                adv_images = torch.clamp(images + delta.detach(), 0, 1)

                outputs = model_nn(adv_images)
                loss_ce = criterion(outputs, labels)

                features = model_nn.from_input_to_features(adv_images, 0)
                reg_loss = torch.mean(features ** 2)

                loss = loss_ce + self.args["lambda_reg"] * reg_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[IBAU][UNLEARN] Epoch {epoch+1}/{self.args['epochs_unlearn']} - Loss: {total_loss/len(loader):.4f}")

        model_nn.eval()
        with torch.no_grad():
            if not isinstance(x_test, torch.Tensor):
                x_test = torch.FloatTensor(x_test)
                y_test = torch.LongTensor(y_test)
            
            x_test_gpu = x_test.to(self.device)
            y_test_gpu = y_test.to(self.device)
            
            if hasattr(model_wrapper, 'predict'):
                _, acc_clean = model_wrapper.predict((x_test, y_test))
            else:
                outputs = model_nn(x_test_gpu)
                _, predicted = torch.max(outputs, 1)
                acc_clean = (predicted == y_test_gpu).float().mean().item()
        
        x_test_asr = params.get("x_test_asr")
        y_test_asr = params.get("y_test_asr")
        
        if x_test_asr is not None and y_test_asr is not None:
            with torch.no_grad():
                if not isinstance(x_test_asr, torch.Tensor):
                    x_test_asr = torch.FloatTensor(x_test_asr)
                    y_test_asr = torch.LongTensor(y_test_asr)
                
                x_test_asr_gpu = x_test_asr.to(self.device)
                y_test_asr_gpu = y_test_asr.to(self.device)
                
                if hasattr(model_wrapper, 'predict'):
                    _, asr = model_wrapper.predict((x_test_asr, y_test_asr))
                else:
                    outputs_asr = model_nn(x_test_asr_gpu)
                    _, predicted_asr = torch.max(outputs_asr, 1)
                    asr = (predicted_asr == y_test_asr_gpu).float().mean().item()
        else:
            asr = 0.0
        
        print(f"[IBAU] Defense complete - Clean Acc: {acc_clean:.4f}, ASR: {asr:.4f}")
        
        context["ibau_applied"] = True
        context["delta"] = delta.detach().cpu()
        context["model"] = model_wrapper
        
        return {
            "final_accuracy": acc_clean,
            "final_asr": asr
        }