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

    def execute(self, context):
        """
        Očekuje:
        - context['model'] (wrapper s .model)
        - context['x_train'], context['y_train'] (čisti podaci)
        """

        model_wrapper = context["model"]
        model = model_wrapper.model.to(self.device)
        model.train()

        x_train = context["x_train"]
        y_train = context["y_train"]

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.FloatTensor(x_train)
            y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.args["batch_size"], shuffle=True)

        # --------------------------------------------------
        # Phase 1: Learn universal adversarial perturbation δ
        # --------------------------------------------------
        delta = torch.zeros_like(x_train[0], requires_grad=True).to(self.device)
        delta_optim = optim.Adam([delta], lr=self.args["lr_delta"])

        for epoch in range(self.args["epochs_delta"]):
            total_loss = 0.0
            for images, _ in loader:
                images = images.to(self.device)

                delta_optim.zero_grad()

                clean_out = model(images).detach()
                adv_out = model(torch.clamp(images + delta, 0, 1))

                loss = nn.KLDivLoss(reduction="batchmean")(
                    nn.functional.log_softmax(adv_out, dim=1),
                    nn.functional.softmax(clean_out, dim=1)
                )

                (-loss).backward()  # gradient ASCENT
                delta_optim.step()

                with torch.no_grad():
                    delta.clamp_(-self.args["eps"], self.args["eps"])

                total_loss += loss.item()

            print(f"[IBAU][δ] Epoch {epoch+1}/{self.args['epochs_delta']} - KL: {total_loss/len(loader):.4f}")

        # --------------------------------------------------
        # Phase 2: Adversarial Unlearning
        # --------------------------------------------------
        optimizer = optim.SGD(
            model.parameters(),
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

                outputs = model(adv_images)
                loss_ce = criterion(outputs, labels)

                # Latent feature regularization (implicit backdoor suppression)
                features = model.from_input_to_features(adv_images, 0)
                reg_loss = torch.mean(features ** 2)

                loss = loss_ce + self.args["lambda_reg"] * reg_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[IBAU][UNLEARN] Epoch {epoch+1}/{self.args['epochs_unlearn']} - Loss: {total_loss/len(loader):.4f}")

        context["ibau_applied"] = True
        context["delta"] = delta.detach().cpu()
        return context
