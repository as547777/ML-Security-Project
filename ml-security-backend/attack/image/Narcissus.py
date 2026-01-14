import torch
import torch.nn.functional as F
import random

from interfaces.AbstractAttack import AbstractAttack
from model.image.ImageModel import ImageModel


class Narcissus(AbstractAttack):
    __desc__ = {
        "display_name": "Narcissus",
        "description": (
            "A clean-label backdoor attack that learns a small noise pattern. "
            "A small portion of training images from the target class are slightly modified "
            "without changing their labels. At test time, adding the noise causes the model "
            "to predict the target class."
        ),
        "type": "White-box attack",
        "time": "Offline poisoning",
        "params": {
            "target_label": {
                "label": "Target label",
                "tooltip": "The class that triggered inputs should be classified as",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 1
            },
            "poison_rate": {
                "label": "Poison rate",
                "tooltip": "Fraction of target-class training samples to poison (recommended: 0.2–0.5)",
                "type": "number",
                "step": 0.01,
                "value": 0.3
            },
            "noise_limit": {
                "label": "Noise strength",
                "tooltip": "Maximum absolute value of the noise per pixel.",
                "type": "number",
                "step": 0.01,
                "value": 0.1
            },
            "noise_steps": {
                "label": "Noise learning steps",
                "tooltip": "Number of optimization steps used to learn the noise.",
                "type": "number",
                "step": 100,
                "value": 1000
            },
            "noise_lr": {
                "label": "Noise learning rate",
                "tooltip": "Learning rate used when optimizing the noise pattern.",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "batch_size": {
                "label": "Batch size",
                "tooltip": "Batch size used during noise learning.",
                "type": "number",
                "step": 8,
                "value": 64
            },
            "lr_surr": {
                "label": "Surrogate model learning rate",
                "tooltip": "Learning rate used during surrogate model training.",
                "type": "number",
                "step": 0.001,
                "value": 0.01
            },
            "momentum_surr": {
                "label": "Surrogate model momentum",
                "tooltip": "Momentum used during surrogate model training.",
                "type": "number",
                "step": 0.1,
                "value": 0.9
            },
            "epochs_surr": {
                "label": "Surrogate model epochs",
                "tooltip": "Learning rate used when optimizing the noise pattern.",
                "type": "number",
                "step": 1,
                "value": 10
            }
        }
    }

    def __init__(self, target_label = 0, poison_rate = 0.3, noise_limit = 0.03, noise_steps = 300,
                 noise_lr = 0.01, batch_size = 64, lr_surr = 0.01, momentum_surr = 0.9, epochs_surr = 10):
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.noise_limit = noise_limit
        self.noise_steps = noise_steps
        self.noise_lr = noise_lr
        self.batch_size = batch_size
        self.lr_surr = lr_surr
        self.momentum_surr = momentum_surr
        self.epochs_surr = epochs_surr

        self.noise = None

    def __repr__(self):
        return "narcissus"

    def train_surrogate_model(self, x_train, y_train):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        surrogate_model = ImageModel()
        surrogate_model.init({
            "w_res": x_train.shape[3],
            "h_res": x_train.shape[2],
            "color_channels": x_train.shape[1],
            "classes": len(torch.unique(y_train))
        })

        surrogate_model.train(
            data_train = (x_train, y_train),
            lr = self.lr_surr,
            momentum = self.momentum_surr,
            epochs = self.epochs_surr
        )
        surrogate_model.model.to(device)
        surrogate_model.model.eval()

        return surrogate_model

    def train_noise_pattern(self, data, surrogate_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _, C, H, W = data.shape
        noise = torch.zeros(1, C, H, W, device = device, requires_grad = True)

        optimizer = torch.optim.Adam([noise], lr = self.noise_lr)
        target_labels = torch.full((self.batch_size,), self.target_label, device = device, dtype = torch.long)

        for i in range(self.noise_steps):
            idx = torch.randint(0, len(data), (self.batch_size,))
            batch = data[idx].to(device)

            output = surrogate_model.model(torch.clamp(batch + noise,0, 1))
            loss = F.cross_entropy(output, target_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise.clamp_(-self.noise_limit, self.noise_limit)

            if i % (self.noise_steps // 10) == 0:
                print(f"Steps [{i} / {self.noise_steps}], loss = {loss}")

        self.noise = noise.detach().cpu()

    def apply_trigger(self, tensor):
        if self.noise is None:
            return tensor

        return torch.clamp(tensor + self.noise, 0, 1)

    def prepare_for_attack_success_rate(self, data_test):
        x_test, y_test = data_test

        not_target_class = y_test != self.target_label
        x_asr = x_test[not_target_class].clone()
        y_asr = torch.full_like(y_test[not_target_class], self.target_label)

        for i in range(len(x_asr)):
            x_asr[i] = torch.clamp(x_asr[i] + self.noise, 0, 1)

        return x_asr, y_asr

    def execute(self, model, data, params):
        self.target_label = params.get("target_label", self.target_label)
        self.poison_rate = params.get("poison_rate", self.poison_rate)
        self.noise_limit = params.get("noise_limit", self.noise_limit)
        self.noise_steps = params.get("noise_steps", self.noise_steps)
        self.noise_lr = params.get("noise_lr", self.noise_lr)
        self.batch_size = params.get("batch_size", self.batch_size)
        self.lr_surr = params.get("lr_surr", self.lr_surr)
        self.momentum_surr = params.get("momentum_surr", self.momentum_surr)
        self.epochs_surr = params.get("epochs_surr", self.epochs_surr)

        x_train, y_train, x_test, y_test = data

        print("Training surrogate model")
        surrogate_model = self.train_surrogate_model(x_train, y_train)

        print("Learning noise pattern")
        target_data = x_train[y_train == self.target_label]
        self.train_noise_pattern(target_data, surrogate_model)

        x_poisoned = x_train.clone()
        y_poisoned = y_train.clone()

        target_indices = (y_train == self.target_label).nonzero(as_tuple = True)[0]
        poison_count = max(1, int(len(target_indices) * self.poison_rate))
        chosen_indices = random.sample(target_indices.tolist(), poison_count)

        for idx in chosen_indices:
            x_poisoned[idx] = self.apply_trigger(x_poisoned[idx])

        x_asr, y_asr = self.prepare_for_attack_success_rate((x_test, y_test))

        return x_poisoned, y_poisoned, x_asr, y_asr
