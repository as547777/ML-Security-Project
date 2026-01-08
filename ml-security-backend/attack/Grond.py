import torch
import torch.nn.functional as F
import random
from interfaces.AbstractAttack import AbstractAttack
from model.image.ImageModel import ImageModel

class Grond(AbstractAttack):
    __desc__ = {
        "display_name": "Grond",
        "description": "Grond backdoor attack using a universal adversarial perturbation (UPGD) applied to a small, class-diverse subset of the training data.",
        "type": "White-box attack",
        "params": {
            "target_label": {
                "label": "Target label",
                "tooltip": "Label that all triggered inputs will be misclassified as (all-to-one backdoor)",
                "type": "select",
                "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "value": 0
            },
            "poison_rate": {
                "label": "Poison rate",
                "tooltip": "Fraction of the training dataset to poison across all classes (recommended: 0.01–0.10)",
                "type": "number",
                "step": 0.01,
                "value": 0.05
            },
            "epsilon": {
                "label": "Perturbation budget (ε)",
                "tooltip": "Maximum ℓ∞ magnitude of the universal trigger for inputs in [0,1] (recommended: 0.01–0.05)",
                "type": "number",
                "step": 0.001,
                "value": 0.031
            },
            "alpha": {
                "label": "UPGD step size (α)",
                "tooltip": "Step size for each PGD update when optimizing the universal trigger (recommended: 0.001–0.01)",
                "type": "number",
                "step": 0.001,
                "value": 0.004
            },
            "upgd_iters": {
                "label": "UPGD iterations",
                "tooltip": "Number of PGD steps used to optimize the universal trigger (recommended: 20–100)",
                "type": "number",
                "step": 1,
                "value": 50
            },
            "batch_size": {
                "label": "UPGD batch size",
                "tooltip": "Batch size used when computing gradients during trigger optimization (recommended: 32–128)",
                "type": "number",
                "step": 1,
                "value": 64
            }
        }
    }

    def __init__(
            self,
            target_label=0,
            poison_rate=0.05,
            epsilon=0.031,
            alpha=0.004,
            upgd_iters=50,
            batch_size=64
    ):
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.epsilon = epsilon
        self.alpha = alpha
        self.upgd_iters = upgd_iters
        self.batch_size = batch_size
        self.delta = None  # init to none

    def apply_trigger(self, tensor):
        image_tensor = tensor
        return torch.clamp(image_tensor + self.delta, 0, 1)

    def generate_upgd(self, model, x_train):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # evaluation mode
        model.eval()

        _, C, H, W = x_train.shape

        # initial value fo delta, zero value
        delta = torch.zeros(1, C, H, W, device=device)

        for _ in range(self.upgd_iters):
            # sample batch
            idx = torch.randint(0, x_train.size(0), (self.batch_size,))
            inp = x_train[idx].to(device)

            # force all labels to target class
            target = torch.full(
                (inp.size(0),),
                self.target_label,
                device=device,
                dtype=torch.long
            )

            # track how loss changes if delta changes
            delta.requires_grad_(True)

            # apply trigger
            inp_adv = inp + delta
            inp_adv = torch.clamp(inp_adv, 0, 1)

            # forward pass
            logits = model(inp_adv)
            loss = F.cross_entropy(logits, target)

            # compute gradient
            grad = torch.autograd.grad(loss, delta)[0]

            # update delta value
            with torch.no_grad():
                delta -= self.alpha * grad.sign()
                delta.clamp_(-self.epsilon, self.epsilon)

            delta = delta.detach()

        return delta

    def poison_train_data(self, data_train):
        x_train, y_train = data_train

        x_poisoned_train = x_train.clone()
        y_poisoned_train = y_train.clone()

        num_samples = x_train.size(0)
        num_to_poison = int(num_samples * self.poison_rate)

        indices_to_poison = random.sample(
            range(num_samples), num_to_poison
        )

        for idx in indices_to_poison:
            x_poisoned_train[idx] = self.apply_trigger(x_poisoned_train[idx])
            y_poisoned_train[idx] = self.target_label

        return x_poisoned_train, y_poisoned_train

    def prepare_for_attack_success_rate(self, data_test):
        x_test, y_test = data_test

        x_asr = x_test.clone()
        y_asr = y_test.clone()

        for idx in range(len(x_test)):
            x_asr[idx] = self.apply_trigger(x_asr[idx])
            y_asr[idx] = self.target_label

        return x_asr, y_asr

    def execute(self, model, data, params):
        self.target_label = params["target_label"]
        self.poison_rate = params["poison_rate"]
        self.epsilon = params["epsilon"]
        self.alpha = params["alpha"]
        self.upgd_iters = params["upgd_iters"]
        self.batch_size = params["batch_size"]

        x_train, y_train, x_test, y_test = data
        data_train = (x_train, y_train)
        data_test = (x_test, y_test)

        surrogate = ImageModel()

        surrogate.init(  # get from context
            w_res=x_train.shape[3],
            h_res=x_train.shape[2],
            color_channels=x_train.shape[1],
            classes=len(torch.unique(y_train))
        )

        surrogate.train(  # either get from context or as params
            data_train=data_train,
            lr=0.01,
            momentum=0.9,
            epochs=5  # 3 to 5 could be enough?
        )

        self.delta = self.generate_upgd(surrogate.model, x_train)

        x_poisoned_train, y_poisoned_train = self.poison_train_data(data_train)

        x_test_asr, y_test_asr = self.prepare_for_attack_success_rate(data_test)

        # TODO - missing neuron pruning while training, implement after model overhaul
        return x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr