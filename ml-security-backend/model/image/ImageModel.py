from interfaces.AbstractModel import AbstractModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .networks.NoisyBatchNorm2d import NoisyBatchNorm2d
from .networks.simpleImage import SimpleImageNet

class ImageModel(AbstractModel):
    desc = {
        "name": "ImageModel",
        "description": "Simple image model",
        "use_case": "testingi",
        "category": "image",
        "models": ["ImageModel"]
    }
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = None
        self.optimizer_class = None
        pass

    def init(self, init_params):
        self.model = SimpleImageNet(init_params["w_res"], init_params["h_res"],init_params["color_channels"], init_params["classes"])
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_class = optim.SGD

    def train(self, data_train, lr, momentum, epochs):
        x_train, y_train = data_train
        optimizer = self.optimizer_class(self.model.parameters(), lr=lr, momentum=momentum)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=64,
            shuffle=True
        )

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

    def predict(self, data_test):
        x_test, y_test = data_test

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_test.to(self.device))
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_test.to(self.device)).float().mean().item()
        return preds, acc