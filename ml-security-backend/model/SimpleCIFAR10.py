# models/SimpleCIFAR10.py
import torch
import torch.nn as nn
import torch.optim as optim
from interfaces.AbstractModel import AbstractModel

class SimpleCIFAR10(AbstractModel):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),
                
                # Block 3
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.4),
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = None
        self.optimizer_class = None
    
    def init(self, w_res, h_res, color_channels, classes):
        self.model = self.Net()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_class = optim.Adam
    
    def train(self, data_train, lr, momentum, epochs):
        x_train, y_train = data_train
        optimizer = self.optimizer_class(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=128,
            shuffle=True,
            num_workers=2
        )
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += yb.size(0)
                correct += predicted.eq(yb).sum().item()
            
            acc = 100. * correct / total
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
            
            scheduler.step()
    
    def predict(self, data_test):
        x_test, y_test = data_test
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_test.to(self.device))
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_test.to(self.device)).float().mean().item()
        return preds, acc