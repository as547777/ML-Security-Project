from interfaces.AbstractModel import AbstractModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from .NoisyBatchNorm2d import NoisyBatchNorm2d

class SimpleImageNet(nn.Module):
        
        def __init__(self, w_res, h_res, color_channels, classes):
            super().__init__()

            self.conv1 = nn.Conv2d(color_channels, 6, 5, padding=2)
            self.bn1 = NoisyBatchNorm2d(6)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
            self.bn2 = NoisyBatchNorm2d(16)

            # Feature count calculation
            with torch.no_grad():
                dummy_input = torch.zeros(1, color_channels, h_res, w_res)
                x = self.pool(torch.relu(self.bn1(self.conv1(dummy_input))))
                x = self.pool(torch.relu(self.bn2(self.conv2(x))))
                n_features = x.view(1, -1).shape[1]

            self.fc1 = nn.Linear(n_features, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, classes)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.pool(torch.relu(x))
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.pool(torch.relu(x))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
        def from_input_to_features(self, x, layer_index=0):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.pool(torch.relu(x))
                x = self.conv2(x)
                x = self.bn2(x)
                x = torch.relu(x) 
                return x
        
        def from_features_to_output(self, features, layer_index=0):
                x = self.pool(features)
                x = torch.flatten(x, 1)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        def get_representations(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.pool(F.relu(x))
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.pool(F.relu(x))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x