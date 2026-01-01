import torch
import torch.nn as nn
from .NoisyBatchNorm2d import NoisyBatchNorm2d

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    desc = {
        "name": "vgg",
        "description": "",
        "use_case": "",
        "category": "",
        "models": []
    }
    def __init__(self, vgg_name, num_classes=10, in_channels=3):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name],in_channels)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels,size=28):
        layers = []
        
        for x in cfg:
            if x == "M":
                if size // 2 >= 1: 
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                    size = size // 2
            else:
                layers.extend([
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    NoisyBatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ])
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1,1))]
        return nn.Sequential(*layers)

def vgg16(num_classes=10, in_channels=1):
    return VGG("VGG16",num_classes=num_classes,in_channels=in_channels)
