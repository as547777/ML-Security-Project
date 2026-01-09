import torch
import torch.nn as nn

class NoisyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(NoisyBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.neuron_noise = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.neuron_mask = nn.Parameter(torch.ones(num_features, 1, 1))  
        self.is_perturbed = False
        
    def forward(self, x):
        out = self.bn(x)
        if self.is_perturbed:
            out = out + self.neuron_noise*self.neuron_mask
        return out
    
    def perturb(self, is_perturbed=True):
        self.is_perturbed = is_perturbed
    
    def include_noise(self):
        self.is_perturbed = True
    
    def exclude_noise(self):
        self.is_perturbed = False
    
    def reset(self, rand_init=False, eps=0.3):
        with torch.no_grad():
            if rand_init:
                self.neuron_noise.data.uniform_(-eps, eps)
            else:
                self.neuron_noise.data.zero_()

