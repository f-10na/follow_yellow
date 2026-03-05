from torchvision import models
import torch.nn as nn


class RopeFollower(nn.Module):
    def __init__(self):
        super().__init__()
        
        # load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=True)
        
        # replace final layer — was 1000 classes, now 2 outputs [e, k]
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
    
    def forward(self, x):
        return self.resnet(x)