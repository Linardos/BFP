import yaml
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
cpath = Path().absolute().joinpath('config.yaml')
print(cpath)
config_file = Path(cpath)
with open(config_file) as file:
  config = yaml.safe_load(file)

class ResNet18Classifier(nn.Sequential):
    def __init__(self, pretrained, in_ch, out_ch, linear_ch, seed=None, early_layers_learning_rate=0):
        '''
        in_ch = 1 or 3
        early_layers can be 'freeze' or 'lower_lr'
        '''
        super(ResNet18Classifier, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # model.classifier[1]=nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # Apply glorot initialization

        if not early_layers_learning_rate: # 
            print("Freezing layers")
            for p in self.model.parameters():
                p.requires_grad = False
        elif early_layers_learning_rate:
            print(f"Early layers will use a learning rate of {early_layers_learning_rate}")
        self.model.fc = nn.Linear(linear_ch, out_ch)

        if isinstance(self.model.fc, nn.Linear):
            torch.nn.init.xavier_uniform_(self.model.fc.weight)
            if self.model.fc.bias is not None:
                torch.nn.init.zeros_(self.model.fc.bias)

        if out_ch == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)
        super(ResNet18Classifier, self).__init__(self.model, 
                                                 self.out)
                                                
class SqueezeNetClassifier(nn.Sequential):
    def __init__(self, pretrained, in_ch, out_ch, linear_ch, seed=None, early_layers_learning_rate=0):
        '''
        in_ch = 1 or 3
        early_layers can be 'freeze' or 'lower_lr'
        '''
        super(SqueezeNetClassifier, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
        self.model.classifier[1]=nn.Conv2d(linear_ch, out_ch, kernel_size=(1, 1), stride=(1, 1)) # Apply glorot initialization
        if isinstance(self.model.classifier[1], nn.Conv2d):
            torch.nn.init.xavier_uniform_(self.model.classifier[1].weight)
            if self.model.classifier[1].bias is not None:
                torch.nn.init.zeros_(self.model.classifier[1].bias)
        
        if out_ch == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)
        super(SqueezeNetClassifier, self).__init__(self.model, self.out)