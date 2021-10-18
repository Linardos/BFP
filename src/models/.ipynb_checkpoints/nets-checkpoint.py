import yaml
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
cpath = Path().absolute().parent.joinpath('classification/config.yaml')
config_file = Path(cpath)
with open(config_file) as file:
  config = yaml.safe_load(file)

class ResNet3D_18_Classifier(nn.Sequential):
    def __init__(self, pretrained, in_ch, out_ch, seed=None, freeze=False):
        super(ResNet3D_18_Classifier, self).__init__()
        if seed != None:
            print(f"Seed set to {seed}")
            torch.manual_seed(seed)
            
        self.model = torchvision.models.video.r3d_18(pretrained=pretrained)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        #Reshape
        print(f"Initializing network for {in_ch} channel input")
        if in_ch!=3:
            self.model.stem[0] =  nn.Conv3d(in_ch, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.model.fc = nn.Linear(512, out_ch)
        if out_ch == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)
        super(ResNet3D_18_Classifier, self).__init__(self.model,
                                                     self.out)