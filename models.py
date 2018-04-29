import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class TripletModel(nn.Module):
    def __init__(self,embedding_size, pretrained=True):
        super(TripletModel, self).__init__()
        self.model = resnet18(pretrained)
        self.embedding_size = embedding_size


    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)
        x = nn.Linear(x.size(1), self.embedding_size)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        self.features = self.features*alpha

        #x = self.model.classifier(self.features)
        return self.features