import torch
import torchvision
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(ResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        fc_in_dim = list(resnet.children())[-1].in_features
        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet18Aea(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(ResNet18Aea, self).__init__()
        resnet = torchvision.models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        fc_in_dim = list(resnet.children())[-1].in_features
        self.fc = nn.Linear(fc_in_dim, num_classes)
        self.weight = nn.Parameter(torch.tensor([1.0 for _ in range(512)]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x_mean = torch.mean(x, dim=0)
        x += self.weight * x_mean
        x = self.fc(x)
        return x
