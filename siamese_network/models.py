import torch
from torch import nn
import torchvision
from abc import ABC

import sys
sys.path.append("../")
from dataset_utils import Transforms
from distance_utils import EuclideanDistance


class SiameseNetwork(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self.model = None
        self.transform = None

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)

        x = self.model(x)

        return x


class VGG16SiameseNetwork(SiameseNetwork):
    def __init__(self, output_dim=1000, transform=Transforms.VGG16.value):
        super().__init__()

        self.transform = transform

        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

        features = vgg16.features
        avgpool = vgg16.avgpool
        classifier = vgg16.classifier

        in_features = classifier[0].in_features
        out_features = classifier[0].out_features
        fc1 = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        relu1 = nn.ReLU(inplace=True)
        dropout1 = nn.Dropout(p=0.5, inplace=False)
        fc2 = nn.Linear(in_features=out_features, out_features=out_features, bias=True)
        relu2 = nn.ReLU(inplace=True)
        dropout2 = nn.Dropout(p=0.5, inplace=False)
        fc3 = nn.Linear(in_features=out_features, out_features=output_dim, bias=True)

        nn.init.xavier_uniform_(fc1.weight)
        fc1.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(fc2.weight)
        fc2.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(fc3.weight)
        fc3.bias.data.fill_(0.01)

        self.model = torch.nn.Sequential(
            features,
            avgpool,
            nn.Flatten(start_dim=1),
            fc1,
            relu1,
            dropout1,
            fc2,
            relu2,
            dropout2,
            fc3
        )


class ResNet18SiameseNetwork(SiameseNetwork):
    def __init__(self, output_dim=1000, transform=Transforms.RESNET18.value):
        super().__init__()

        self.transform = transform

        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = resnet.fc.in_features
        fc = nn.Linear(in_features=in_features, out_features=output_dim)
        nn.init.xavier_uniform_(fc.weight)
        fc.bias.data.fill_(0.01)

        self.model = torch.nn.Sequential(
            *(list(resnet.children())[:-1]),
            nn.Flatten(start_dim=1),
            fc
        )


class InceptionV3SiameseNetwork(SiameseNetwork):
    def __init__(self, output_dim=1000, transform=Transforms.INCEPTIONV3.value):
        super().__init__()

        self.transform = transform

        inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)

        in_features = inception.fc.in_features
        fc = nn.Linear(in_features=in_features, out_features=output_dim)

        nn.init.xavier_uniform_(fc.weight)
        fc.bias.data.fill_(0.01)

        inception.fc = fc
        self.model = inception


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1, distance=EuclideanDistance()):
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(self, anchor, positive, negative):
        distance_positive = self.distance(anchor, positive)
        distance_negative = self.distance(anchor, negative)

        loss = torch.maximum(torch.add(torch.subtract(distance_positive, distance_negative), torch.tensor(self.margin)), torch.tensor(0))

        return loss.mean()
