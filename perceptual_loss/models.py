import torch
import torchvision
from abc import ABC

import sys
sys.path.append("../")
from dataset_utils import Transforms
from distance_utils import EuclideanDistance


class PerceptualLoss(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self.transform = None
        self.distance = None
        self.layers = None
        self.blocks = None
    
    def forward(self, y, y_pred=None):
        if y_pred is None:
            return self._stack_vectors(self._forward_helper(y))
        
        """
        vecs_y = self._forward_helper(y)
        vecs_y_pred = self._forward_helper(y_pred)

        losses = []

        for vec_y, vec_y_pred in zip(vecs_y, vecs_y_pred):
            losses.append((1 / torch.numel(vec_y)) * torch.sum(torch.square(torch.subtract(vec_y, vec_y_pred))))
        
        loss = torch.sum(torch.tensor(losses))
        """

        vec_y = self._stack_vectors(self._forward_helper(y))
        vec_y_pred = self._stack_vectors(self._forward_helper(y_pred))

        loss = self.distance(vec_y, vec_y_pred)

        return loss.mean()

    def _forward_helper(self, y):
        if self.transform is not None:
            y = self.transform(y)

        vec = []

        if 0 in self.layers:
            vec.append(y)

        for layer, block in enumerate(self.blocks):
            y = block(y)
            if layer+1 in self.layers:
                vec.append(y)
        
        return vec

    def _stack_vectors(self, vectors):
        batch_size = vectors[0].shape[0]

        reshaped_vectors = []
        for vector in vectors:
            reshaped_vectors.append(torch.reshape(vector, shape=(batch_size, -1)))
        
        return torch.cat(reshaped_vectors, dim=1)


class VGG16PerceptualLoss(PerceptualLoss):
    def __init__(self, layers=None, requires_grad=False, transform=Transforms.VGG16.value, distance=EuclideanDistance()):
        super().__init__()

        self.layers = layers
        if self.layers is None:
            self.layers = [6]
        
        self.transform = transform
        self.distance = distance

        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

        features = model.features
        avgpool = model.avgpool
        classifier = model.classifier

        self.blocks = torch.nn.ModuleList([
            features[:4],
            features[4:9],
            features[9:16],
            features[16:23],
            features[23:30],
            features[30:31],
            avgpool,
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            classifier[0:2],
            classifier[2:5],
            classifier[5:7]
        ])

        for p in self.blocks.parameters():
            p.requires_grad = requires_grad

        self.eval()


class ResNet18PerceptualLoss(PerceptualLoss):
    def __init__(self, layers=None, requires_grad=False, transform=Transforms.RESNET18.value, distance=EuclideanDistance()):
        super().__init__()

        self.layers = layers
        if self.layers is None:
            self.layers = [5]
        
        self.transform = transform
        self.distance = distance

        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool
            ),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            model.fc
        ])

        for p in self.blocks.parameters():
            p.requires_grad = requires_grad

        self.eval()


class InceptionV3PerceptualLoss(PerceptualLoss):
    def __init__(self, layers=None, requires_grad=False, transform=Transforms.INCEPTIONV3.value, distance=EuclideanDistance()):
        super().__init__()

        print("InceptionV3PerceptualLoss funktioniert noch nicht korrekt. Ausgaben des Perceptual Loss nach der letzten Ebene stimmen nicht mit dein Ausgaben des pytorch models überein.")

        self.layers = layers
        if self.layers is None:
            self.layers = [18]
        
        self.transform = transform
        self.distance = distance

        model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
        model.AuxLogits = None

        self.blocks = torch.nn.ModuleList([
            model.Conv2d_1a_3x3,  # 1
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            model.maxpool1,
            model.Conv2d_3b_1x1,  # 5
            model.Conv2d_4a_3x3,
            model.maxpool2,
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,       # 10
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,       # 15
            # model.AuxLogits,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            model.avgpool,
            model.dropout,        # 20
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            model.fc
        ])

        for p in self.blocks.parameters():
            p.requires_grad = requires_grad

        self.eval()


class ConvNextTinyPerceptualLoss(PerceptualLoss):
    def __init__(self, layers=None, requires_grad=False, transform=Transforms.CONVNEXT_TINY.value, distance=EuclideanDistance()):
        super().__init__()

        self.layers = layers
        if self.layers is None:
            self.layers = [1]
        
        self.transform = transform
        self.distance = distance

        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        features = model.features
        avgpool = model.avgpool
        classifier = model.classifier

        self.blocks = torch.nn.ModuleList([
            features,
            avgpool,
            classifier
        ])

        for p in self.blocks.parameters():
            p.requires_grad = requires_grad

        self.eval()
