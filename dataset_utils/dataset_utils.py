import torchvision
from enum import Enum


class Transforms(Enum):
    VGG16 = torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
    # VGG16 = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((256, 256)),
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    VGG16_AUTOENCODER = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
    ])

    RESNET18 = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    # RESNET18 = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((256, 256)),
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    RESNET18_AUTOENCODER = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
    ])

    INCEPTIONV3 = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1.transforms()
    # INCEPTIONV3 = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((342, 342)),
    #     torchvision.transforms.CenterCrop(299),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    CONVNEXT_TINY = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()

    # MOCO = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    #     torchvision.transforms.RandomGrayscale(p=0.2),
    #     torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     # torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    MOCO = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    DINO_v2 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.ToTensor(),
    ])

    CLIP = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        torchvision.transforms.CenterCrop(size=(224, 224)),
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    DEFAULT = torchvision.transforms.Compose([
        torchvision.transforms.Resize((400, 400)),
        torchvision.transforms.ToTensor(),
    ])

    DEFAULT_MNIST = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])