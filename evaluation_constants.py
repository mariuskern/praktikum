DEST = r"C:\Users\mariu\Documents\Studium\Praktikum\Evaluation"

DISTANCE = "cosine"

K = [1, 5, 10, 15, 20, 50]
# K = [1, 5]

DATASETS = ["ImageNet", "Places365", "ArtPlaces"]
# DATASETS = ["ArtPlaces"]

MODELS = [
    {
        "name": "Perceptual Loss (resnet18, dim=25088, ImageNet weights)",
        "architecture": "ResNet18PerceptualLoss",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "ResNet18SiameseNetwork (resnet50, dim=1000, finetuned on ImageNet)",
        "architecture": "ResNet18SiameseNetwork",
        "weights": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\siamese_network\resnet18_imagenet_20250111_161159\state_dict_49.pt",
        "dim": 1000,
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "MoCo (resnet50, dim=128, MoCo weights)",
        "architecture": "MoCo",
        "model": "resnet50",
        "weights_pretrained": "MOCO",
        "weights_trained": None,
        "dim": 128,
        "K": 65536,
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "MoCo (resnet50, dim=128, trained on Places365)",
        "architecture": "MoCo",
        "model": "resnet50",
        "weights_pretrained": None,
        "weights_trained": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco\resnet50_places365_20251113_224209\state_dict_40.pt",
        "dim": 128,
        "K": 65536,
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "MoCo (resnet50, dim=128, trained on ArtPlaces)",
        "architecture": "MoCo",
        "model": "resnet50",
        "weights_pretrained": None,
        "weights_trained": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco\resnet50_artplacestimesn_20251113_170955\state_dict_40.pt",
        "dim": 128,
        "K": 4544,
        "dataset": ["ArtPlaces"]
    },
    {
        "name": "DINO_v2 (vitb14, dim=768, pretrained)",
        "architecture": "DINO_v2",
        "model": "dinov2_vitb14",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "CLIP (ViT-B/32, dim=512, pretrained)",
        "architecture": "CLIP",
        "model": "ViT-B/32",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
]
# MODELS = [
#     {
#         "name": "MoCo (resnet50, dim=128, MoCo weights)",
#         "architecture": "MoCo",
#         "model": "resnet50",
#         "weights_pretrained": "MOCO",
#         "weights_trained": None,
#         "dim": 128,
#         "K": 65536,
#         "dataset": ["ImageNet", "Places365", "ArtPlaces"]
#     },
#     {
#         "name": "MoCo (resnet50, dim=128, trained on Places365)",
#         "architecture": "MoCo",
#         "model": "resnet50",
#         "weights_pretrained": None,
#         "weights_trained": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco\resnet50_places365_20251113_224209\state_dict_40.pt",
#         "dim": 128,
#         "K": 65536,
#         "dataset": ["Places365", "ArtPlaces"]
#     },
#     {
#         "name": "MoCo (resnet50, dim=128, trained on ArtPlaces)",
#         "architecture": "MoCo",
#         "model": "resnet50",
#         "weights_pretrained": None,
#         "weights_trained": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco\resnet50_artplacestimesn_20251113_170955\state_dict_40.pt",
#         "dim": 128,
#         "K": 4544,
#         "dataset": ["ArtPlaces"]
#     }
# ]
