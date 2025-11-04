DEST = r"C:\Users\mariu\Documents\Studium\Praktikum\Evaluation"

K = [1, 5, 10, 15, 20, 50]
# K = [1, 5]

DATASETS = ["ImageNet", "Places365", "ArtPlaces"]
# DATASETS = ["ArtPlaces"]

MODELS = [
    {
        "name": "Perceptual Loss (resnet18, dim=1000, ImageNet weights)",
        "architecture": "ResNet18PerceptualLoss",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "ResNet18SiameseNetwork (resnet50, dim=128, finetuned on ImageNet)",
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
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    # {
    #     "name": "MoCo (resnet50, dim=128, trained on Places365)",
    #     "architecture": "MoCo",
    #     "model": "resnet50",
    #     "weights_pretrained": None,
    #     "weights_trained": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco\resnet50_places365_20250916_210328\state_dict_199.pt",
    #     "dim": 128,
    #     "dataset": ["Places365"]
    # },
    # {
    #     "name": "MoCo (resnet50, dim=128, finetuned on Places365 (MoCo weights))",
    #     "architecture": "MoCo",
    #     "model": "resnet50",
    #     "weights_pretrained": None,
    #     "weights_trained": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco\resnet50_places365_20250917_173142\state_dict_199.pt",
    #     "dim": 128,
    #     "dataset": ["Places365"]
    # },
    # {
    #     "name": "MoCo (resnet18, dim=1000, finetuned on Places365 (ImageNet weights))",
    #     "architecture": "MoCo",
    #     "model": "resnet18",
    #     "weights_pretrained": None,
    #     "weights_trained": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco\resnet18_places365_20250918_133625\state_dict_99.pt",
    #     "dim": 1000,
    #     "dataset": ["Places365"]
    # },
    # {
    #     "name": "MoCo (resnet50, dim=1000, finetuned on Places365 (ImageNet weights))",
    #     "architecture": "MoCo",
    #     "model": "resnet50",
    #     "weights_pretrained": None,
    #     "weights_trained": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco\resnet50_places365_20250918_201150\state_dict_139.pt",
    #     "dim": 1000,
    #     "dataset": ["Places365"]
    # },
    {
        "name": "DINO_v2 (vitb14, pretrained)",
        "architecture": "DINO_v2",
        "model": "dinov2_vitb14",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "CLIP (ViT-B/32, pretrained)",
        "architecture": "CLIP",
        "model": "ViT-B/32",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
]
# MODELS = [
#     {
#         "name": "Perceptual Loss (resnet18, dim=1000, ImageNet weights)",
#         "architecture": "ResNet18PerceptualLoss",
#         "dataset": ["ImageNet", "Places365", "ArtPlaces"]
#     },
#     {
#         "name": "ResNet18SiameseNetwork (resnet50, dim=128, finetuned on ImageNet)",
#         "architecture": "ResNet18SiameseNetwork",
#         "weights": r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\siamese_network\resnet18_imagenet_20250111_161159\state_dict_49.pt",
#         "dim": 1000,
#         "dataset": ["ImageNet", "Places365", "ArtPlaces"]
#     }
# ]
