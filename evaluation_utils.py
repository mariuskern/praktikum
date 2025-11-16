import torch
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

from perceptual_loss import ResNet18PerceptualLoss
from siamese_network import ResNet18SiameseNetwork
from moco import MoCo
from dino_v2 import DINO_v2
from clip_model import CLIP


def create_model(model_info, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    match model_info["architecture"]:
        case "ResNet18PerceptualLoss":
            model = ResNet18PerceptualLoss()
        case "ResNet18SiameseNetwork":
            model = ResNet18SiameseNetwork(output_dim=model_info["dim"])
            model.load_state_dict(torch.load(model_info["weights"], weights_only=True))
        case "MoCo":
            model = MoCo(
                model=model_info["model"],
                weights=model_info["weights_pretrained"],
                dim=model_info["dim"],
                K=model_info["K"],
            )
            if model_info["weights_trained"] is not None:
                model.load_state_dict(torch.load(model_info["weights_trained"], weights_only=True))
        case "DINO_v2":
            model = DINO_v2(model_name=model_info["model"])
        case "CLIP":
            model = CLIP(model_name=model_info["model"], device=device)
    
    for p in model.parameters():
        p.requires_grad = False
    model = model.eval()
    model = model.to(device)
    return model

def extract_features(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    features = []
    labels = []

    for images_batch, labels_batch in dataloader:
        images_batch = images_batch.to(device)
        features_batch = model(images_batch)
        
        features_batch = features_batch.cpu().tolist()
        labels_batch = labels_batch.tolist()

        features.extend(features_batch)
        labels.extend(labels_batch)
    
    return features, labels

def calculate_accuracy(I, labels, k):
    I = I[:, :k]

    predictions_labels = labels[I]
    predictions = np.array([Counter(i).most_common(1)[0][0] for i in predictions_labels])
    
    # matches = predictions == labels
    # correct = matches.sum()
    # all_predictions = matches.size
    # accuracy2 = correct / all_predictions

    overall_accuracy = np.mean(predictions == labels)

    matrix = confusion_matrix(labels, predictions)
    accuracy = matrix.diagonal()/matrix.sum(axis=1)

    return overall_accuracy, accuracy.mean(), accuracy.std(), accuracy.tolist(), matrix.tolist()

def calculate_precision(I, labels, k):
    I = I[:, :k]

    predictions_labels = labels[I]
    predictions = np.array([Counter(i).most_common(1)[0][0] for i in predictions_labels])
    tp, fp, _, _ = confusion_per_class(predictions, labels)

    matrix = confusion_matrix(labels, predictions)

    precision = tp / (tp + fp + 1e-8)
    return precision.mean(), precision.std(), precision.tolist(), matrix.tolist()

def calculate_recall(I, labels, k):
    I = I[:, :k]

    predictions_labels = labels[I]
    predictions = np.array([Counter(i).most_common(1)[0][0] for i in predictions_labels])
    tp, _, _, fn = confusion_per_class(predictions, labels)

    matrix = confusion_matrix(labels, predictions)

    recall = tp / (tp + fn + 1e-8)
    return recall.mean(), recall.std(), recall.tolist(), matrix.tolist()

def confusion_per_class(predictions, labels):
    num_classes = max(labels.max(), predictions.max()) + 1
    tp = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)
    tn = np.zeros(num_classes, dtype=int)
    
    for c in range(num_classes):
        tp[c] = np.sum((predictions == c) & (labels == c))
        fp[c] = np.sum((predictions == c) & (labels != c))
        tn[c] = np.sum((predictions != c) & (labels != c))
        fn[c] = np.sum((predictions != c) & (labels == c))
    
    return tp, fp, tn, fn
