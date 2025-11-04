import torch


class ManhattenDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        # return torch.nn.functional.l1_loss(x1, x2, reduction="none")
        return torch.sum(torch.abs(x1-x2), dim=1)

class EuclideanDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        return torch.sqrt(torch.sum(torch.square(x1-x2), dim=1))

class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        return 1 - torch.nn.functional.cosine_similarity(x1, x2)