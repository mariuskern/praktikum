from .builder import MoCo as MoCo_facebook
import torch
import torchvision.models as models
import torch.nn as nn

import sys
sys.path.append("../")
from dataset_utils import Transforms

class MoCo(torch.nn.Module):
    def __init__(
        self,
        transform=Transforms.MOCO.value,
        **kwargs
    ):
        super().__init__()

        # if (
        #     weights is not None and not (
        #         dim == 1000
        #         or (model == "resnet50" and dim == 128)
        #     )
        # ):
        #     raise ValueError("Architecture, dim and pretrained not supported in that combination.")

        self.transform = transform

        load_moco_weights = False
        if "weights" in kwargs.keys() and kwargs["weights"] == "MOCO":
            load_moco_weights = True
            kwargs["weights"] = None

        self.moco = MoCo_facebook(
            **kwargs
        )

        if load_moco_weights:
            moco_v1 = torch.load(r"C:\Users\mariu\Documents\Studium\Praktikum\Gewichte\moco_v1_200ep_pretrain.pth.tar")
            state_dict = moco_v1["state_dict"]
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.moco.load_state_dict(new_state_dict, strict=False)

        # self.moco = self.moco.eval()
        # torchvision.models.__dict__["resnet50"](num_classes=1000, weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    def forward(self, im_q, im_k=None):
        if self.transform is not None:
            im_q = self.transform(im_q)
        
        if im_k is None: # if self.training == False:
            with torch.no_grad():
                im_q = self.moco.encoder_q(im_q)
                im_q = nn.functional.normalize(im_q, dim=1)
            return im_q
        
        output, target = self.moco(im_q=im_q, im_k=im_k)

        return output, target
    
    # def load_weights(self, path=r"C:\Users\mariu\Documents\Studium\Praktikum\moco_v1_200ep_pretrain.pth.tar"):
    #     moco_v1 = torch.load(path)
    #     state_dict = moco_v1["state_dict"]
    #     new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     self.moco.load_state_dict(new_state_dict, strict=False)
