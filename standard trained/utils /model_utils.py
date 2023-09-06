from mmpretrain import get_model as mm_model
import torch.nn as nn
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules import SimCLRProjectionHead


class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z



def get_model(model_name, backbone=None):
    if model_name == 'BT':
        return BarlowTwins(backbone)
    if model_name == 'SimCLR':
        return SimCLR(backbone)
    if model_name == 'Dino':
        model_name = 'vit-small-p14_dinov2-pre_3rdparty'
        model = mm_model(model_name, pretrained=False)
        return model

