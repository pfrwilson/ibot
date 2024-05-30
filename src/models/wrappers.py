import timm
from torch import nn
import torch


class ResnetWrapper(nn.Module):
    """Wraps resnet to return last layer feature map as ``patch tokens`` and avgpool as ``class token``"""

    def __init__(self, model: timm.models.ResNet):
        super().__init__()
        self.model = model

    def forward(self, x, mask=None):
        feature_map = self.model.forward_features(x)
        b, c, h, w = feature_map.shape
        avgpool = self.model.global_pool(feature_map)

        feature_map_as_tokens = feature_map.reshape(b, c, h * w).permute(0, 2, 1)
        b, c = avgpool.shape
        avgpool = avgpool[:, None, :]

        return torch.cat([avgpool, feature_map_as_tokens], dim=1)


