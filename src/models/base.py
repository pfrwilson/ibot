from abc import ABC, abstractmethod
from torch import nn 
import torch 


class BackboneNetwork(nn.Module, ABC): 
    @abstractmethod
    def get_class_token(self, x) -> torch.Tensor: ...

    @abstractmethod 
    def get_feature_map(self, x) -> torch.Tensor: ... 

    def _tokens_to_feature_map(self, tokens: torch.Tensor): 
        B, N, D = tokens.shape 
        H = W = int(N ** 0.5)
        assert H * W == N

        tokens = tokens.view(B, H, W, D)
        tokens = tokens.permute(0, 3, 1, 2)

        return tokens