from .vision_transformer import VisionTransformer
from .patch_embed import PatchEmbed, FFTPatchEmbed, FFTPatchEmbedV2
from .swin_transformer import swin_tiny, swin_small, swin_base, swin_large
import torch 
from torch import nn
from . import vision_transformer as vits
from torchvision import models as torchvision_models


_MODEL_REGISTRY = {}


def register_model(func): 
    _MODEL_REGISTRY[func.__name__] = func
    return func


def get_model(name, **kwargs):
    return _MODEL_REGISTRY[name](**kwargs)


@register_model
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


@register_model
def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


@register_model
def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


@register_model
def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


@register_model
def vit_small_fft_v2(patch_size=16, **kwargs):
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbedV2, **kwargs)


@register_model
def vit_small_fft_v1(patch_size=16, **kwargs): 
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbed, **kwargs)  


@register_model 
def medsam_ibot(**kwargs): 
    from medsam import MedSAMIBot
    model = MedSAMIBot(**kwargs)
    model.embed_dim = 768
    return model


def _resnet(name, **kwargs):
    from timm.models import resnet
    model = resnet.__dict__[name](pretrained=False, **kwargs)
    embed_dim = model.fc.weight.shape[1]
    model.embed_dim = embed_dim
    model.fc = nn.Identity()
    from .wrappers import ResnetWrapper
    model = ResnetWrapper(model)
    return model


@register_model
def resnet18(**kwargs):
    return _resnet('resnet18', **kwargs)


@register_model
def resnet34(**kwargs):
    return _resnet('resnet34', **kwargs)


@register_model
def resnet50(**kwargs):
    return _resnet('resnet50', **kwargs)