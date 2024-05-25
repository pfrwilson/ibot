from .vision_transformer import VisionTransformer
from .patch_embed import PatchEmbed, FFTPatchEmbed, FFTPatchEmbedV2
from .swin_transformer import swin_tiny, swin_small, swin_base, swin_large


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


def vit_small_fft_v2(patch_size=16, **kwargs):
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbedV2, **kwargs)


def vit_small_fft_v1(patch_size=16, **kwargs): 
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbed, **kwargs)  



