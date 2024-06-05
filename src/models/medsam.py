from torch import nn
import torch 
from segment_anything import sam_model_registry
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling import Sam
import os
import math


def build_medsam():
    """
    Builds the MedSAM model by building the SAM model and loading the medsam checkpoint.
    """
    MEDSAM_CHECKPOINT = os.environ.get('MEDSAM_CHECKPOINT')
    if MEDSAM_CHECKPOINT is None:
        raise ValueError("MEDSAM_CHECKPOINT environment variable must be set to use medsam.")
    checkpoint = MEDSAM_CHECKPOINT
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    return model


class MedSAMIBot(nn.Module):
    def __init__(self, freeze_pos_embed=False):
        # keep defaults for now
        super().__init__()

        medsam_model = build_medsam()
        image_encoder = medsam_model.image_encoder
        del image_encoder.neck # parameters of neck are unused, delete it to avoid DDP "unused parameter" errors

        self.image_encoder_wrapped = ImageEncoderViTWithClassTokenAndMasking(
            medsam_model.image_encoder
        )

        if freeze_pos_embed: 
            for p in self.image_encoder_wrapped.image_encoder.patch_embed.parameters():
                p.requires_grad = False

    def forward(self, image, mask=None, return_all_tokens=True):
        outputs = self.image_encoder_wrapped(image, mask)
        return outputs


class ImageEncoderViTWithClassTokenAndMasking(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        embedding_dim: int = 768,
        patch_size=16, 
        img_size=1024,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self._patch_size = patch_size
        self._img_size = img_size

        self.class_token_to_image_attns = nn.ModuleList(
            [
                ClassTokenBlock(
                    emb_dim=embedding_dim, mlp_dim=embedding_dim * 4, heads=8
                )
                for _ in range(len(self.image_encoder.blocks))
            ]
        )
        self.mask_token = torch.nn.Parameter(torch.randn(embedding_dim))

    def forward(self, x, mask=None):
        x = self.image_encoder.patch_embed(x)

        if mask is not None:
            x[mask] = self.mask_token.to(x.dtype)

        x = x + self.interpolate_pos_encoding(x)

        cls_token = self.class_token.expand(x.shape[0], -1, -1)

        for blk, blk2 in zip(
            self.image_encoder.blocks, self.class_token_to_image_attns
        ):
            x = blk(x)
            cls_token = blk2(x, cls_token)

        # concatenate to typical output shape expected by vision transformers -
        # B, N, C where N is the number of patches + 1 (for the class token)
        # and class token is the first element along the N dimension
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.cat([cls_token, x], dim=1)

        return x

    def interpolate_pos_encoding(self, x):
        npatch_in_h = x.shape[1]
        npatch_in_w = x.shape[2]

        patch_pos_embed = self.image_encoder.pos_embed

        npatch_native_h = patch_pos_embed.shape[1]
        npatch_native_w = patch_pos_embed.shape[2]

        if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
            return self.image_encoder.pos_embed

        w0 = npatch_in_w
        h0 = npatch_in_h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.permute(0, 3, 1, 2),
            scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        return patch_pos_embed

    def get_image_features(self, image):
        x = self.image_encoder.patch_embed(image)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed
        for blk in self.image_encoder.blocks:
            x = blk(x)
        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2)
        return x


class ClassTokenBlock(nn.Module):
    def __init__(self, emb_dim, mlp_dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, emb_dim)
        )

    def forward(self, image, cls_token):
        shortcut = cls_token
        B, H, W, C = image.shape
        image = image.reshape(B, H * W, C)
        cls_token = self.attn(cls_token, image, image)[0]
        cls_token = cls_token + shortcut
        cls_token = cls_token + self.mlp(cls_token)
        return cls_token