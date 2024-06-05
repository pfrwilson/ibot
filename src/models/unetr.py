import einops
import torch
from monai.networks.nets.unetr import (
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from .vision_transformer import VisionTransformer
from .medsam import Sam


class UNETR(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        embedding_size=768,
        feature_size=64,
        out_channels=1,
        norm_name="instance",
        input_size=1024, 
        output_size=256,
    ):
        """
        Args: 
            image_encoder: a vision transformer backbone model. Its call pattern should be `image_encoder(x)`. 
            The return value should be a list of tensors, shaped to B, H, W, C. Only hidden states at certain
            indices will be used in the UNETR model. 12 hidden states are expected.
            embedding_size: the size of the output of the image_encoder model.
            feature_size: the size of the feature maps in the UNETR model. For ViT, this would be image_size / patch_size.
            out_channels: the number of output channels.
            norm_name: the name of the normalization layer to use in the UNETR model.
            input_size: the size of the input image.
            output_size: the size of the output heatmap.
        """

        super().__init__()

        self.image_encoder = image_encoder

        embedding_size = embedding_size
        feature_size = feature_size  # divides embedding size

        # if the input size is greater than the output size, we need to downsample. 
        # however, we don't sample the input but rather the transformer intermediate outputs
        if input_size > output_size:
            self.downsample = torch.nn.MaxPool2d(input_size // output_size)
        else:
            self.downsample = torch.nn.Identity()

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=3,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            upsample_kernel_size=2,
            stride=1,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            num_layer=0, 
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out_block = UnetOutBlock(
            spatial_dims=2,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def vit_out_to_conv_in(self, x):
        return x.permute(0, 3, 1, 2)

    def forward(self, x):
        hiddens = self.image_encoder(x)
        x1 = self.downsample(x)
        x2 = self.downsample(self.vit_out_to_conv_in(hiddens[3]))
        x3 = self.downsample(self.vit_out_to_conv_in(hiddens[6]))
        x4 = self.downsample(self.vit_out_to_conv_in(hiddens[9]))
        x5 = self.downsample(self.vit_out_to_conv_in(hiddens[11]))

        enc1 = self.encoder1(x1)
        enc2 = self.encoder2(x2)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)
        dec4 = x5

        dec3 = self.decoder1(dec4, enc4)
        dec2 = self.decoder2(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder4(dec1, enc1)

        return self.out_block(dec0)
    

class VITImageEncoderWrapperForUNETR(torch.nn.Module):
    def __init__(self, vit: VisionTransformer):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        hiddens = self.vit.get_intermediate_layers(x, n=self.vit.get_num_layers())
        hiddens_feature_maps = [] 
        for hidden in hiddens: 

            patch_tokens = hidden[:, self.vit.n_cls_tokens:, :]
            npatch = int(patch_tokens.shape[1] ** .5)
            feature_map = einops.rearrange(
                patch_tokens, "b (npatch1 npatch2) c -> b npatch1 npatch2 c", npatch1=npatch, npatch2=npatch
            )
            hiddens_feature_maps.append(feature_map)
        return hiddens_feature_maps


class SAMWrapperForUNETR(torch.nn.Module): 
    def __init__(self, sam: Sam):
        super().__init__()
        self.image_encoder = sam.image_encoder
        del self.image_encoder.neck

    def forward(self, x): 
        image_encoder = self.image_encoder
        x = image_encoder.patch_embed(x)
        if image_encoder.pos_embed is not None:
            x = x + image_encoder.pos_embed

        hiddens = []
        for blk in image_encoder.blocks:
            x = blk(x)
            hiddens.append(x)

        return hiddens


if __name__ == "__main__":
    from .vision_transformer import vit_small
    vit = vit_small(patch_size=16, img_size=224)
    image_encoder = VITImageEncoderWrapperForUNETR(vit)
    model = UNETR(image_encoder, embedding_size=384, feature_size=14, input_size=256, output_size=64)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)

    from .medsam import build_medsam
    sam = build_medsam()
    image_encoder = SAMWrapperForUNETR(sam)
    model = UNETR(image_encoder, embedding_size=768, feature_size=64, input_size=1024, output_size=256)
    x = torch.randn(1, 3, 1024, 1024)
    out = model(x)
    print(out.shape)