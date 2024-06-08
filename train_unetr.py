from argparse import ArgumentParser
from rich_argparse import ArgumentDefaultsRichHelpFormatter
import torch 
from src.models import unetr
from src.models.vision_transformer import VisionTransformer


def get_arg_parser(): 
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_path', help='Path to the saved model (pickled model, not state dict)')
    return parser 


def train(args): 
    model = torch.load(args.model_path)
    if isinstance(model, VisionTransformer): 
        image_encoder = unetr.VITImageEncoderWrapperForUNETR(vit=model)
        embed_dim = model.embed_dim
    unetr_model = unetr.UNETR(image_encoder, embedding_size=embed_dim, feature_size=14, input_size=224, output_size=224)
    image = torch.randn(1, 3, 224, 224)
    print(unetr_model(image).shape)


def main(): 
    parser = ArgumentParser(formatter_class=ArgumentDefaultsRichHelpFormatter, parents=[get_arg_parser()])
    args = parser.parse_args()
    train(args)


if __name__ == '__main__': 
    main()