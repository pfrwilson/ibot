from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from omegaconf import OmegaConf
import torch


def main(): 
    parser = ArgumentParser(description="Extract model from a training run checkpoint", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help="Path to the checkpoint to extract")
    parser.add_argument('--output', '-o', type=str, default='model.pth', help="Path to save the extracted model")
    parser.add_argument('--mode', choices=['main_ibot_multimodel'], default='main_ibot_multimodel', help="Mode to extract the model checkpoint")
    parser.add_argument('--model', '-m', type=str, default='teacher', help="Model to extract")

    args = parser.parse_args()

    state_dict = torch.load(args.checkpoint, map_location='cpu')

    conf = state_dict['args']

    match args.mode: 
        case 'main_ibot_multimodel':
            from main_ibot_multimodel import build_models
            vit_student, vit_teacher, cnn_student, cnn_teacher = build_models(conf)
            match args.model: 
                case 'student' | 'vit_student':
                    model = vit_student.backbone
                    msg = model.load_state_dict(
                        extract_state_dict_with_prefix(state_dict['student'], 'vit.module.backbone.')
                    )
                    print(msg)
                case 'teacher' | 'vit_teacher':
                    model = vit_teacher.backbone
                    msg = model.load_state_dict(
                        extract_state_dict_with_prefix(state_dict['teacher'], 'vit.backbone.')
                    )
                    print(msg)
                case 'cnn_student':
                    model = cnn_student.backbone
                    msg = model.load_state_dict(
                        extract_state_dict_with_prefix(state_dict['student'], 'cnn.module.backbone.')
                    )
                    print(msg)
                case 'cnn_teacher':
                    model = cnn_teacher.backbone
                    msg = model.load_state_dict(
                        extract_state_dict_with_prefix(state_dict['teacher'], 'cnn.backbone.')
                    )
                    print(msg)
                case _:
                    raise ValueError(f"Unknown model {args.model}")

    print(f"Saving model to {args.output}")
    torch.save(model, args.output)


def extract_state_dict_with_prefix(state_dict, prefix): 
    return {
        k.replace(prefix, ''): v for k, v in state_dict.items() if k.startswith(prefix)
    }


if __name__ == "__main__":
    main()