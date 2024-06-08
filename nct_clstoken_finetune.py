from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from turtle import back

from pyparsing import C
import torch.utils
import wandb
from src.datasets import NCT2013PatchesDataset
from torchvision import transforms as T
from dotenv import load_dotenv
import os
import torch
from exact_datasets.nct2013.cohort_selection_v2 import (
    CohortSelector,
    CohortSelectionOptions,
)
from src.argparse_utils import add_args, get_kwargs
from src.ssl_evaluation import FineTuning

load_dotenv()


def get_args_parser(): 
    parser = ArgumentParser(
        description="Finetune the model for class token classification on the NCT2013 patches dataset",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="outputs",
        help="Directory to save the test outputs",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model to finetune"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ["NCT_FULL_PATCHES"],
        help="Path to the NCT2013 patches dataset",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    add_args(parser, CohortSelectionOptions, group="Cohort selection")
    group = parser.add_argument_group("Training options")
    group.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    group.add_argument(
        "--backbone_lr", type=float, default=1e-5, help="Learning rate for the backbone"
    )
    group.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train for"
    )
    group.add_argument(
        "--input_size", type=int, default=224, help="Size of the input image"
    )
    group.add_argument(
        "--augment", action="store_true", help="Whether to enable data augmentations"
    )
    group.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    wandb.init(project="ibot", job_type="nct_clstoken_finetune", config=vars(args))
    wandb.define_metric("finetune_val_auc", summary="best", goal="maximize")
    args = wandb.config
    print(args)

    all_dataset = NCT2013PatchesDataset(args.data_path, core_ids="all")
    cohort_selector = CohortSelector(all_dataset.metadata_table)
    train_c, val_c, test_c, _ = cohort_selector.select_cohort(
        CohortSelectionOptions(**get_kwargs(args, CohortSelectionOptions))
    )
    print(f"Train: {len(train_c)} Val: {len(val_c)} Test: {len(test_c)}")

    build_dataset = lambda core_ids, augment: NCT2013PatchesDataset(
        root=args.data_path,
        core_ids=core_ids,
        transform=Transform(input_size=args.input_size, augment=augment),
    )
    train_dataset = build_dataset(train_c, augment=args.augment)
    val_dataset = build_dataset(val_c, augment=False)
    test_dataset = build_dataset(test_c, augment=False)
    kw = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **kw)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kw)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kw)

    n_positive = sum([label for _, label in train_dataset])
    n_negative = len(train_dataset) - n_positive
    class_weights = torch.tensor([n_negative, n_positive], dtype=torch.float32) / len(
        train_dataset
    )
    print(f"Class weights: {class_weights} - using inverse class weights for loss")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights**-1).to(args.device)

    backbone = torch.load(args.model).to(args.device)
    finetuner = FineTuning(
        backbone,
        criterion,
        "adam",
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        epochs=args.epochs,
        in_features=backbone.embed_dim,
        n_classes=2,
        log_fn=wandb.log,
    )
    finetuner.run(train_loader, val_loader)  # type: ignore


class Transform:
    def __init__(self, input_size, augment=False):
        self.augment = augment
        self.input_size = input_size

    def __call__(self, image, metadata):
        image = image.convert("RGB")
        image = T.Resize((self.input_size, self.input_size))(image)
        image = T.ToTensor()(image)
        image = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(
            image
        )

        if self.augment:
            image = T.RandomHorizontalFlip()(image)
            image = T.RandomVerticalFlip()(image)
            image = T.RandomAffine(0, translate=(0.1, 0.1))(image)
            image = T.RandomApply(
                [T.RandomResizedCrop(self.input_size, scale=(0.8, 1.0))], p=0.5
            )(image)

        cancer_label = metadata["grade"] != "Benign"
        cancer_label = torch.tensor(cancer_label).long()
        return image, cancer_label


if __name__ == "__main__":
    main()
