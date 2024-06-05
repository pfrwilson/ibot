from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
from . import utils
import torch 


def NormalizeToTensor(mean=(0.485, 0.456, 0.406), std=(0.485, 0.456, 0.406)):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


@dataclass
class DataAugmentation:
    global_crops_scale: tuple[float, float] = (0.14, 1)
    local_crops_scale: tuple[float, float] = (0.05, 0.4)
    global_crops_number: int = 2
    local_crops_number: int = 0
    global_crops_size: int = 224
    local_crops_size: int = 96
    jitter_prob: float = 0.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    blur_prob_1: float = 0.0
    blur_prob_2: float = 0.0
    solarization_prob: float = 0.0
    initial_crop_size: int | None = None
    initial_resize_size: int | None = None

    def __post_init__(
        self,        
    ):
        self.initial_crop = (
            transforms.RandomCrop(self.initial_crop_size)
            if self.initial_crop_size is not None
            else lambda x: x
        )
        self.initial_resize = (
            transforms.Resize(
                (self.initial_resize_size, self.initial_resize_size), antialias=True
            )
            if self.initial_resize_size is not None
            else lambda x: x
        )

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=self.jitter_prob,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self.global_crops_number = self.global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.global_crops_size,
                    scale=self.global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=self.blur_prob_1),
                normalize,
            ]
        )
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.global_crops_size,
                    scale=self.global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=self.blur_prob_1),
                utils.Solarization(self.solarization_prob),
                normalize,
            ]
        )
        # transformation for the local crops
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.local_crops_size,
                    scale=self.local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=self.blur_prob_2),
                normalize,
            ]
        )

    def __call__(self, image):
        image = self.initial_resize(image)
        image = self.initial_crop(image)
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

    def to_numpy(self, tensor):
        C, H, W = tensor.shape
        tensor *= torch.tensor(self.std)[..., None, None]
        tensor += torch.tensor(self.mean)[..., None, None]
        return tensor.permute(1, 2, 0).numpy()



