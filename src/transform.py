from dataclasses import dataclass
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms.v2.functional import horizontal_flip
from . import utils
import torch
import numpy as np
from torchvision.transforms.v2 import Lambda
from torchvision.transforms import v2 as T
from torchvision import tv_tensors as tvt


def NormalizeToTensor(mean=(0.485, 0.456, 0.406), std=(0.485, 0.456, 0.406)):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class SpeckleNoise:
    def __init__(self, amount=0.1, apply_to_greyscale=True):
        self.amount = amount
        self.apply_to_greyscale = apply_to_greyscale

    def __call__(self, img):
        fmt = img.format
        if self.apply_to_greyscale:
            img = img.convert("L")
        img = np.array(img) / 255.0
        img = img + self.amount * img.std() * np.random.randn(*img.shape)
        img = np.clip(img, 0, 1)
        return Image.fromarray((img * 255).astype(np.uint8)).convert(fmt)


class SaltAndPepperNoise:
    def __init__(self, amount=0.1, apply_to_greyscale=True):

        self.amount = amount
        self.apply_to_greyscale = apply_to_greyscale

    def __call__(self, img):
        if self.apply_to_greyscale:
            img = img.convert("L")
        img = np.array(img)
        mask = np.random.rand(*img.shape) < self.amount
        img[mask] = np.random.randint(0, 256, mask.sum())
        return Image.fromarray(img).convert("RGB")


class RandomNoise:
    def __init__(self, amount_range=(0, 1), apply_to_greyscale=True, type="speckle"):
        self.amount_range = amount_range
        self.apply_to_greyscale = apply_to_greyscale
        self.type = type

    def __call__(self, img):
        if self.type == "speckle":
            return SpeckleNoise(
                np.random.uniform(*self.amount_range), self.apply_to_greyscale
            )(img)
        elif self.type == "salt_and_pepper":
            return SaltAndPepperNoise(
                np.random.uniform(*self.amount_range), self.apply_to_greyscale
            )(img)
        else:
            raise ValueError(f"Unknown noise type: {self.type}")


class RandomGamma:
    def __init__(self, gamma_range=(0.5, 4)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = np.random.uniform(*self.gamma_range)
        return T.functional.adjust_gamma(img, gamma)


class RandomContrast:
    def __init__(self, contrast_range=(0.6, 3)):
        self.contrast_range = contrast_range

    def __call__(self, img):
        contrast = np.random.uniform(*self.contrast_range)
        return T.functional.adjust_contrast(img, contrast)


def get_pixel_level_augmentations_for_greyscale_images(
    blur_prob: float = 0.0,
    speckle_prob: float = 0.0,
    speckle_amount: tuple[float, float] = (0.01, 0.1),
    salt_and_pepper_prob: float = 0.0,
    salt_and_pepper_amount: tuple[float, float] = (0.01, 0.1),
    random_gamma_prob: float = 0.0,
    random_gamma_range: tuple[float, float] = (0.5, 4),
    random_contrast_prob: float = 0.0,
    random_contrast_range: tuple[float, float] = (0.6, 3),
):
    return T.Compose(
        [
            T.RandomApply(
                [RandomNoise((speckle_amount), apply_to_greyscale=True, type="speckle")],
                p=speckle_prob,
            ) if speckle_prob > 0 else T.Identity(),
            T.RandomApply(
                [
                    RandomNoise(
                        (salt_and_pepper_amount), apply_to_greyscale=True, type="salt_and_pepper"
                    )
                ],
                p=salt_and_pepper_prob,
            ) if salt_and_pepper_prob > 0 else T.Identity(),
            T.RandomApply([RandomGamma(random_gamma_range)], p=random_gamma_prob) if random_gamma_prob > 0 else T.Identity(),
            T.RandomApply([RandomContrast(random_contrast_range)], p=random_contrast_prob) if random_contrast_prob > 0 else T.Identity(),
            T.RandomApply([T.GaussianBlur(3)], p=blur_prob) if blur_prob > 0 else T.Identity(),
        ]
    )


class ImageMaskTransform:
    """Default transform for image-mask pairs, PIL to tensor"""

    def __init__(
        self,
        size: int = 512,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        initial_crop_size: int | None = None,
        initial_crop_scale: tuple[float, float] | None = None,
        pixel_level_augmentations=None,
        to_tensor=True,
    ):
        self.size = size
        self.mean = mean
        self.std = std
        self.pixel_level_augmentations = pixel_level_augmentations
        self.to_tensor = to_tensor

        if initial_crop_size is not None:
            if initial_crop_scale is not None:
                self.initial_crop = T.RandomResizedCrop(
                    initial_crop_size, scale=initial_crop_scale
                )
            else:
                self.initial_crop = T.Compose(
                    [
                        T.RandomCrop(initial_crop_size),
                        T.Resize(initial_crop_size),
                    ]
                )
        else:
            self.initial_crop = T.Identity()

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        image = tvt.Image(image) / 255.0
        mask = tvt.Mask(mask)

        image, mask = self.initial_crop(image, mask)

        if self.pixel_level_augmentations is not None:
            image = T.ToPILImage()(image)
            image = self.pixel_level_augmentations(image)
            image = tvt.Image(image)

        image, mask = T.Resize((self.size, self.size))(image, mask)

        if not self.to_tensor:
            image, mask = T.ToPILImage()(image, mask)
            return image, mask

        image, mask = T.ToTensor()(image, mask)
        image = T.ToDtype(torch.float32)(image)
        image = T.Normalize(mean=self.mean, std=self.std)(image)
        return image, mask[0].long()


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
    speckle_prob: float = 0.0
    salt_and_pepper_prob: float = 0.0
    random_gamma_prob: float = 0.0
    random_contrast_prob: float = 0.0
    initial_crop_size: int | None = None
    initial_crop_scale: tuple[float, float] | None = None
    initial_resize_size: int | None = None
    noise_before_crop: bool = False
    return_PIL: bool = False
    horizontal_flip_prob: float = 0.5

    def __post_init__(
        self,
    ):
        if self.initial_crop_size is not None:
            if self.initial_crop_scale is not None:
                self.initial_crop = T.RandomResizedCrop(
                    self.initial_crop_size, scale=self.initial_crop_scale
                )
            else:
                self.initial_crop = T.Compose(
                    [
                        T.RandomCrop(self.initial_crop_size),
                        T.Resize(self.initial_crop_size),
                    ]
                )
        else:
            self.initial_crop = lambda x: x

        # self.initial_crop = (
        #     T.RandomCrop(self.initial_crop_size)
        #     if self.initial_crop_size is not None
        #     else lambda x: x
        # )
        # self.initial_resize = (
        #     T.Resize(
        #         (self.initial_resize_size, self.initial_resize_size), antialias=True
        #     )
        #     if self.initial_resize_size is not None
        #     else lambda x: x
        #     )

        strong_augs = T.Compose(
            [
                T.RandomApply(
                    [RandomNoise(apply_to_greyscale=True, type="speckle")],
                    p=self.speckle_prob,
                ),
                T.RandomApply(
                    [
                        RandomNoise(
                            (0.01, 0.1), apply_to_greyscale=True, type="salt_and_pepper"
                        )
                    ],
                    p=self.salt_and_pepper_prob,
                ),
                T.RandomApply([RandomGamma()], p=self.random_gamma_prob),
                T.RandomApply([RandomContrast()], p=self.random_contrast_prob),
            ]
        )

        weak_augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=self.jitter_prob,
                ),
                T.RandomGrayscale(p=0.2),
                utils.GaussianBlur(p=self.blur_prob_1),
            ]
        )

        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        self.global_crops_number = self.global_crops_number

        def make_global_transform(crop):
            if self.noise_before_crop:
                transforms_ = [strong_augs, crop, weak_augs]
            else:
                transforms_ = [crop, strong_augs, weak_augs]
            if self.return_PIL:
                transforms_.append(Lambda(lambda x: x.convert("RGB")))
            else:
                transforms_.append(normalize)
            return T.Compose(transforms_)

        # transformation for the first global crop
        self.global_transfo1 = make_global_transform(
            T.RandomResizedCrop(
                self.global_crops_size,
                scale=self.global_crops_scale,
                interpolation=Image.BICUBIC,
            ),
        )
        # transformation for the rest of global crops
        self.global_transfo2 = make_global_transform(
            T.RandomResizedCrop(
                self.global_crops_size,
                scale=self.global_crops_scale,
                interpolation=Image.BICUBIC,
            ),
        )
        # transformation for the local crops
        self.local_transfo = make_global_transform(
            T.RandomResizedCrop(
                self.local_crops_size,
                scale=self.local_crops_scale,
                interpolation=Image.BICUBIC,
            ),
        )

    def __call__(self, image):
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


if __name__ == "__main__":
    from .datasets import NCT2013ImagesAndCancerMasksDataset

    dataset = NCT2013ImagesAndCancerMasksDataset(
        "/ssd005/projects/exactvu_pca/nct2013_bmode_png"
    )
    dataset.transform = lambda im, *_: DataAugmentation(
        global_crops_number=2,
        local_crops_number=0,
        global_crops_size=224,
        local_crops_size=96,
        initial_crop_size=256,
        initial_resize_size=256,
        jitter_prob=0,
        blur_prob_1=0,
        blur_prob_2=0,
        solarization_prob=0,
        speckle_prob=0.2,
        salt_and_pepper_prob=0.1,
        random_gamma_prob=0.5,
        random_contrast_prob=0.3,
        noise_before_crop=False,
        return_PIL=True,
        initial_crop_scale=(0.4, 1),
        global_crops_scale=(0.2, 1),
        horizontal_flip_prob=0.0,
    )(im)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(2, 10, figsize=(20, 2))
    for i in range(10):
        img = dataset[0]
        for j, crop in enumerate(img):
            ax[j][i].imshow(crop)
            ax[j][i].axis("off")

    plt.show()
    plt.savefig("augmentation.png", dpi=1000)
