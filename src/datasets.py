import json
import os
from pathlib import Path
import SimpleITK as sitk
from typing import Callable, Literal
import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
from torchvision.datasets import ImageFolder


class MicroSegNetDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        split: Literal["train", "test"] = "train",
        preprocess: bool | None = None,
        raw_data_dir: Path | str | None = None,
    ):
        """
        Args: 
            transform: Takes tuple of image, mask (PIL.Image) and returns transformed image, mask
        """

        self.root = root if isinstance(root, Path) else Path(root)
        self.transform = transform
        self.split = split

        self._image_folder = self.root / self.split / "micro_ultrasound_scans"
        self._mask_folder = self.root / self.split / "expert_annotations"

        self._image_paths = sorted(
            self._image_folder.iterdir(), key=self._extract_indices
        )
        self._mask_paths = sorted(
            self._mask_folder.iterdir(), key=self._extract_indices
        )

        if len(self._image_paths) == 0 or preprocess:
            # should try to preprocess the data
            assert raw_data_dir is not None, "Raw data directory must be provided"
            raw_data_dir = (
                raw_data_dir if isinstance(raw_data_dir, Path) else Path(raw_data_dir)
            )
            self._preprocess_data(self.root, raw_data_dir)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):
        image_path = self._image_paths[index]
        mask_path = self._mask_paths[index]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask

    def _extract_indices(self, path: Path):
        return list(map(int, re.search(r"(\d+)", path.name).groups()))

    def _preprocess_data(self, target_dir: Path, source_dir: Path):
        target_dir.joinpath("train").mkdir(exist_ok=True)
        target_dir.joinpath("test").mkdir(exist_ok=True)
        for split in ["train", "test"]:
            scans_dir = source_dir.joinpath(split).joinpath("micro_ultrasound_scans")
            annotations_dir = source_dir.joinpath(split).joinpath("expert_annotations")
            target_scans_dir = target_dir.joinpath(split).joinpath(
                "micro_ultrasound_scans"
            )
            target_annotations_dir = target_dir.joinpath(split).joinpath(
                "expert_annotations"
            )
            target_scans_dir.mkdir(exist_ok=True)
            target_annotations_dir.mkdir(exist_ok=True)

            def _read_id_from_path(path):
                id = int(re.search("(\d+)", str(path.name)).groups()[0])
                return id

            scans_paths = sorted(scans_dir.iterdir(), key=_read_id_from_path)
            annotations_paths = sorted(
                annotations_dir.iterdir(), key=_read_id_from_path
            )

            for id, (scan, target) in enumerate(
                tqdm(
                    zip(scans_paths, annotations_paths),
                    desc=f"Preprocessing {split} data",
                    total=len(scans_paths),
                )
            ):
                scan = sitk.GetArrayFromImage(sitk.ReadImage(scan)).astype("uint8")
                annotation = sitk.GetArrayFromImage(sitk.ReadImage(target)).astype(
                    "uint8"
                )

                for frame_idx, (frame, target_frame) in enumerate(
                    zip(scan, annotation)
                ):
                    scan_output_path = target_scans_dir / f"{id}_{frame_idx}.png"
                    annotations_path = target_annotations_dir / f"{id}_{frame_idx}.png"
                    frame = Image.fromarray(frame)
                    target_frame = Image.fromarray(target_frame)
                    frame.save(scan_output_path)
                    target_frame.save(annotations_path)


class ImageMaskTransform: 
    """Default transform for image-mask pairs, PIL to tensor"""

    def __init__(self, size: int = 512, mean: list[float] = [0.485, 0.456, 0.406], std: list[float] = [0.229, 0.224, 0.225]): 
        self.size = size 
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert('RGB')
        image = T.Resize((self.size, self.size))(image)
        image = T.ToTensor()(image)
        image = T.Normalize(mean=self.mean, std=self.std)(image)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.int64)
        return image, mask 


class NCT2013PatchesDataset(Dataset): 
    """Dataset which returns patches and metadata for the NCT2013 dataset."""

    def __init__(self, root: str, core_ids: list[str] | Literal['all'] = 'all', transform: Callable | None = None): 
        """Builds the dataset

        Args:
            root (str): Path to the root directory of the dataset
            core_ids (list[str] | Literal['all']): List of core ids to include in the dataset. If 'all', all core ids are included.
            transform (Callable | None): Transform to apply to the image and metadata. Defaults to None. If not None, the transform should take in a 
                tuple[image: PIL.Image, metadata: dict] and return the transformed image and metadata.
        """

        self.root = root
        self.transform = transform

        self.metadata_table = pd.read_csv(os.path.join(self.root, 'metadata.csv'))

        core_ids = core_ids if core_ids != 'all' else list(self.metadata_table['core_id'].unique())

        self.core_ids = core_ids
        self._indices = []
        for core_id in core_ids: 
            for frame in range(len([path for path in os.listdir(os.path.join(self.root, core_id)) if path.endswith('.png')])): 
                self._indices.append((core_id, frame))

    def _lookup_image_path(self, core_id, frame): 
        return os.path.join(self.root, core_id, f'{core_id}_{frame}.png')

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, idx): 
        core_id, frame = self._indices[idx]
        metadata = self.metadata_table[self.metadata_table['core_id'] == core_id].iloc[0].to_dict()
        image_path = self._lookup_image_path(core_id, frame)
        image = Image.open(image_path)
        if self.transform is not None: 
            return self.transform(image, metadata)
        return image, metadata


class NCT2013ImagesAndCancerMasksDataset(Dataset): 
    def __init__(self, root: str, core_ids: list[str] | Literal['all'] = 'all', transform: Callable | None = None): 
        self.root = root
        self.transform = transform

        if core_ids == 'all': 
            core_ids = os.listdir(root)

        self.core_ids = core_ids

    def __len__(self): 
        return len(self.core_ids)
    
    def __getitem__(self, idx): 
        core_id = self.core_ids[idx]
        img_path = os.path.join(self.root, core_id, 'bmode.png')
        mask_path = os.path.join(self.root, core_id, 'cancer_mask.png')
        metadata_path = os.path.join(self.root, core_id, 'metadata.json')

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        metadata = json.load(open(metadata_path))

        output = (img, mask, metadata) if self.transform is None else self.transform(img, mask, metadata)
        return output


if __name__ == "__main__":
    #dataset = NCT2013PatchesDataset(root='/ssd005/projects/exactvu_pca/nct2013_bmode_patches')
    dataset = NCT2013ImagesAndCancerMasksDataset('/ssd005/projects/exactvu_pca/nct2013_bmode_png')
    breakpoint()
    pass