from typing import Any
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from exact_datasets.nct2013 import NCT2013Dataset, DataKeys
from exact_datasets.nct2013 import cohort_selection as cs
from torchvision.transforms import v2
from torchvision.tv_tensors import Image, Mask
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import time


class NCTProbing:
    def __init__(self, batch_size=8, holdout_center="UVA", size=512):
        print(f"Selecting cores for probing (leaving out {holdout_center})...")
        train, val, test = cs.get_patient_splits_by_center(leave_out=holdout_center)

        print("Getting core ids...")
        train_c = cs.get_core_ids(train)
        val_c = cs.get_core_ids(val)

        print(
            f"Using {len(train_c)} cores for training and {len(val_c)} cores for validation"
        )
        self.train_ds = NCT2013Dataset(
            core_ids=train_c + val_c,
            items=[
                DataKeys.BMODE,
                DataKeys.NEEDLE_MASK,
                DataKeys.PROSTATE_MASK,
                DataKeys.GRADE_GROUP,
                DataKeys.PCT_CANCER,
                DataKeys.PATIENT_ID,
                DataKeys.CORE_ID
            ],
            transform=TransformNCT(size=size),
        )
        sampler = torch.utils.data.DistributedSampler(self.train_ds)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=batch_size, num_workers=4, sampler=sampler
        )
    
    def run_probing(self, model, epoch, device, is_main_process=True):
        return run_probing(model, self.train_loader, epoch, device, is_main_process)


@torch.no_grad()
def run_probing(model, loader, epoch: int, device: str, is_main_process=True) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()

    train_outputs = extract_all_data(model, loader, device)

    if is_main_process: 

        patient_ids = np.unique(train_outputs["patient_id"])

        from sklearn.model_selection import KFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        metrics = {}


        # ======== Linear probing at patch level ===================
        average_train_auc = 0
        average_auc = 0
        for train_index, val_index in KFold(n_splits=5, shuffle=True, random_state=0).split(
            patient_ids
        ):

            train_patient_ids = patient_ids[train_index]
            val_patient_ids = patient_ids[val_index]

            train_index = np.where(np.isin(train_outputs["patient_id"], train_patient_ids))[
                0
            ]
            val_index = np.where(np.isin(train_outputs["patient_id"], val_patient_ids))[0]

            X_train = train_outputs["features"][train_index]
            y_train = train_outputs["labels"][train_index]
            inv_train = train_outputs["involvement"][train_index]

            # for training, exclude low involvement cancer cores
            mask = (y_train == 1) & (inv_train < 40)
            X_train = X_train[~mask]
            y_train = y_train[~mask]

            clf = LogisticRegression(
                random_state=0, max_iter=10000, class_weight="balanced"
            ).fit(X_train, y_train)
            y_hat_train = clf.predict_proba(X_train)[:, 1]

            X_val = train_outputs["features"][val_index]
            y_val = train_outputs["labels"][val_index]
            y_hat_val = clf.predict_proba(X_val)[:, 1]

            train_auc = roc_auc_score(y_train, y_hat_train)
            val_auc = roc_auc_score(y_val, y_hat_val)

            average_train_auc += train_auc
            average_auc += val_auc

        average_auc /= 5
        average_train_auc /= 5

        train_metrics = {"auc": average_train_auc}
        val_metrics = {"auc": average_auc}

        return train_metrics, val_metrics


@torch.no_grad()
def extract_all_data(model, loader, device):

    outputs = {}

    model.eval()

    # Collect all the labels and masks from training and validation
    def extract_features(batch, model, device):
        bmode = batch["bmode"].to(device)

        tokens, (cls_prediction, patch_prediction) = model([bmode], return_backbone_feat=True)
        image_features = tokens_to_feature_map(tokens, cls_is_included=True)
        patch_prediction = tokens_to_feature_map(patch_prediction, cls_is_included=False)
        # image_features = model.image_encoder_wrapped.get_image_features(
        #     bmode
        # )  # B, C, H, W

        needle_mask = batch["needle_mask"].to(device)
        needle_mask = nn.functional.interpolate(
            needle_mask,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="nearest",
        )
        prostate_mask = batch["prostate_mask"].to(device)
        prostate_mask = nn.functional.interpolate(
            prostate_mask,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="nearest",
        )

        mask = (needle_mask > 0.5) & (prostate_mask > 0.5)
        mask = mask[:, 0, :, :]

        batch_idx = torch.arange(len(bmode), device=device)
        batch_idx = batch_idx[:, None, None, None].repeat(
            1, 1, mask.shape[1], mask.shape[2]
        )

        from einops import rearrange

        image_features = rearrange(image_features, "b c h w -> b h w c")
        patch_prediction = rearrange(patch_prediction, "b c h w -> b h w c")
        batch_idx = rearrange(batch_idx, "b c h w -> b h w c")

        image_features = image_features[mask]
        patch_prediction = patch_prediction[mask]
        batch_idx = batch_idx[mask].view(-1).cpu().numpy()

        def map_batch_level_feature_to_patch_level_feature(
            batch_level_features, batch_idx
        ):
            if isinstance(batch_level_features, torch.Tensor):
                batch_level_features = batch_level_features.view(-1).tolist()
            out = []
            for i in range(len(batch_idx)):
                out.append(batch_level_features[batch_idx[i]])
            return np.array(out)

        cancer = batch["cancer"]
        cancer = map_batch_level_feature_to_patch_level_feature(
            cancer.view(-1), batch_idx
        )
        involvement = batch["involvement"]
        involvement = map_batch_level_feature_to_patch_level_feature(
            involvement.view(-1), batch_idx
        )
        grade_group = batch["grade_group"]
        grade_group = map_batch_level_feature_to_patch_level_feature(
            grade_group, batch_idx
        )
        core_id = batch["core_id"]
        core_id = map_batch_level_feature_to_patch_level_feature(core_id, batch_idx)
        patient_id = batch["patient_id"]
        patient_id = map_batch_level_feature_to_patch_level_feature(
            patient_id, batch_idx
        )

        return image_features, patch_prediction, cancer, involvement, grade_group, core_id, patient_id

    for batch in tqdm(loader, desc="Extracting features"):

        image_features, patch_prediction, cancer, involvement, grade_group, core_id, patient_id = (
            extract_features(batch, model, device)
        )
        outputs.setdefault("features", []).append(image_features)
        # outputs.setdefault("patch_prediction", []).append(patch_prediction)
        outputs.setdefault("labels", []).append(cancer)
        outputs.setdefault("involvement", []).append(involvement)
        outputs.setdefault("grade_group", []).append(grade_group)
        outputs.setdefault("core_id", []).append(core_id)
        outputs.setdefault("patient_id", []).append(patient_id)

    outputs["features"] = torch.cat(outputs["features"], dim=0).cpu().numpy()
    # outputs["patch_prediction"] = torch.cat(outputs["patch_prediction"], dim=0).cpu().numpy()
    outputs["labels"] = np.concatenate(outputs["labels"]).reshape(-1)
    outputs["involvement"] = np.concatenate(outputs["involvement"]).reshape(-1)
    outputs["grade_group"] = np.concatenate(outputs["grade_group"]).reshape(-1)
    outputs["core_id"] = np.concatenate(outputs["core_id"]).reshape(-1)
    outputs["patient_id"] = np.concatenate(outputs["patient_id"]).reshape(-1)

    def gather_across_processes(array):
        output_list = [np.zeros_like(array) for _ in range(dist.get_world_size())]
        dist.all_gather_object(output_list, array)
        return np.concatenate(output_list)
            
    for key in outputs:
        outputs[key] = gather_across_processes(outputs[key])

    return outputs


class TransformNCT:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=512):
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, sample):
        bmode = sample["bmode"]
        needle_mask = sample["needle_mask"]
        prostate_mask = sample["prostate_mask"]
        pct_cancer = sample["pct_cancer"]
        grade_group = sample["grade_group"]

        bmode = torch.from_numpy(bmode).float() / 255
        bmode = bmode[None, ...].repeat_interleave(3, 0)
        prostate_mask = torch.from_numpy(prostate_mask)[None, ...].float()
        needle_mask = torch.from_numpy(needle_mask)[None, ...].float()

        bmode = Image(bmode)
        needle_mask = Mask(needle_mask)
        prostate_mask = Mask(prostate_mask)

        bmode, needle_mask, prostate_mask = v2.Resize(
            (self.size, self.size), interpolation=3
        )(bmode, needle_mask, prostate_mask)
        # bmode, needle_mask, prostate_mask = v2.CenterCrop((self.size))(bmode, needle_mask, prostate_mask)
        bmode = v2.Normalize(self.mean, self.std)(bmode)

        pct_cancer = torch.tensor(pct_cancer).float()
        grade_group = torch.tensor(grade_group).long()
        cancer = (grade_group > 0).long()

        output = {}
        output['bmode'] = bmode 
        output['needle_mask'] = needle_mask
        output['prostate_mask'] = prostate_mask
        output['grade_group'] = grade_group
        output['cancer'] = cancer
        output['involvement'] = pct_cancer
        output['core_id'] = sample['core_id']
        output['patient_id'] = sample['patient_id']

        return output


def tokens_to_feature_map(tokens, cls_is_included=True): 
    if cls_is_included:
        # discard cls token
        tokens = tokens[:, 1:, :]

    B, N, D = tokens.shape 
    H = W = int(N ** 0.5)
    assert H * W == N

    tokens = tokens.view(B, H, W, D)
    tokens = tokens.permute(0, 3, 1, 2)

    return tokens