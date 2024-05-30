import os
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader, dataloader
from torch.utils.data.distributed import Dataset
from torchvision import transforms as T 
import torch.distributed
from tqdm import tqdm 
import torch
from torch import Tensor, nn
import numpy as np 
from sklearn.metrics import *
import matplotlib.pyplot as plt
import warnings 
import wandb 
from .utils import concat_all_gather, is_main_process, MultiCropWrapper
from abc import ABC, abstractmethod
from .transform import NormalizeToTensor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV



def compute_binary_classification_metrics(y_score, y_true, log_images=False):
    """Calculate metrics for the cancer classification problem.

    Args:
        y_score (np.array or torch.Tensor) - A column vector of predicted probabilities for
            cancer (1) or benign(0)
        y_true (np.array or torch.Tensor) - A column vector of true labels for cancer (1) or benign(0)
        log_images (bool) - If True, log images of the histogram of predictions and the ROC curve to
            wandb. Default is False.
    """

    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    # augmentations can cause NaNs
    nanvalues = np.isnan(y_score)
    y_score = y_score[~nanvalues]
    y_true = y_true[~nanvalues]

    metrics = {}

    try: 
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        warnings.warn("ROC AUC score could not be calculated. Setting to 0.5")
        metrics["auc"] = 0.5

    # find the sensitivity at fixed specificities
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    for specificity in [0.20, 0.40, 0.60, 0.80]:
        sensitivity = tpr[np.argmax(fpr > 1 - specificity)]
        metrics[f"sens_at_{specificity*100:.0f}_spe"] = sensitivity

    # choose the threshold that maximizes balanced accuracy
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    metrics["f1"] = f1_score(y_true, y_score > best_threshold)

    if log_images:
        plt.hist(y_score[y_true == 0], bins=100, alpha=0.5, density=True)
        plt.hist(y_score[y_true == 1], bins=100, alpha=0.5, density=True)
        plt.legend(["Benign", "Cancer"])
        plt.xlabel(f"Probability of cancer")
        plt.ylabel("Density")
        plt.title(f"AUC: {metrics['auc']:.3f}")
        metrics["histogram"] = wandb.Image(plt, caption="Histogram of core predictions")
        plt.close()

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        metrics["roc_curve"] = wandb.Image(plt, caption="ROC curve")
        plt.close()

    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_score > best_threshold)
    metrics['auprc'] = average_precision_score(y_true, y_score)

    return metrics


class IBOTModule(nn.Module, ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: 
        """Run forward pass on the model. 
        
        Args: 
            x: Input tensor. should be an image tensor of shape B, C, H, W

        Returns:
            Output tensor. This will be a tensor of shape B, N, C, where B is the batch size 
            and N is the number of tokens (one token per patch plus one for the cls token). 
            cls token is the 0'th token.
        """


class LinearProbing: 
    def __init__(self, train_loader, val_loader, device):
        """Initialize the linear probing evaluator.
        
        Args:
            train_loader: DataLoader for the training set - batches should be image, label tuples.
            val_loader: DataLoader for the validation set. 
            device: Device to run the evaluation on.
        """
        self.train_loader = train_loader 
        self.val_loader = val_loader 
        self.device = device

    @torch.no_grad()
    def _extract_features(self, loader, model, desc: str = None): 
        model.eval().to(self.device)

        features = []
        labels = []
        for (image, label) in tqdm(loader, desc=desc): 
            image = image.to(self.device)
            label = label.to(self.device)
            tokens, _ = model([image], return_backbone_feat=True)
            cls = tokens[:, 0, :]
            cls = concat_all_gather(cls.contiguous())
            label = concat_all_gather(label.contiguous())
            features.append(cls)
            labels.append(label)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels 

    def run_probing(self, model: MultiCropWrapper):
        """Returns the metrics for the linear probing task."""

        X_train, y_train = self._extract_features(self.train_loader, model)
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        X_val, y_val = self._extract_features(self.val_loader, model)
        X_val = X_val.cpu().numpy()
        y_val = y_val.cpu().numpy()

        if torch.distributed.is_initialized(): 
            if torch.distributed.get_rank() != 0: 
                return 

        clf = LogisticRegression(max_iter=5000, class_weight='balanced')
        clf.fit(X_train, y_train)

        y_pred_train = clf.predict_proba(X_train)[:, -1]
        y_pred_val = clf.predict_proba(X_val)[:, -1]

        train_metrics = compute_binary_classification_metrics(y_pred_train, y_train)
        val_metrics = compute_binary_classification_metrics(y_pred_val, y_val)
 
        return train_metrics, val_metrics 


def build_linear_probe_for_nct_patches(
    data_path: str = os.environ['NCT_PATCHES'],
    input_size: int = 512,
    batch_size: int = 8,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        NormalizeToTensor(), 
    ])

    train_ds = ImageFolder(os.path.join(data_path, 'train'), transform=transform, target_transform=lambda l: torch.tensor(l).long())
    val_ds = ImageFolder(os.path.join(data_path, 'val'), transform=transform, target_transform=lambda l: torch.tensor(l).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return LinearProbing(train_loader, val_loader, device)