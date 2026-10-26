"""
dataset.py
==========
Dataset loading, preprocessing, augmentation, and train/val/test splitting for
TrashNet (clean baseline) and RealWaste (real-world cross-domain test set).

Covers Section 3.2.2 (Data Collection Methods) and Section 3.2.3
(Sampling Strategy / Table 2) of the proposal.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image

import config


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_clean_transforms(img_size=config.IMG_SIZE):
    """Light augmentation used for the controlled-condition TrashNet baseline
    (Section 3.2.2: resize, normalise, rotation/flip/crop)."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(img_size, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_heavy_transforms(img_size=config.IMG_SIZE):
    """Heavier augmentation used for the domain-generalisation experiments
    (Objective 3: heavier augmentation to simulate real-world clutter,
    lighting variation, and contamination)."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),   # simulates occlusion/contamination
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Generic folder-of-classes dataset (works for both TrashNet and RealWaste)
# ---------------------------------------------------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png")

class WasteImageFolder(Dataset):
    """Loads images from `root/<class_name>/*.jpg` where class_name is one of
    `classes`. Keeps raw PIL images until `transform` is applied so the same
    underlying samples can be re-wrapped with different transforms
    (clean vs. heavy augmentation) without re-scanning disk."""

    def __init__(self, root, classes, transform=None):
        self.root = root
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.samples = []
        for c in classes:
            class_dir = os.path.join(root, c)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(IMG_EXTS):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[c]))
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under '{root}' for classes {classes}. "
                f"Check config.py paths and folder names."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TransformSubset(Dataset):
    """Wraps a Subset (or Dataset) so a specific transform can be attached,
    letting the same split use different transforms for train vs eval."""

    def __init__(self, base_dataset_no_transform, indices, transform):
        self.base = base_dataset_no_transform
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def make_trashnet_splits(train_tf, eval_tf, seed=config.RANDOM_SEED):
    """Implements the 70/15/15 train/val/held-out-clean-test split from
    Table 2 and Section 3.2.3."""
    set_seed(seed)
    base = WasteImageFolder(config.TRASHNET_DIR, config.TRASHNET_CLASSES, transform=None)
    n = len(base)
    indices = list(range(n))
    random.shuffle(indices)

    n_train = int(config.TRAIN_SPLIT * n)
    n_val = int(config.VAL_SPLIT * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = TransformSubset(base, train_idx, train_tf)
    val_ds = TransformSubset(base, val_idx, eval_tf)
    test_ds = TransformSubset(base, test_idx, eval_tf)   # held-out CLEAN test
    return train_ds, val_ds, test_ds


class _RelabeledWrapper(Dataset):
    """Wraps a list of (path, label) samples with a given transform."""
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_realwaste_splits(eval_tf, finetune_tf, seed=config.RANDOM_SEED):
    """Splits RealWaste into two DISJOINT sets, per Section 3.2.3:

      - dg_finetune_ds : a small reserved subset (config.DG_FINETUNE_FRACTION,
                         Table 2 last row) used ONLY for fine-tuning /
                         domain-generalisation, over the FULL 9-class
                         RealWaste label space.
      - zero_shot_ds   : the REMAINING images, restricted to the 6 classes
                         shared with TrashNet (Objective 2 / RQ1), used for
                         the zero-shot cross-domain measurement.

    The two sets are built from disjoint image indices so fine-tuning never
    contaminates the zero-shot evaluation, exactly as described in the
    proposal ("this refinement step never contaminates the zero-shot
    cross-domain measurement obtained in Objective 2").
    """
    set_seed(seed)

    full_base = WasteImageFolder(config.REALWASTE_DIR, config.REALWASTE_CLASSES, transform=None)
    n = len(full_base)
    indices = list(range(n))
    random.shuffle(indices)
    n_ft = int(config.DG_FINETUNE_FRACTION * n)
    ft_idx = set(indices[:n_ft])
    remainder_idx = [i for i in indices if i not in ft_idx]

    # --- Reserved fine-tuning subset, full 9-class RealWaste ---------------
    dg_finetune_ds = TransformSubset(full_base, sorted(ft_idx), finetune_tf)

    # --- Zero-shot cross-domain test set: remainder, shared classes only --
    zero_shot_samples = []
    for i in remainder_idx:
        path, rw_label = full_base.samples[i]
        rw_name = full_base.classes[rw_label]
        if rw_name in config.REALWASTE_TO_TRASHNET:
            tn_name = config.REALWASTE_TO_TRASHNET[rw_name]
            tn_label = config.TRASHNET_CLASSES.index(tn_name)
            zero_shot_samples.append((path, tn_label))
    zero_shot_ds = _RelabeledWrapper(zero_shot_samples, eval_tf)

    # --- Held-out portion of the remainder, full 9-class labels, for
    # evaluating the fine-tuned 9-class model post-adaptation -------------
    dg_eval_samples = [full_base.samples[i] for i in remainder_idx]
    dg_eval_ds = _RelabeledWrapper(dg_eval_samples, eval_tf)

    return zero_shot_ds, dg_finetune_ds, dg_eval_ds, full_base.classes


# ---------------------------------------------------------------------------
# Mixup (Objective 3: domain-generalisation technique)
# ---------------------------------------------------------------------------
def mixup_batch(images, labels, num_classes, alpha=0.2):
    """Standard mixup (Zhang et al. 2018), used as one of the domain-
    generalisation techniques evaluated in Objective 3 / Research Question 2."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[perm]
    labels_onehot = torch.zeros(batch_size, num_classes, device=images.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[perm]
    return mixed_images, mixed_labels           