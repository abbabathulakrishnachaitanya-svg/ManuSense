"""
app/vision/preprocessing.py
────────────────────────────
Dataset loading and image pre-processing pipeline for ManuSense.

Design decisions
────────────────
1.  ImageFolder convention
    The organizer dataset ships as  train/<class_name>/*.png  which maps
    directly onto torchvision.datasets.ImageFolder.  Class labels are
    discovered automatically from folder names — no hardcoding.

2.  Grayscale → 3-channel conversion
    ResNet18 (and most ImageNet-pretrained models) expect 3-channel input.
    We convert grayscale to RGB by repeating the single channel three times
    via transforms.Grayscale(num_output_channels=3).  This preserves all
    pixel information while satisfying the model's input contract.

3.  Reproducible train/val split
    torch.utils.data.random_split with a fixed Generator seed gives the same
    split every run, which is required for honest evaluation.

4.  Normalisation
    ImageNet mean/std is used because we start from an ImageNet-pretrained
    ResNet18.  If training from scratch is chosen later, compute dataset
    statistics instead.

5.  Augmentation (training only)
    Random horizontal flip + small rotation adds variety without distorting
    defect morphology.  Aggressive augmentation (colour jitter, large crops)
    is deliberately avoided on greyscale industrial images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from app.config import settings


# ── ImageNet statistics (used because we fine-tune a pre-trained model) ───────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool) -> transforms.Compose:
    """
    Return the torchvision transform pipeline for training or validation.

    Training augmentation is intentionally conservative: a light random
    horizontal flip and a small rotation help the model generalise without
    distorting defect patterns that are orientation-sensitive.

    Args:
        train: True → include augmentation; False → deterministic pipeline.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    size = settings.image_size

    base = [
        # Convert grayscale PNG (L mode) to a 3-channel tensor so that
        # pretrained ResNet18 weights are compatible without architecture change.
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
    ]

    if train:
        augment = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
        ]
    else:
        augment = []

    normalise = [
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    return transforms.Compose(base + augment + normalise)


def build_datasets(
    dataset_root: Path | None = None,
) -> Tuple[Dataset, Dataset, list[str]]:
    """
    Discover class folders under ``dataset_root`` and return train/val splits.

    Args:
        dataset_root: Path to the ImageFolder-compatible directory.
                      Defaults to ``settings.train_dir``.

    Returns:
        (train_dataset, val_dataset, class_names)
        class_names are sorted alphabetically (torchvision ImageFolder default).

    Raises:
        FileNotFoundError: If ``dataset_root`` does not exist.
        ValueError:        If the folder contains no class sub-directories.
    """
    root = Path(dataset_root) if dataset_root else Path(settings.train_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root.resolve()}\n"
            "Check TRAIN_DIR in your .env file."
        )

    # Load the full dataset with training transforms first (we re-wrap below)
    full_dataset = datasets.ImageFolder(root=str(root), transform=None)

    class_names: list[str] = full_dataset.classes
    if not class_names:
        raise ValueError(f"No class sub-directories found under {root}")

    n_total = len(full_dataset)
    n_train = int(n_total * settings.train_val_split)
    n_val   = n_total - n_train

    # Reproducible split — same seed → same indices every run
    generator = torch.Generator().manual_seed(settings.random_seed)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Wrap subsets with their respective transforms.
    # We use TransformSubset so each split gets its own pipeline without
    # duplicating the underlying data.
    train_ds = _TransformSubset(train_subset, transform=get_transforms(train=True))
    val_ds   = _TransformSubset(val_subset,   transform=get_transforms(train=False))

    return train_ds, val_ds, class_names


def build_dataloaders(
    dataset_root: Path | None = None,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and validation DataLoaders ready for the training loop.

    Args:
        dataset_root: Passed through to :func:`build_datasets`.

    Returns:
        (train_loader, val_loader, class_names)
    """
    train_ds, val_ds, class_names = build_datasets(dataset_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_names


def preprocess_single_image(image_path: str | Path) -> torch.Tensor:
    """
    Load and pre-process a single image for inference (no augmentation).

    Args:
        image_path: Path to the inspection image (any PIL-readable format).

    Returns:
        Float tensor of shape (1, 3, image_size, image_size), normalised and
        ready to be passed to the model.

    Raises:
        FileNotFoundError: If the image does not exist.
    """
    from PIL import Image  # local import to keep module importable before Pillow install

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")

    img = Image.open(path).convert("L")   # ensure grayscale regardless of source mode
    pipeline = get_transforms(train=False)
    tensor = pipeline(img)                # shape: (3, H, W)
    return tensor.unsqueeze(0)            # add batch dim → (1, 3, H, W)


# ── Internal helper ──────────────────────────────────────────────────────────

class _TransformSubset(Dataset):
    """
    Wraps a torch Subset and applies a per-split transform at __getitem__
    time, replacing the parent dataset's transform.

    This avoids the common pitfall of applying training augmentation to the
    validation split when both share the same underlying ImageFolder object.
    """

    def __init__(self, subset: torch.utils.data.Subset, transform: transforms.Compose) -> None:
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        # ImageFolder returns (PIL.Image, label) when its own transform=None
        image, label = self.subset.dataset[self.subset.indices[idx]]
        image = self.transform(image)
        return image, label
