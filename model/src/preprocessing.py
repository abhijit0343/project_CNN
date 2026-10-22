"""
preprocessing.py
────────────────────────────────────────────────────────────────────────────────
Builds PyTorch DataLoaders from the raw Brain Tumour MRI dataset.

Dataset layout expected on disk
────────────────────────────────
  src/dataset/
      Training/
          glioma/        *.jpg / *.png
          meningioma/
          notumor/
          pituitary/
      Testing/
          glioma/
          meningioma/
          notumor/
          pituitary/

The training split is further divided into 85 % train / 15 % val at runtime
using torch.utils.data.Subset.  Two separate ImageFolder objects (one with
augmentation transforms, one without) are used so that the val subset is never
augmented even though it draws from the same source folder.

ImageNet statistics are used for normalisation because the backbone
(EfficientNet-B0) was pretrained on ImageNet.

Usage
─────
    from preprocessing import get_dataloaders

    train_loader, val_loader, test_loader, meta = get_dataloaders()
    print(meta)  # {'class_names': [...], 'num_classes': 4, ...}
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── ImageNet statistics ────────────────────────────────────────────────────────
# EfficientNet was pretrained on ImageNet, so we must normalise with the same
# per-channel mean and standard deviation used during that pretraining.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224):
    """
    Return (train_transform, val_transform) for the given image size.

    Train transform — online augmentation applied each epoch:
      • Resize to img_size+20 then RandomCrop back to img_size
        → simulates slight scale variation without losing edges
      • RandomHorizontalFlip  (50 % probability)
      • RandomRotation ±15°   → invariance to head tilt in MRI scans
      • ColorJitter ±20 %     → brightness / contrast variation
      • ToTensor + Normalize  → convert to [0,1] then apply ImageNet stats

    Val / test transform — deterministic, no augmentation:
      • Resize to img_size × img_size
      • ToTensor + Normalize
    """
    train_tf = transforms.Compose([
        # Resize slightly larger, then crop → random spatial jitter
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, val_tf


def get_dataloaders(
    dataset_dir=None,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,   # MUST be 0 on Windows (multiprocessing broken in notebooks)
):
    """
    Build and return DataLoaders for train, val, and test splits.

    Parameters
    ──────────
    dataset_dir : path to the folder containing Training/ and Testing/
                  Defaults to <this file's parent>/dataset
    img_size    : square image side-length fed to the model (224 for EfficientNet)
    batch_size  : number of images per mini-batch
    val_split   : fraction of Training set held out for validation (0.15 = 15 %)
    seed        : random seed for reproducible train/val index split
    num_workers : DataLoader worker processes — keep 0 on Windows

    Returns
    ───────
    train_loader, val_loader, test_loader : torch DataLoader objects
    meta : dict with class_names, num_classes, mean, std, img_size, split sizes
    """
    # ── Resolve dataset directory ──────────────────────────────────────────────
    if dataset_dir is None:
        dataset_dir = Path(__file__).parent / "dataset"
    dataset_dir = Path(dataset_dir)

    train_dir = dataset_dir / "Training"
    test_dir  = dataset_dir / "Testing"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Expected layout: dataset/Training/<class>/*.jpg"
        )

    # ── Build transforms ───────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms(img_size)

    # ── Two separate ImageFolder instances ────────────────────────────────────
    # We load the same Training folder twice — once with augmentation (for the
    # train Subset) and once without (for the val Subset).  This avoids the
    # subtle bug where reassigning val_ds.dataset also changes the transforms
    # seen by the train indices because they share the same underlying object.
    train_full = datasets.ImageFolder(train_dir, transform=train_tf)
    val_full   = datasets.ImageFolder(train_dir, transform=val_tf)

    # ── Deterministic index split ──────────────────────────────────────────────
    n_total    = len(train_full)
    val_size   = int(val_split * n_total)
    train_size = n_total - val_size

    # Fix the generator seed so the split is reproducible across runs
    generator     = torch.Generator().manual_seed(seed)
    indices       = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    # Subsets point to different ImageFolder objects → correct transforms each
    train_ds = Subset(train_full, train_indices)
    val_ds   = Subset(val_full,   val_indices)
    test_ds  = datasets.ImageFolder(test_dir, transform=val_tf)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # pin_memory=True copies tensors into pinned (page-locked) host memory
    # before the GPU transfer, which speeds up the host-to-device copy.
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader  = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader    = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader   = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Metadata dict ──────────────────────────────────────────────────────────
    # class_names are sorted alphabetically by ImageFolder (glioma, meningioma,
    # notumor, pituitary) — this order defines the integer class indices.
    class_names = train_full.classes
    meta = {
        "class_names" : class_names,
        "num_classes" : len(class_names),
        "img_size"    : img_size,
        "mean"        : IMAGENET_MEAN,
        "std"         : IMAGENET_STD,
        "train_size"  : train_size,
        "val_size"    : val_size,
        "test_size"   : len(test_ds),
    }

    return train_loader, val_loader, test_loader, meta


def denormalise(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse ImageNet normalisation so a tensor can be displayed as an image.

    Accepts a CxHxW float tensor; returns an HxWxC numpy array in [0, 1].
    Used in notebooks / Grad-CAM overlays to recover the original pixel values.
    """
    t = tensor.clone().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m      # inverse: x = z * std + mean
    return t.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("preprocessing.py — building DataLoaders …")
    train_l, val_l, test_l, meta = get_dataloaders()
    print(f"  Classes    : {meta['class_names']}")
    print(f"  Train size : {meta['train_size']:,}")
    print(f"  Val   size : {meta['val_size']:,}")
    print(f"  Test  size : {meta['test_size']:,}")
    imgs, lbls = next(iter(train_l))
    print(f"  Batch shape: {tuple(imgs.shape)}   (B, C, H, W)")
    print("preprocessing.py — OK ✅")
