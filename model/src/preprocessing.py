# ==============================================================================
# preprocessing.py: Data Pipeline, Augmentations, & PyTorch DataLoaders
# ==============================================================================
# This module transforms raw Brain MRI scan image files into normalized,
# batch-ready PyTorch Tensors for training, validation, and testing.
#
# Key Responsibilities:
# 1. Medical Image Augmentation: Enhances generalization against scanner differences
# 2. Strict Train/Validation Split: Prevents data leakage between training and validation
# 3. Double ImageFolder Technique: Ensures validation never receives training augmentations
# 4. Pinned-Memory DataLoaders: Enables fast GPU transfer across PCIe buses
# ==============================================================================

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ------------------------------------------------------------------------------
# STEP 1: IMAGENET PRETRAINING CONSTANTS
# ------------------------------------------------------------------------------
# EfficientNet-B0 was pretrained on ImageNet (1.2+ million images).
# To reuse its pre-learned convolutional filter weights, inputs must be normalized
# using the exact per-channel mean and standard deviation from ImageNet.
# Formula: normalized_pixel = (pixel - mean) / std
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Mean pixel values for [Red, Green, Blue]
IMAGENET_STD  = [0.229, 0.224, 0.225]  # Standard deviations for [Red, Green, Blue]


# ------------------------------------------------------------------------------
# STEP 2: TRANSFORMATION PIPELINES (AUGMENTATION VS DETERMINISTIC)
# ------------------------------------------------------------------------------
def get_transforms(img_size: int = 224):
    """
    Builds data preprocessing pipelines for training and validation/testing.

    Train Pipeline (Stochastic / Online Augmentations):
    - Run dynamically on the CPU on each epoch to artificially expand the dataset.
    - Mitigates overfitting on small medical datasets without altering pathological facts.

    Validation & Test Pipeline (Deterministic):
    - No random modifications; guarantees exact, reproducible evaluation.
    """
    train_tf = transforms.Compose([
        # 1. Scale slightly larger (e.g. 244x244) before cropping
        transforms.Resize((img_size + 20, img_size + 20)),
        
        # 2. Randomly crop a 224x224 window: Simulates slight positioning shifts in MRI gantry
        transforms.RandomCrop(img_size),
        
        # 3. Horizontal Flip (50% probability): Brains have bilateral symmetry;
        #    a tumor on the left hemisphere could just as validly develop on the right.
        transforms.RandomHorizontalFlip(),
        
        # 4. Random Rotation (+/- 15 degrees): Accounts for slight patient head tilt during scan
        transforms.RandomRotation(15),
        
        # 5. Color Jitter: Simulates variations in MRI scanner contrast / magnetic field strength
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        
        # 6. ToTensor: Converts PIL image [0, 255] to PyTorch float32 Tensor [0.0, 1.0] in (C, H, W)
        transforms.ToTensor(),
        
        # 7. Normalize: Shifts values to zero-mean unit-variance matching ImageNet
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        # Resize directly to target model input size (224x224 for EfficientNet-B0)
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, val_tf


# ------------------------------------------------------------------------------
# STEP 3: DATALOADER FACTORY
# ------------------------------------------------------------------------------
def get_dataloaders(
    dataset_dir=None,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,   # Set to 0 on Windows to avoid multi-processing socket crashes
):
    """
    Constructs PyTorch DataLoaders for Training, Validation, and Testing sets.
    
    Split Strategy:
    - Training/ is split into 85% Train and 15% Validation using random permutation.
    - Testing/ is kept 100% separate as the final holdout test set.
    """
    # Resolve directory paths
    if dataset_dir is None:
        dataset_dir = Path(__file__).parent / "dataset"
    dataset_dir = Path(dataset_dir)

    train_dir = dataset_dir / "Training"
    test_dir  = dataset_dir / "Testing"

    # Verify dataset exists on disk before attempting to load
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Expected layout: dataset/Training/<class>/*.jpg"
        )

    # Fetch transformation objects
    train_tf, val_tf = get_transforms(img_size)

    # --------------------------------------------------------------------------
    # ARCHITECTURAL TRICK: TWO SEPARATE IMAGEFOLDER OBJECTS
    # --------------------------------------------------------------------------
    # In PyTorch, a 'Subset' references the transform of its parent dataset.
    # If we created one ImageFolder with train_tf and split it, the validation subset
    # would also receive random crops and flips!
    # By creating two separate ImageFolder objects from the same folder, train_ds
    # uses train_tf (augmented) while val_ds uses val_tf (clean & deterministic).
    train_full = datasets.ImageFolder(train_dir, transform=train_tf)
    val_full   = datasets.ImageFolder(train_dir, transform=val_tf)

    # --------------------------------------------------------------------------
    # DETERMINISTIC INDEX SPLITTING (85% Train / 15% Validation)
    # --------------------------------------------------------------------------
    n_total    = len(train_full)
    val_size   = int(val_split * n_total)
    train_size = n_total - val_size

    # Fix generator seed so the same image indices are selected every single run
    generator     = torch.Generator().manual_seed(seed)
    indices       = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    # Create distinct Subset instances
    train_ds = Subset(train_full, train_indices)
    val_ds   = Subset(val_full,   val_indices)
    test_ds  = datasets.ImageFolder(test_dir, transform=val_tf)

    # --------------------------------------------------------------------------
    # BUILD PYTORCH DATALOADERS
    # --------------------------------------------------------------------------
    # pin_memory=True allocates page-locked memory on CPU host, speeding up CUDA copies
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader  = DataLoader(train_ds, shuffle=True,  **loader_kwargs)  # Shuffle training batches
    val_loader    = DataLoader(val_ds,   shuffle=False, **loader_kwargs)  # Do not shuffle validation
    test_loader   = DataLoader(test_ds,  shuffle=False, **loader_kwargs)  # Do not shuffle test set

    # --------------------------------------------------------------------------
    # METADATA BUNDLE
    # --------------------------------------------------------------------------
    # ImageFolder automatically sorts subdirectories alphabetically:
    # 0: glioma, 1: meningioma, 2: notumor, 3: pituitary
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


# ------------------------------------------------------------------------------
# STEP 4: DENORMALIZATION HELPER (FOR VISUALIZATION & GRAD-CAM)
# ------------------------------------------------------------------------------
def denormalise(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Inverts the ImageNet normalisation formula:
        original_pixel = (normalized_pixel * std) + mean
    
    Converts a normalized PyTorch Tensor (C, H, W) into an RGB NumPy array (H, W, C)
    with pixel values clamped between 0.0 and 1.0, suitable for Matplotlib or OpenCV.
    """
    t = tensor.clone().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m      # Inverse linear formula
    # Clamp to [0, 1] range and permute dimensions from (C, H, W) -> (H, W, C)
    return t.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()


# ------------------------------------------------------------------------------
# QUICK MODULE TEST
# ------------------------------------------------------------------------------
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
