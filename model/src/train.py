# ==============================================================================
# train.py: Two-Phase Transfer Learning Trainer for NeuroScan CNN
# ==============================================================================
# This script trains an EfficientNet-B0 Convolutional Neural Network on Brain
# Tumour MRI scans (glioma, meningioma, notumor, pituitary).
#
# Core Techniques Employed:
# 1. Transfer Learning: Uses ImageNet-pretrained weights instead of training from scratch.
# 2. Two-Phase Strategy:
#    - Phase 1: Freeze backbone, train only the newly attached classification head.
#    - Phase 2: Unfreeze top 30% of backbone layers, fine-tune at 10x lower learning rate.
# 3. Automatic Mixed Precision (AMP): fp16 for matrix multiplies + dynamic GradScaler.
# 4. Label Smoothing: Prevents overconfident probability predictions.
# 5. Cosine Annealing Learning Rate: Smooth decay avoiding abrupt optimizer shocks.
# ==============================================================================

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ------------------------------------------------------------------------------
# STEP 1: MODULE PATH RESOLUTION
# ------------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import dataset preparation pipeline from local preprocessing module
from preprocessing import get_dataloaders   # noqa: E402

# Models directory where weights and training metadata are saved
MODELS_DIR = SRC_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ==============================================================================
# STEP 2: NEURAL NETWORK ARCHITECTURE DEFINITION
# ==============================================================================
import timm  # PyTorch Image Models library


class NeuroScanCNN(nn.Module):
    """
    Custom Deep CNN Model:
    - Backbone: EfficientNet-B0 (feature extractor with compound scaling)
    - Head: Global Average Pooling + BatchNorm + 2-layer MLP with Dropout

    WHY DEFINED AT MODULE LEVEL (NOT INSIDE A FUNCTION):
    Python's pickle serialization mechanism requires top-level class definitions.
    If placed inside build_model(), its namespace is 'build_model.<locals>.NeuroScanCNN',
    causing pickle.dump() to crash with AttributeError.
    """

    def __init__(self, num_classes: int = 4, freeze_backbone: bool = False):
        super().__init__()
        
        # 1. Load Pretrained EfficientNet-B0
        # num_classes=0 strips off the default 1000-class ImageNet classification layer.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,      # Removes default classification head
        )
        
        # Optionally freeze all weights in the backbone for Phase 1
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # EfficientNet-B0 outputs 1280 feature channels at its deepest conv layer
        num_feat = self.backbone.num_features  # 1280

        # 2. Custom Classification Head:
        # Transforms the 3D spatial feature map (B, 1280, 7, 7) into 4 tumor class logits.
        self.head = nn.Sequential(
            # Spatial Reduction: Squeezes 7x7 spatial grid into a 1x1 summary per feature map
            nn.AdaptiveAvgPool2d(1),    # Output shape: (Batch, 1280, 1, 1)
            
            # Flatten: Flattens multi-dimensional tensor into a 1D vector per image
            nn.Flatten(),               # Output shape: (Batch, 1280)
            
            # Batch Normalization: Rescales incoming 1280 features to zero mean and unit variance.
            # Crucial during early epochs to prevent large gradients from destabilizing the head.
            nn.BatchNorm1d(num_feat),
            
            # Dense Layer 1: Projects 1280 features down to 256
            nn.Linear(num_feat, 256),
            nn.ReLU(inplace=True),      # Non-linear activation (replaces negative values with 0)
            
            # Regularization 1: Drops 40% of activations randomly to prevent neuron co-dependence
            nn.Dropout(p=0.4),
            
            # Dense Layer 2: Compresses 256 features down to 128
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            
            # Regularization 2: Drops 30% of activations randomly
            nn.Dropout(p=0.3),
            
            # Output Projection: 128 features -> 4 raw class logits
            # Note: No Softmax here because nn.CrossEntropyLoss applies LogSoftmax internally!
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward computation pass:
        Input: (Batch, 3, 224, 224) RGB MRI scans
        Output: (Batch, 4) unnormalized class logits
        """
        # forward_features() runs through all conv blocks and returns (B, 1280, 7, 7)
        x = self.backbone.forward_features(x)
        # Pass features through our custom classification head
        x = self.head(x)
        return x


def build_model(num_classes: int, freeze_backbone: bool = True) -> NeuroScanCNN:
    """
    Factory helper function to instantiate NeuroScanCNN.
    """
    return NeuroScanCNN(num_classes=num_classes, freeze_backbone=freeze_backbone)


# ==============================================================================
# STEP 3: TRAINING & VALIDATION HELPERS (AMP ENABLED)
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Executes one full training epoch (Forward pass -> Loss -> Backprop -> Optimizer step).

    Mixed Precision (AMP) Workflow:
    1. torch.autocast: Executes convolutions and matrix multiplies in fp16 (half-precision).
       This halves VRAM usage and speeds up computation on NVIDIA Tensor Cores.
    2. scaler.scale(loss).backward(): Scales loss by 65536 to prevent fp16 gradient underflow.
    3. scaler.step(optimizer): Unscales gradients, checks for Inf/NaN, then updates weights.
    4. scaler.update(): Dynamically adjusts scale factor for subsequent batch.
    """
    model.train()  # Enables Dropout and BatchNorm update mode
    running_loss = 0.0
    correct = total = 0

    for images, labels in tqdm(loader, leave=False, desc="  train"):
        # Transfer batch to GPU (non_blocking=True overlaps copy with GPU compute)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Clear gradients from previous iteration (set_to_none is faster than zero_grad)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass inside fp16 Autocast context
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)
            loss   = criterion(logits, labels)

        # Backpropagation with scaled loss
        scaler.scale(loss).backward()
        # Optimizer step via scaler
        scaler.step(optimizer)
        scaler.update()

        # Track statistics across the epoch
        running_loss += loss.item() * images.size(0)
        # argmax(dim=1) extracts predicted class index with highest logit score
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += images.size(0)

    # Return dataset-wide average loss and accuracy (0.0 to 1.0)
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluates the model on validation or testing data.
    @torch.no_grad() disables the computational graph (no gradient tape),
    drastically reducing memory overhead and accelerating evaluation.
    """
    model.eval()  # Disables Dropout and freezes BatchNorm running statistics
    running_loss = 0.0
    correct = total = 0

    for images, labels in tqdm(loader, leave=False, desc="  val  "):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass in fp16 for fast evaluation
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += images.size(0)

    return running_loss / total, correct / total


# ==============================================================================
# STEP 4: TRAINING LOOP WITH EARLY STOPPING & COSINE ANNEALING
# ==============================================================================

def run_training(model, train_loader, val_loader, device,
                 num_epochs, lr, patience, label, ckpt_path):
    """
    Runs an entire training phase with learning rate scheduling and early stopping.

    Loss Function:
    - CrossEntropyLoss with label_smoothing=0.1: Replaces hard one-hot targets [0, 0, 1, 0]
      with softened probabilities [0.025, 0.025, 0.925, 0.025]. This penalizes overconfidence.

    Optimizer:
    - AdamW with weight_decay=1e-4: Applies true L2 weight regularization to keep weights small.

    LR Scheduler:
    - CosineAnnealingLR: Gradually decays learning rate down to ~0 in a half-cosine curve.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimize only parameters that require gradients (ignores frozen backbone layers)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler    = torch.amp.GradScaler(device.type)

    best_val_acc = 0.0
    patience_ctr = 0
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  epochs={num_epochs}  lr={lr}  patience={patience}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        # 1. Train one epoch
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scaler, device)
        # 2. Validate on holdout set
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        
        # 3. Step the learning rate schedule
        scheduler.step()

        # Record metrics in history dictionary
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        flag = " ⬆ saved" if vl_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:02d}/{num_epochs}  "
            f"train [loss={tr_loss:.4f}  acc={tr_acc*100:.2f}%]  "
            f"val   [loss={vl_loss:.4f}  acc={vl_acc*100:.2f}%]{flag}"
        )

        # Early Stopping & Checkpoint Logic:
        # Save model checkpoint only when validation accuracy reaches a new best
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"  Best val accuracy: {best_val_acc * 100:.2f}%  → {ckpt_path.name}")
    return history


# ==============================================================================
# STEP 5: MAIN EXECUTION FLOW (PHASE 1 + PHASE 2)
# ==============================================================================

def main(args=None):
    # Parse command line flags
    parser = argparse.ArgumentParser(description="Train NeuroScan CNN")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--img-size",    type=int,   default=224)
    parser.add_argument("--lr-head",     type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--seed",        type=int,   default=42)
    cfg = parser.parse_args(args)

    # Set random seed for full reproducibility
    torch.manual_seed(cfg.seed)

    # Auto-detect CUDA hardware acceleration (GPU) or fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --------------------------------------------------------------------------
    # SUB-STEP 1: INITIALIZE DATALOADERS
    # --------------------------------------------------------------------------
    print("\nBuilding DataLoaders …")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset_dir=cfg.dataset_dir,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        num_workers=0,
    )
    print(f"  Classes : {meta['class_names']}")
    print(f"  Train   : {meta['train_size']:,}  |  Val : {meta['val_size']:,}  |  Test : {meta['test_size']:,}")

    # --------------------------------------------------------------------------
    # SUB-STEP 2: PHASE 1 TRAINING (HEAD ONLY, BACKBONE FROZEN)
    # --------------------------------------------------------------------------
    # EfficientNet weights are frozen. Only the new classification head is trained.
    # This prevents destructive weight updates to pretrained representations.
    print("\nBuilding model …")
    model = build_model(meta["num_classes"], freeze_backbone=True).to(device)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_p:,}")
    print(f"  Trainable params : {trainable_p:,}  ({trainable_p/total_p*100:.1f}%)")

    ckpt_p1    = MODELS_DIR / "best_phase1.pt"
    history_p1 = run_training(
        model, train_loader, val_loader, device,
        num_epochs=cfg.epochs, lr=cfg.lr_head,
        patience=cfg.patience,
        label="Phase 1 — Head Only (backbone frozen)",
        ckpt_path=ckpt_p1,
    )
    # Reload best weights achieved during Phase 1
    model.load_state_dict(torch.load(ckpt_p1, weights_only=True))
    print("Loaded best Phase-1 weights.")

    # --------------------------------------------------------------------------
    # SUB-STEP 3: PHASE 2 TRAINING (FINE-TUNING TOP 30% OF BACKBONE)
    # --------------------------------------------------------------------------
    # Unfreeze the top 30% deepest layers of EfficientNet-B0.
    # The bottom 70% remain frozen because low-level edges/corners are universal.
    backbone_params = list(model.backbone.named_parameters())
    freeze_until    = int(0.70 * len(backbone_params))
    for _, p in backbone_params:
        p.requires_grad = True
    for _, p in backbone_params[:freeze_until]:
        p.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 2 — unfroze top 30 % of backbone.")
    print(f"  Trainable params : {trainable_after:,}")

    ckpt_p2    = MODELS_DIR / "best_phase2.pt"
    history_p2 = run_training(
        model, train_loader, val_loader, device,
        num_epochs=10, lr=cfg.lr_finetune,
        patience=cfg.patience,
        label="Phase 2 — Fine-tune (top 30 % backbone + head)",
        ckpt_path=ckpt_p2,
    )
    # Reload best weights achieved during Phase 2
    model.load_state_dict(torch.load(ckpt_p2, weights_only=True))
    print("Loaded best Phase-2 weights.")

    # --------------------------------------------------------------------------
    # SUB-STEP 4: UNBIASED TEST EVALUATION
    # --------------------------------------------------------------------------
    print("\nFinal test-set evaluation …")
    test_loss, test_acc = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"  Test loss : {test_loss:.4f}")
    print(f"  Test acc  : {test_acc * 100:.2f}%")

    # --------------------------------------------------------------------------
    # SUB-STEP 5: PERSISTENCE (SAVE FINAL BUNDLE & METADATA)
    # --------------------------------------------------------------------------
    # 1. Complete PyTorch bundle: Includes state_dict + class names + normalization constants
    final_path = MODELS_DIR / "neuroscan_final.pt"
    torch.save({
        "model_state_dict" : model.state_dict(),
        "class_names"      : meta["class_names"],
        "img_size"         : meta["img_size"],
        "num_classes"      : meta["num_classes"],
        "backbone"         : "efficientnet_b0",
        "test_accuracy"    : round(float(test_acc), 4),
        "mean"             : meta["mean"],
        "std"              : meta["std"],
    }, final_path)
    print(f"\n✅ neuroscan_final.pt  → {final_path}")

    # 2. JSON Metadata: Enables inference scripts and microservices to read model properties
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "class_names"   : meta["class_names"],
            "img_size"      : meta["img_size"],
            "num_classes"   : meta["num_classes"],
            "backbone"      : "efficientnet_b0",
            "test_accuracy" : round(float(test_acc), 4),
            "mean"          : meta["mean"],
            "std"           : meta["std"],
        }, f, indent=2)
    print(f"✅ model_meta.json     → {meta_path}")

    # 3. Training History JSON: Stored for plotting loss & accuracy curves later
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({"phase1": history_p1, "phase2": history_p2}, f, indent=2)
    print(f"✅ training_history.json → {history_path}")

    print(f"\n{'='*60}")
    print(f"  Training complete!  Test accuracy: {test_acc * 100:.2f}%")
    print(f"  Models dir: {MODELS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

