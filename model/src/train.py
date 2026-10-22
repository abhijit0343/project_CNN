"""
train.py
────────────────────────────────────────────────────────────────────────────────
Trains the NeuroScan CNN (EfficientNet-B0 backbone) on Brain Tumour MRI images
using a two-phase transfer-learning strategy with mixed precision (AMP).

Transfer learning strategy
──────────────────────────
  Phase 1 — head-only training (backbone frozen)
      EfficientNet-B0 is loaded with pretrained ImageNet weights and all its
      parameters are frozen (requires_grad = False).  Only the custom
      classification head is trained.  This lets the head quickly adapt to the
      medical-image feature space without destroying pretrained representations.

  Phase 2 — fine-tuning (top 30 % of backbone unfrozen)
      The best Phase-1 checkpoint is reloaded.  The bottom 70 % of backbone
      layers remain frozen; the top 30 % are thawed and trained with the head
      at a 10× smaller learning rate.  Cosine annealing then smoothly decays
      the LR to near-zero, refining high-level MRI-specific feature detectors.

Mixed Precision (AMP)
─────────────────────
  torch.autocast + torch.amp.GradScaler are used throughout.
  fp16 forward/backward passes cut VRAM usage ~2× and speed up Tensor Core ops.
  GradScaler dynamically scales the loss to prevent fp16 gradient underflow,
  then descales before the optimizer step.

Outputs saved to models/
─────────────────────────
  best_phase1.pt        — best val-accuracy weights from Phase 1
  best_phase2.pt        — best val-accuracy weights from Phase 2
  neuroscan_final.pt    — full bundle: state_dict + metadata
  model_meta.json       — class_names, img_size, mean/std for inference
  training_history.json — per-epoch loss & accuracy for both phases

Usage
─────
    python train.py
    python train.py --epochs 30 --batch-size 16
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import get_dataloaders   # noqa: E402

MODELS_DIR = SRC_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Model definition
# ══════════════════════════════════════════════════════════════════════════════

import timm  # imported at module level so NeuroScanCNN is picklable


class NeuroScanCNN(nn.Module):
    """
    EfficientNet-B0 backbone + custom classification head for brain tumour MRI.

    WHY THIS CLASS IS AT MODULE LEVEL (not inside build_model)
    ──────────────────────────────────────────────────────────
    Python's pickle module serialises objects by storing their class's
    *fully-qualified import path* (e.g. ``train.NeuroScanCNN``) rather than
    the class source code.  When you later call pickle.load(), Python imports
    that path to reconstruct the object.

    If the class is defined inside a function (a "local class"), its path
    is ``train.build_model.<locals>.NeuroScanCNN`` — not importable at module
    scope — so pickle raises:
        AttributeError: Can't pickle local object 'build_model.<locals>.NeuroScanCNN'

    Moving the class here (module level) gives it the importable path
    ``train.NeuroScanCNN``, which pickle can resolve correctly.

    Architecture
    ────────────
    Backbone — EfficientNet-B0 (pretrained ImageNet, head removed):
      forward_features() returns a spatial map (B, 1280, H', W').

    Custom classification head:
      AdaptiveAvgPool2d(1)       → (B, 1280, 1, 1)  global average pool
      Flatten()                  → (B, 1280)
      BatchNorm1d(1280)          stabilises feature magnitudes early in training
      Linear(1280 → 256) + ReLU
      Dropout(0.4)               prevents co-adaptation of neurons
      Linear(256  → 128) + ReLU
      Dropout(0.3)
      Linear(128  → num_classes) raw logits — CrossEntropyLoss handles softmax
    """

    def __init__(self, num_classes: int = 4, freeze_backbone: bool = False):
        super().__init__()
        # Load EfficientNet-B0 without its 1000-class ImageNet head.
        # pretrained=True downloads weights from the timm model hub on first run.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,      # strips the default classification head
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        num_feat = self.backbone.num_features  # 1280 for EfficientNet-B0

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (B, 1280, H', W') → (B, 1280, 1, 1)
            nn.Flatten(),               # → (B, 1280)
            nn.BatchNorm1d(num_feat),   # normalise feature distribution
            nn.Linear(num_feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),  # raw logits, no softmax
        )

    def forward(self, x):
        # forward_features() skips the default head and returns the
        # spatial feature map (B, 1280, H', W') from the last conv block.
        x = self.backbone.forward_features(x)
        x = self.head(x)
        return x


def build_model(num_classes: int, freeze_backbone: bool = True) -> NeuroScanCNN:
    """
    Convenience factory — returns a NeuroScanCNN instance.

    Parameters
    ──────────
    num_classes     : number of output classes (4 for this dataset)
    freeze_backbone : freeze backbone weights for Phase-1 head-only training
    """
    return NeuroScanCNN(num_classes=num_classes, freeze_backbone=freeze_backbone)


# ══════════════════════════════════════════════════════════════════════════════
# Per-epoch helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Run one complete forward + backward pass over the training DataLoader.

    AMP details
    ───────────
    torch.autocast wraps the forward pass:
      - Eligible ops (conv, gemm) run in fp16 → ~2× throughput on Tensor Cores.
      - Sensitive ops (softmax, loss) stay in fp32 for numerical stability.
    scaler.scale(loss).backward() multiplies the loss by the GradScaler's
      scale factor so fp16 gradients don't underflow to zero.
    scaler.step(optimizer) descales the gradients, checks for inf/nan, then
      calls optimizer.step() only if gradients are finite.
    scaler.update() adjusts the scale factor for the next iteration.

    Returns: (avg_loss, accuracy) — both floats
    """
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in tqdm(loader, leave=False, desc="  train"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test data.

    @torch.no_grad() disables the autograd tape for this entire function —
    no gradients are stored, saving memory and accelerating inference.

    Returns: (avg_loss, accuracy) — both floats
    """
    model.eval()
    running_loss = 0.0
    correct = total = 0

    for images, labels in tqdm(loader, leave=False, desc="  val  "):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += images.size(0)

    return running_loss / total, correct / total


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_training(model, train_loader, val_loader, device,
                 num_epochs, lr, patience, label, ckpt_path):
    """
    Full training loop with early stopping, cosine-annealing LR, and AMP.

    Early stopping
    ──────────────
    If val accuracy fails to improve for `patience` consecutive epochs the loop
    exits and the best checkpoint on disk is preserved unchanged.

    LR schedule — CosineAnnealingLR
    ────────────────────────────────
    Smoothly anneals LR from `lr` down to ~0 over T_max=num_epochs steps.
    This avoids overshooting sharp minima near the end of training.

    Loss — CrossEntropyLoss with label_smoothing=0.1
    ─────────────────────────────────────────────────
    Hard one-hot labels (0/1) are replaced with (0.025/0.925).
    This penalises overconfident predictions and improves calibration.

    Returns: history dict with lists for train_loss, val_loss, train_acc, val_acc
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scaler, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

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

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    print(f"  Best val accuracy: {best_val_acc * 100:.2f}%  → {ckpt_path.name}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
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

    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── DataLoaders ───────────────────────────────────────────────────────────
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

    # ── Phase 1: head only ────────────────────────────────────────────────────
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
    model.load_state_dict(torch.load(ckpt_p1, weights_only=True))
    print("Loaded best Phase-1 weights.")

    # ── Phase 2: unfreeze top 30 % of backbone ────────────────────────────────
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
    model.load_state_dict(torch.load(ckpt_p2, weights_only=True))
    print("Loaded best Phase-2 weights.")

    # ── Test accuracy ─────────────────────────────────────────────────────────
    print("\nFinal test-set evaluation …")
    test_loss, test_acc = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"  Test loss : {test_loss:.4f}")
    print(f"  Test acc  : {test_acc * 100:.2f}%")

    # ── Save final model bundle ────────────────────────────────────────────────
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

    # ── JSON metadata ─────────────────────────────────────────────────────────
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

    # ── Training history ──────────────────────────────────────────────────────
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
