# ==============================================================================
# evaluate.py: Model Evaluation, Statistical Reporting, & Metric Visualization
# ==============================================================================
# This module loads a trained NeuroScan CNN checkpoint and evaluates its performance
# on the holdout test set (unseen patient MRI scans).
#
# Generated Outputs (saved into models/):
# 1. evaluation_report.txt  - Precision, Recall, F1-Score, and Support per class
# 2. confusion_matrix.png   - Visual heatmap comparing Ground Truth vs Predictions
# 3. per_class_accuracy.png - Bar chart of per-class detection accuracy (Recall)
# 4. training_curves.png    - Side-by-side plots of Loss & Accuracy across both phases
# ==============================================================================

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ------------------------------------------------------------------------------
# STEP 1: IMPORT LOCAL MODULES
# ------------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import get_dataloaders   # noqa: E402
from train import build_model              # noqa: E402

MODELS_DIR = SRC_DIR / "models"


# ==============================================================================
# STEP 2: CORE EVALUATION FUNCTION
# ==============================================================================

def evaluate(model_path, dataset_dir=None, device=None):
    """
    Loads saved model weights and computes rigorous metrics on the test dataset.

    Metric Explanations in Medical Context:
    - Accuracy:  Overall fraction of correct predictions across all 4 classes.
    - Precision: Out of all scans predicted as tumor X, how many were actually X?
                 (High Precision = Few False Alarms / False Positives).
    - Recall:    Out of all patients who genuinely had tumor X, how many were identified?
                 (High Recall = Few Missed Cases / False Negatives — critical in oncology).
    - F1-Score:  Harmonic mean of Precision and Recall.
    - Macro Avg: Simple average of metrics across all classes (treats rare tumors equally).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Auto-detect GPU acceleration
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice  : {device}")
    print(f"Model   : {model_path}")

    # --------------------------------------------------------------------------
    # CHECKPOINT LOADING & METADATA EXTRACTION
    # --------------------------------------------------------------------------
    # weights_only=True prevents arbitrary code execution vulnerabilities during unpickling
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Checkpoint is the complete neuroscan_final.pt bundle
        state_dict  = checkpoint["model_state_dict"]
        class_names = checkpoint["class_names"]
        num_classes = checkpoint["num_classes"]
        img_size    = checkpoint["img_size"]
    else:
        # Checkpoint is a raw state_dict (e.g., best_phase1.pt / best_phase2.pt)
        # Fall back to reading model metadata from model_meta.json
        state_dict = checkpoint
        meta_path  = MODELS_DIR / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"model_meta.json not found in {MODELS_DIR}.\n"
                "Run train.py first, or use the full neuroscan_final.pt bundle."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        class_names = meta["class_names"]
        num_classes = meta["num_classes"]
        img_size    = meta["img_size"]

    print(f"Classes : {class_names}")

    # --------------------------------------------------------------------------
    # MODEL INSTANTIATION & WEIGHT INJECTION
    # --------------------------------------------------------------------------
    # Recreate the exact EfficientNet-B0 architecture and inject learned parameters
    model = build_model(num_classes, freeze_backbone=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode (Dropout turned OFF, BatchNorm frozen)
    print("Weights loaded ✓")

    # --------------------------------------------------------------------------
    # TEST DATALOADER INITIALIZATION
    # --------------------------------------------------------------------------
    print("Building test DataLoader …")
    _, _, test_loader, _ = get_dataloaders(
        dataset_dir=dataset_dir, img_size=img_size,
        batch_size=32, num_workers=0,
    )

    # --------------------------------------------------------------------------
    # INFERENCE LOOP ACROSS TEST SET
    # --------------------------------------------------------------------------
    # torch.no_grad() disables PyTorch's autograd engine, saving memory
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            # Forward pass: argmax(dim=1) picks index of the highest logit score
            preds  = model(images).argmax(dim=1).cpu().numpy()
            y_pred_list.extend(preds)
            y_true_list.extend(labels.numpy())

    # Convert lists to NumPy arrays for scikit-learn metrics
    y_true   = np.array(y_true_list)
    y_pred   = np.array(y_pred_list)
    test_acc = (y_true == y_pred).mean()
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(y_true, y_pred,
                                     target_names=class_names, digits=4)

    print(f"\n{'─'*56}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"{'─'*56}")
    print(report)

    return {
        "y_true"      : y_true,
        "y_pred"      : y_pred,
        "test_acc"    : float(test_acc),
        "class_names" : class_names,
        "report"      : report,
        "cm"          : cm,
    }


# ==============================================================================
# STEP 3: VISUALIZATION HELPERS
# ==============================================================================

def plot_confusion_matrix(cm, class_names, save_dir: Path):
    """
    Renders and saves a color-coded confusion matrix heatmap.
    - Diagonal cells (top-left to bottom-right): Correct predictions.
    - Off-diagonal cells: Misclassifications.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # annot=True prints integer count in each box, cmap="Blues" highlights high values
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Label (Ground Truth)", fontsize=11)
    ax.set_xlabel("Predicted Label (Model)",   fontsize=11)
    plt.tight_layout()
    out_path = save_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Confusion matrix saved → {out_path}")


def plot_per_class_accuracy(cm, class_names, save_dir: Path):
    """
    Computes per-class recall:
        Per-Class Accuracy = (True Positives in Class) / (Total Actual Scans in Class)
    Identifies which tumor varieties (e.g., Glioma vs Meningioma) are most challenging.
    """
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, per_class_acc * 100, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Accuracy — Test Set", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = save_dir / "per_class_accuracy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Per-class accuracy saved → {out_path}")


def plot_training_curves(history_path: Path, save_dir: Path):
    """
    Visualizes learning dynamics over time by plotting Loss and Accuracy curves.
    A vertical dotted line marks the transition from Phase-1 (warmup) to Phase-2 (fine-tuning).
    """
    if not history_path.exists():
        print(f"  (skipping training curves — {history_path} not found)")
        return

    with open(history_path) as f:
        h = json.load(f)

    # Stitch Phase 1 and Phase 2 history sequences together
    p1, p2  = h["phase1"], h["phase2"]
    p1_len  = len(p1["train_acc"])
    acc     = p1["train_acc"]  + p2["train_acc"]
    vacc    = p1["val_acc"]    + p2["val_acc"]
    loss    = p1["train_loss"] + p2["train_loss"]
    vloss   = p1["val_loss"]   + p2["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, tr, vl, ylabel, title in zip(
        axes,
        [acc,  loss],
        [vacc, vloss],
        ["Accuracy", "Loss"],
        ["Accuracy vs Epoch", "Loss vs Epoch"],
    ):
        ax.plot(tr, label="Train",      linewidth=2)
        ax.plot(vl, label="Validation", linewidth=2, linestyle="--")
        # Divider line between Phase 1 and Phase 2
        ax.axvline(p1_len - 0.5, color="grey", linestyle=":", linewidth=1.5,
                   label="Fine-tune start")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("NeuroScan CNN — Training History", fontsize=15)
    plt.tight_layout()
    out_path = save_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Training curves saved → {out_path}")


# ==============================================================================
# STEP 4: CLI ENTRY POINT
# ==============================================================================

def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate NeuroScan CNN")
    parser.add_argument("--model",       default="neuroscan_final.pt")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--no-plots",    action="store_true")
    cfg = parser.parse_args(args)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate(MODELS_DIR / cfg.model,
                       dataset_dir=cfg.dataset_dir, device=device)

    # Generate charts unless explicitly disabled
    if not cfg.no_plots:
        plot_confusion_matrix(results["cm"],   results["class_names"], MODELS_DIR)
        plot_per_class_accuracy(results["cm"], results["class_names"], MODELS_DIR)
        plot_training_curves(MODELS_DIR / "training_history.json", MODELS_DIR)

    # Save summary text report to disk
    report_path = MODELS_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model      : {cfg.model}\n")
        f.write(f"Test Acc   : {results['test_acc'] * 100:.2f}%\n\n")
        f.write(results["report"])
    print(f"✅ Text report saved → {report_path}")


if __name__ == "__main__":
    main()

