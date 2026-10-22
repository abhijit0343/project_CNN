"""
pipeline.py
────────────────────────────────────────────────────────────────────────────────
End-to-end NeuroScan CNN pipeline.  Run this single file to go from raw MRI
images all the way to a saved, evaluated, and visualised model.

Execution order
───────────────
  Step 1 — preprocessing  : verify dataset, build DataLoaders, print split sizes
  Step 2 — train          : Phase-1 head training → Phase-2 fine-tuning
                            saves: best_phase1.pt, best_phase2.pt,
                                   neuroscan_final.pt, model_meta.json,
                                   training_history.json
  Step 3 — evaluate       : test-set metrics + all plots
                            saves: confusion_matrix.png, per_class_accuracy.png,
                                   training_curves.png, evaluation_report.txt
  Step 4 — gradcam        : Grad-CAM heatmap grid for one image per class
                            saves: gradcam_samples.png
  Step 5 — pickle export  : wraps model + metadata into a single .pkl bundle
                            saves: neuroscan_model.pkl

Why pickle in addition to .pt?
───────────────────────────────
PyTorch's ``torch.save`` (which uses pickle internally) is the recommended way
to persist models in a PyTorch-native ecosystem.  A separate ``pickle`` bundle
is useful when the downstream consumer (e.g. a Flask/FastAPI route, a Jupyter
notebook, or a non-PyTorch team member) wants to load the model with plain
Python and ``pickle.load`` without calling ``torch.load`` + ``build_model``.

The pickle bundle contains:
    {
        'model'        : the full nn.Module object (architecture + weights),
        'class_names'  : ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'img_size'     : 224,
        'num_classes'  : 4,
        'backbone'     : 'efficientnet_b0',
        'test_accuracy': 0.xxxx,
        'mean'         : [...],
        'std'          : [...],
    }

⚠️  Important: the pickle file embeds the model *class definition*.  Whoever
loads it must have the same Python + PyTorch + timm environment installed.
For cross-environment deployment prefer ``neuroscan_final.pt`` + ``build_model``.

Usage
─────
    python pipeline.py                    # full pipeline, default settings
    python pipeline.py --epochs 10        # faster run for testing
    python pipeline.py --skip-train       # evaluate + gradcam + pickle only
    python pipeline.py --no-plots        # skip all matplotlib figures
    python pipeline.py --samples 3       # 3 Grad-CAM samples per class
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

# ── Make src/ importable regardless of working directory ─────────────────────
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Local module imports ──────────────────────────────────────────────────────
from preprocessing import get_dataloaders          # noqa: E402
from train import build_model, run_training        # noqa: E402
from evaluate import (                             # noqa: E402
    evaluate,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_curves,
)
from gradcam import generate_gradcam_grid          # noqa: E402

MODELS_DIR  = SRC_DIR / "models"
DATASET_DIR = SRC_DIR / "dataset"
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Pretty logging helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(step: int, title: str):
    """Print a clearly visible step header."""
    print(f"\n{'━'*64}")
    print(f"  STEP {step} — {title}")
    print(f"{'━'*64}")


def _done(label: str, elapsed: float):
    print(f"  ✅ {label} complete  ({elapsed:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Pickle export
# ══════════════════════════════════════════════════════════════════════════════

def export_pickle(model: torch.nn.Module, meta: dict, save_dir: Path) -> Path:
    """
    Serialise the trained model + metadata into a single pickle file.

    What gets pickled
    ─────────────────
    We pickle the *entire nn.Module object* (not just the state_dict).
    This means the class definition (NeuroScanCNN) and all its weights are
    embedded.  The consumer can load it with:

        import pickle
        with open('neuroscan_model.pkl', 'rb') as f:
            bundle = pickle.load(f)
        model      = bundle['model']
        class_names = bundle['class_names']
        model.eval()

    Protocol
    ────────
    We use ``pickle.HIGHEST_PROTOCOL`` (protocol 5 on Python ≥ 3.8) which
    supports out-of-band buffers and is the most space-efficient option.

    Parameters
    ──────────
    model    : trained nn.Module (already on CPU for portability)
    meta     : dict with class_names, img_size, mean, std, test_accuracy, …
    save_dir : directory where the .pkl will be written

    Returns
    ───────
    Path to the saved pickle file
    """
    # Move to CPU before pickling — CUDA tensors can't be loaded on a machine
    # without a GPU if the pickle was created with CUDA tensors inside.
    model_cpu = model.cpu()

    bundle = {
        "model"        : model_cpu,
        "class_names"  : meta["class_names"],
        "img_size"     : meta["img_size"],
        "num_classes"  : meta["num_classes"],
        "backbone"     : meta.get("backbone", "efficientnet_b0"),
        "test_accuracy": meta.get("test_accuracy", None),
        "mean"         : meta.get("mean", [0.485, 0.456, 0.406]),
        "std"          : meta.get("std",  [0.229, 0.224, 0.225]),
    }

    pkl_path = save_dir / "neuroscan_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = pkl_path.stat().st_size / 1_048_576
    try:
        print(f"  [OK] Pickle saved -> {pkl_path}  ({size_mb:.1f} MB)")
    except Exception:
        pass
    return pkl_path


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg):
    pipeline_start = time.time()

    # ── Detect device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*64}")
    print(f"  🧠 NeuroScan CNN — Full Pipeline")
    print(f"{'═'*64}")
    print(f"  Device   : {device}")
    if device.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Models → : {MODELS_DIR}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Preprocessing
    # ──────────────────────────────────────────────────────────────────────────
    _banner(1, "Preprocessing — build DataLoaders")
    t0 = time.time()

    train_loader, val_loader, test_loader, data_meta = get_dataloaders(
        dataset_dir = cfg.dataset_dir,
        img_size    = cfg.img_size,
        batch_size  = cfg.batch_size,
        seed        = cfg.seed,
        num_workers = 0,   # 0 = safe on Windows and in notebooks
    )

    print(f"  Classes  : {data_meta['class_names']}")
    print(f"  Train    : {data_meta['train_size']:,}")
    print(f"  Val      : {data_meta['val_size']:,}")
    print(f"  Test     : {data_meta['test_size']:,}")
    _done("Preprocessing", time.time() - t0)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — Training  (skippable with --skip-train)
    # ──────────────────────────────────────────────────────────────────────────
    _banner(2, "Training — two-phase transfer learning")

    ckpt_p1 = MODELS_DIR / "best_phase1.pt"
    ckpt_p2 = MODELS_DIR / "best_phase2.pt"

    if cfg.skip_train:
        print("  --skip-train set: loading existing weights …")
        if not ckpt_p2.exists():
            print(f"  WARNING: {ckpt_p2} not found. Running training anyway.")
            cfg.skip_train = False

    if not cfg.skip_train:
        t0 = time.time()

        # ── Phase 1: backbone frozen, head only ───────────────────────────────
        import torch.nn as nn
        import torch.optim as optim

        model = build_model(data_meta["num_classes"], freeze_backbone=True).to(device)

        total_p     = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params     : {total_p:,}")
        print(f"  Trainable params : {trainable_p:,}  ({trainable_p/total_p*100:.1f}%)")

        history_p1 = run_training(
            model, train_loader, val_loader, device,
            num_epochs = cfg.epochs,
            lr         = cfg.lr_head,
            patience   = cfg.patience,
            label      = "Phase 1 — Head Only (backbone frozen)",
            ckpt_path  = ckpt_p1,
        )
        model.load_state_dict(torch.load(ckpt_p1, weights_only=True))
        print("  Loaded best Phase-1 weights.")

        # ── Phase 2: unfreeze top 30 % of backbone ────────────────────────────
        backbone_params = list(model.backbone.named_parameters())
        freeze_until    = int(0.70 * len(backbone_params))
        for _, p in backbone_params:
            p.requires_grad = True
        for _, p in backbone_params[:freeze_until]:
            p.requires_grad = False

        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Phase 2 — unfroze top 30 % of backbone.")
        print(f"  Trainable params : {trainable_after:,}")

        history_p2 = run_training(
            model, train_loader, val_loader, device,
            num_epochs = cfg.finetune_epochs,
            lr         = cfg.lr_finetune,
            patience   = cfg.patience,
            label      = "Phase 2 — Fine-tune (top 30 % backbone + head)",
            ckpt_path  = ckpt_p2,
        )
        model.load_state_dict(torch.load(ckpt_p2, weights_only=True))
        print("  Loaded best Phase-2 weights.")

        # ── Quick test eval to get test_accuracy for metadata ─────────────────
        print("\n  Quick test-set eval …")
        criterion  = nn.CrossEntropyLoss()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(device)
                correct += (model(imgs).argmax(1) == lbls.to(device)).sum().item()
                total   += imgs.size(0)
        quick_test_acc = correct / total
        print(f"  Test accuracy    : {quick_test_acc * 100:.2f}%")

        # ── Save full model bundle (.pt) ──────────────────────────────────────
        final_path = MODELS_DIR / "neuroscan_final.pt"
        torch.save(
            {
                "model_state_dict" : model.state_dict(),
                "class_names"      : data_meta["class_names"],
                "img_size"         : data_meta["img_size"],
                "num_classes"      : data_meta["num_classes"],
                "backbone"         : "efficientnet_b0",
                "test_accuracy"    : round(float(quick_test_acc), 4),
                "mean"             : data_meta["mean"],
                "std"              : data_meta["std"],
            },
            final_path,
        )
        print(f"  ✅ neuroscan_final.pt saved → {final_path}")

        # ── Save JSON metadata ────────────────────────────────────────────────
        meta_path = MODELS_DIR / "model_meta.json"
        model_meta = {
            "class_names"   : data_meta["class_names"],
            "img_size"      : data_meta["img_size"],
            "num_classes"   : data_meta["num_classes"],
            "backbone"      : "efficientnet_b0",
            "test_accuracy" : round(float(quick_test_acc), 4),
            "mean"          : data_meta["mean"],
            "std"           : data_meta["std"],
        }
        with open(meta_path, "w") as f:
            json.dump(model_meta, f, indent=2)
        print(f"  ✅ model_meta.json saved → {meta_path}")

        # ── Save training history ─────────────────────────────────────────────
        history_path = MODELS_DIR / "training_history.json"
        with open(history_path, "w") as f:
            json.dump({"phase1": history_p1, "phase2": history_p2}, f, indent=2)
        print(f"  ✅ training_history.json saved → {history_path}")

        _done("Training", time.time() - t0)

    else:
        # Load existing model + meta for downstream steps
        print("  Loading neuroscan_final.pt …")
        ckpt = torch.load(MODELS_DIR / "neuroscan_final.pt",
                          map_location=device, weights_only=True)
        model = build_model(ckpt["num_classes"], freeze_backbone=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model_meta = {k: ckpt[k] for k in
                      ("class_names", "img_size", "num_classes",
                       "backbone", "test_accuracy", "mean", "std")}
        print("  Weights loaded ✓")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Evaluate
    # ──────────────────────────────────────────────────────────────────────────
    _banner(3, "Evaluate — test-set metrics & plots")
    t0 = time.time()

    results = evaluate(
        model_path  = MODELS_DIR / "neuroscan_final.pt",
        dataset_dir = cfg.dataset_dir,
        device      = device,
    )

    if not cfg.no_plots:
        plot_confusion_matrix(results["cm"],    results["class_names"], MODELS_DIR)
        plot_per_class_accuracy(results["cm"],  results["class_names"], MODELS_DIR)
        plot_training_curves(MODELS_DIR / "training_history.json", MODELS_DIR)

    # Save evaluation report text file
    report_path = MODELS_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy : {results['test_acc'] * 100:.2f}%\n\n")
        f.write(results["report"])
    print(f"  ✅ evaluation_report.txt saved → {report_path}")

    _done("Evaluate", time.time() - t0)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — Grad-CAM
    # ──────────────────────────────────────────────────────────────────────────
    _banner(4, "Grad-CAM — heatmap visualisations")
    t0 = time.time()

    if not cfg.no_plots:
        generate_gradcam_grid(
            model_path        = MODELS_DIR / "neuroscan_final.pt",
            test_dir          = DATASET_DIR / "Testing",
            samples_per_class = cfg.samples,
            device            = device,
            save_dir          = MODELS_DIR,
        )
    else:
        print("  --no-plots set: skipping Grad-CAM visualisation.")

    _done("Grad-CAM", time.time() - t0)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Pickle export
    # ──────────────────────────────────────────────────────────────────────────
    _banner(5, "Pickle export — neuroscan_model.pkl")
    t0 = time.time()

    # Read metadata from the JSON (works whether we trained or skipped)
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path) as f:
        pkl_meta = json.load(f)

    # Ensure model is loaded with best weights
    final_ckpt = torch.load(
        MODELS_DIR / "neuroscan_final.pt",
        map_location="cpu",    # CPU for portability of the pickle
        weights_only=True,
    )
    model_for_pkl = build_model(pkl_meta["num_classes"], freeze_backbone=False)
    model_for_pkl.load_state_dict(final_ckpt["model_state_dict"])
    model_for_pkl.eval()

    pkl_path = export_pickle(model_for_pkl, pkl_meta, MODELS_DIR)
    _done("Pickle export", time.time() - t0)

    # ──────────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    mins, secs    = divmod(int(total_elapsed), 60)

    print(f"\n{'═'*64}")
    print(f"  🎉 Pipeline complete!  Total time: {mins}m {secs}s")
    print(f"{'═'*64}")
    print(f"  Classes      : {results['class_names']}")
    print(f"  Test accuracy: {results['test_acc'] * 100:.2f}%")
    print(f"\n  Saved files in {MODELS_DIR}:")

    files_to_report = [
        ("neuroscan_final.pt",      "Full model bundle (state_dict + meta)"),
        ("best_phase1.pt",          "Best Phase-1 checkpoint"),
        ("best_phase2.pt",          "Best Phase-2 checkpoint"),
        ("model_meta.json",         "Class names, img_size, mean/std"),
        ("training_history.json",   "Per-epoch loss & accuracy"),
        ("neuroscan_model.pkl",     "🥒 Pickle bundle (model + meta)"),
        ("evaluation_report.txt",   "Classification report"),
        ("confusion_matrix.png",    "Confusion matrix heatmap"),
        ("per_class_accuracy.png",  "Per-class accuracy bar chart"),
        ("training_curves.png",     "Loss & accuracy curves"),
        ("gradcam_samples.png",     "Grad-CAM heatmap overlays"),
    ]
    for fname, desc in files_to_report:
        p = MODELS_DIR / fname
        tag = "✅" if p.exists() else "⬜"
        print(f"    {tag} {fname:<30} {desc}")

    print(f"{'═'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NeuroScan CNN — end-to-end pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir",     default=None,
                        help="Path to folder containing Training/ and Testing/")
    parser.add_argument("--epochs",          type=int,   default=20,
                        help="Phase-1 training epochs")
    parser.add_argument("--finetune-epochs", type=int,   default=10,
                        help="Phase-2 fine-tuning epochs")
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--img-size",        type=int,   default=224)
    parser.add_argument("--lr-head",         type=float, default=1e-3,
                        help="Phase-1 learning rate (head only)")
    parser.add_argument("--lr-finetune",     type=float, default=1e-4,
                        help="Phase-2 learning rate (fine-tune)")
    parser.add_argument("--patience",        type=int,   default=5,
                        help="Early-stopping patience in epochs")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--samples",         type=int,   default=1,
                        help="Grad-CAM samples per class")
    parser.add_argument("--skip-train",      action="store_true",
                        help="Skip training; load existing model weights")
    parser.add_argument("--no-plots",        action="store_true",
                        help="Skip all matplotlib figure generation")
    cfg = parser.parse_args()

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
