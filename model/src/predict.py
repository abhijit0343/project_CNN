"""
predict.py
────────────────────────────────────────────────────────────────────────────────
Single-image inference for the NeuroScan CNN.

Used by the FastAPI backend to serve predictions over HTTP, and can also be
run directly from the command line for quick testing.

Inference pipeline
──────────────────
  1. Load image from disk (or accept a PIL Image).
  2. Apply the same deterministic val_transform used during training:
       Resize → ToTensor → Normalise with ImageNet mean/std.
  3. Run the model in eval mode inside torch.no_grad() (no autograd tape).
  4. Apply softmax to logits to get class probabilities.
  5. Return the top-1 class name + confidence, and all class scores.

Why softmax?
────────────
The model outputs raw logits (un-normalised scores).  Softmax converts them
to a probability distribution: values in (0, 1) that sum to 1, making the
output interpretable as model confidence.

Usage
─────
    python predict.py path/to/scan.jpg
    python predict.py path/to/scan.jpg --top-k 4
    python predict.py path/to/scan.jpg --json

    # From Python / FastAPI:
    from predict import Predictor
    p = Predictor()
    result = p.predict("scan.jpg")
    # {'label': 'glioma', 'confidence': 0.983, 'scores': {...}}
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Local imports ─────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import build_model   # noqa: E402

MODELS_DIR = SRC_DIR / "models"


# ══════════════════════════════════════════════════════════════════════════════
# Predictor class
# ══════════════════════════════════════════════════════════════════════════════

class Predictor:
    """
    Wraps the NeuroScan CNN for efficient repeated inference.

    Loading a model involves reading bytes from disk, allocating GPU memory,
    and copying the state dict — all expensive.  By encapsulating the model
    inside a class, we load it once at construction time and reuse it for
    every call to predict().  This is the standard pattern for model serving.

    Parameters
    ──────────
    model_path : .pt file saved by train.py (default: models/neuroscan_final.pt)
    device     : 'cuda', 'cpu', or torch.device.  Auto-detected if None.
    """

    def __init__(self, model_path=None, device=None, verbose=True):
        if model_path is None:
            # Default to pickle if available, otherwise .pt
            pkl_file = MODELS_DIR / "neuroscan_model.pkl"
            pt_file  = MODELS_DIR / "neuroscan_final.pt"
            model_path = pkl_file if pkl_file.exists() else pt_file
        
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute() and not self.model_path.exists():
            self.model_path = MODELS_DIR / model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run train.py or pipeline.py first to produce a saved model."
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # ── Load model from Pickle (.pkl) or PyTorch (.pt) ────────────────────
        if self.model_path.suffix == ".pkl":
            import pickle
            with open(self.model_path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"].to(self.device)
            self.class_names = bundle["class_names"]
            num_classes      = bundle.get("num_classes", len(self.class_names))
            img_size         = bundle.get("img_size", 224)
            mean             = bundle.get("mean", [0.485, 0.456, 0.406])
            std              = bundle.get("std",  [0.229, 0.224, 0.225])
            self.model.eval()
        else:
            checkpoint = torch.load(self.model_path, map_location=self.device,
                                    weights_only=True)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict       = checkpoint["model_state_dict"]
                self.class_names = checkpoint["class_names"]
                num_classes      = checkpoint["num_classes"]
                img_size         = checkpoint["img_size"]
                mean             = checkpoint.get("mean", [0.485, 0.456, 0.406])
                std              = checkpoint.get("std",  [0.229, 0.224, 0.225])
            else:
                # Raw state_dict — read metadata from model_meta.json
                state_dict = checkpoint
                meta_path  = MODELS_DIR / "model_meta.json"
                if not meta_path.exists():
                    raise FileNotFoundError(
                        f"model_meta.json not found in {MODELS_DIR}.\n"
                        "Use the full neuroscan_final.pt bundle or ensure model_meta.json exists."
                    )
                with open(meta_path) as f:
                    meta = json.load(f)
                self.class_names = meta["class_names"]
                num_classes      = meta["num_classes"]
                img_size         = meta["img_size"]
                mean             = meta.get("mean", [0.485, 0.456, 0.406])
                std              = meta.get("std",  [0.229, 0.224, 0.225])

            # ── Build model and load weights ──────────────────────────────────
            self.model = build_model(num_classes, freeze_backbone=False).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        # ── Preprocessing transform ───────────────────────────────────────────
        # Must exactly match val_transform from preprocessing.py:
        #   Resize → ToTensor → Normalise
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if verbose:
            print(
                f"Predictor ready  device={self.device}  "
                f"classes={self.class_names}  model={self.model_path.name}"
            )

    def predict(self, image, top_k: int = 1) -> dict:
        """
        Predict the tumour class for a single MRI image.

        Parameters
        ──────────
        image : file path (str / Path) or an already-opened PIL Image
        top_k : how many top predictions to include in the result

        Returns
        ───────
        dict with:
            'label'      str   — top-1 predicted class name
            'confidence' float — softmax probability for top-1 class (0–1)
            'scores'     dict  — {class_name: probability} for all classes
            'top_k'      list  — [{'label': ..., 'confidence': ...}] x top_k
        """
        # ── Load image ────────────────────────────────────────────────────────
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected file path or PIL Image, got {type(image)}")

        # unsqueeze(0): (3, H, W) → (1, 3, H, W) adds the batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # ── Inference ─────────────────────────────────────────────────────────
        # torch.no_grad() prevents PyTorch from recording operations into the
        # autograd graph — no gradient tape is built, saving memory and time.
        with torch.no_grad():
            logits = self.model(tensor)                  # (1, num_classes)
            probs  = F.softmax(logits, dim=1).squeeze()  # (num_classes,)

        probs_np = probs.cpu().numpy()

        # Top-1
        top1_idx  = int(probs_np.argmax())
        top1_name = self.class_names[top1_idx]
        top1_conf = float(probs_np[top1_idx])

        # All class scores
        all_scores = {
            name: round(float(p), 6)
            for name, p in zip(self.class_names, probs_np)
        }

        # Top-k sorted by confidence descending
        sorted_idx = probs_np.argsort()[::-1]
        top_k_list = [
            {"label": self.class_names[i], "confidence": round(float(probs_np[i]), 6)}
            for i in sorted_idx[:top_k]
        ]

        return {
            "label"      : top1_name,
            "confidence" : round(top1_conf, 6),
            "scores"     : all_scores,
            "top_k"      : top_k_list,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Convenience one-shot function
# ══════════════════════════════════════════════════════════════════════════════

def predict(model_path, image_path) -> dict:
    """
    One-shot prediction (loads model fresh each call).

    For repeated inference prefer instantiating a Predictor once:
        p = Predictor()
        for img in images:
            print(p.predict(img))
    """
    return Predictor(model_path=model_path).predict(image_path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    parser = argparse.ArgumentParser(description="Single-image inference — NeuroScan CNN")
    parser.add_argument("image",             help="Path to MRI image file")
    parser.add_argument("--model",           default=None,
                        help="Model filename or path (defaults to neuroscan_model.pkl or neuroscan_final.pt)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--json",  action="store_true",
                        help="Print output as JSON")
    cfg = parser.parse_args(args)

    predictor = Predictor(model_path=cfg.model, verbose=not cfg.json)
    result    = predictor.predict(cfg.image, top_k=cfg.top_k)

    if cfg.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'─'*44}")
        print(f"  Image      : {cfg.image}")
        print(f"  Prediction : {result['label']}")
        print(f"  Confidence : {result['confidence'] * 100:.2f}%")
        print(f"\n  Top-{cfg.top_k} predictions:")
        for rank, item in enumerate(result["top_k"], start=1):
            bar_len = int(item["confidence"] * 32)
            bar     = "█" * bar_len + "░" * (32 - bar_len)
            print(f"    {rank}. {item['label']:<14} {bar} {item['confidence']*100:.2f}%")
        print(f"{'─'*44}\n")


if __name__ == "__main__":
    main()
