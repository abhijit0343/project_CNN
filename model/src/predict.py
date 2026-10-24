# ==============================================================================
# predict.py: Production Inference Engine for Single-Scan MRI Classification
# ==============================================================================
# This module provides the 'Predictor' class used by both the command-line
# interface and the FastAPI microservice to classify individual Brain MRI scans.
#
# Key Steps:
# 1. Singleton Caching: Loads weights once into GPU/CPU RAM for sub-50ms inference.
# 2. Preprocessing: Resizes to 224x224 and applies ImageNet mean/std normalisation.
# 3. Feedforward: Evaluates scan through EfficientNet-B0 inside torch.no_grad().
# 4. Probability Calibration: Converts raw logits into calibrated confidence scores via Softmax.
# ==============================================================================

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ------------------------------------------------------------------------------
# STEP 1: IMPORT LOCAL MODULES
# ------------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import build_model   # noqa: E402

MODELS_DIR = SRC_DIR / "models"


# ==============================================================================
# STEP 2: PREDICTOR WRAPPER CLASS
# ==============================================================================

class Predictor:
    """
    Wraps the NeuroScan CNN model for fast, repeated inference.

    Design Pattern:
    Instantiating this class loads the weights once into memory.
    Subsequent calls to predict() reuse the loaded model, eliminating file I/O latency.
    """

    def __init__(self, model_path=None, device=None, verbose=True):
        # ----------------------------------------------------------------------
        # RESOLVE MODEL WEIGHT PATH
        # ----------------------------------------------------------------------
        if model_path is None:
            # Check for standalone pickle bundle first; fallback to PyTorch .pt bundle
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

        # Detect hardware acceleration (CUDA GPU or CPU fallback)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # ----------------------------------------------------------------------
        # LOAD MODEL: PICKLE (.pkl) OR PYTORCH (.pt)
        # ----------------------------------------------------------------------
        if self.model_path.suffix == ".pkl":
            # Load full serialized model object from Python pickle
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
            # Load PyTorch checkpoint dictionary
            checkpoint = torch.load(self.model_path, map_location=self.device,
                                    weights_only=True)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Full bundle containing state_dict and embedded metadata
                state_dict       = checkpoint["model_state_dict"]
                self.class_names = checkpoint["class_names"]
                num_classes      = checkpoint["num_classes"]
                img_size         = checkpoint["img_size"]
                mean             = checkpoint.get("mean", [0.485, 0.456, 0.406])
                std              = checkpoint.get("std",  [0.229, 0.224, 0.225])
            else:
                # Raw state_dict — read metadata from companion model_meta.json
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

            # Reconstruct architecture and load weights
            self.model = build_model(num_classes, freeze_backbone=False).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        # ----------------------------------------------------------------------
        # DETERMINISTIC PREPROCESSING TRANSFORM
        # ----------------------------------------------------------------------
        # Must match val_tf from preprocessing.py: No random crops or rotations!
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),   # Resize scan to 224x224
            transforms.ToTensor(),                     # [0, 255] -> [0.0, 1.0] Tensor (C, H, W)
            transforms.Normalize(mean=mean, std=std),  # Zero-mean unit-variance
        ])

        if verbose:
            print(
                f"Predictor ready  device={self.device}  "
                f"classes={self.class_names}  model={self.model_path.name}"
            )

    def predict(self, image, top_k: int = 1) -> dict:
        """
        Classifies an MRI scan and returns probabilities for all 4 tumor types.

        Parameters:
        - image: File path (str/Path) or an open PIL Image object
        - top_k: Number of highest-confidence predictions to return

        Returns:
        - dict with 'label', 'confidence', 'scores', and 'top_k' list
        """
        # ----------------------------------------------------------------------
        # 1. INPUT VALIDATION & RGB CONVERSION
        # ----------------------------------------------------------------------
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Expected file path or PIL Image, got {type(image)}")

        # ----------------------------------------------------------------------
        # 2. TENSOR PREPARATION
        # ----------------------------------------------------------------------
        # unsqueeze(0) converts (3, 224, 224) into a batch tensor (1, 3, 224, 224)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # ----------------------------------------------------------------------
        # 3. FEEDFORWARD PASS (EVALUATION MODE)
        # ----------------------------------------------------------------------
        # torch.no_grad() bypasses computational graph tracking for maximum speed
        with torch.no_grad():
            logits = self.model(tensor)                  # Raw logits: shape (1, 4)
            # Softmax maps raw scores into probabilities in (0, 1) that sum to 1.0
            probs  = F.softmax(logits, dim=1).squeeze()  # Shape (4,)

        probs_np = probs.cpu().numpy()

        # ----------------------------------------------------------------------
        # 4. RESULT PARSING & STRUCTURING
        # ----------------------------------------------------------------------
        # Top-1 highest prediction
        top1_idx  = int(probs_np.argmax())
        top1_name = self.class_names[top1_idx]
        top1_conf = float(probs_np[top1_idx])

        # Full score dictionary for all 4 classes
        all_scores = {
            name: round(float(p), 6)
            for name, p in zip(self.class_names, probs_np)
        }

        # Top-k ranking sorted in descending order of probability
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


# ==============================================================================
# STEP 3: CONVENIENCE ONE-SHOT FUNCTION
# ==============================================================================

def predict(model_path, image_path) -> dict:
    """
    Convenience function for one-off CLI calls (loads model fresh).
    """
    return Predictor(model_path=model_path).predict(image_path)


# ==============================================================================
# STEP 4: CLI ENTRY POINT
# ==============================================================================

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
        # Output clean JSON for programmatic pipelines
        print(json.dumps(result, indent=2))
    else:
        # Render visual ASCII confidence bars for terminal users
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

