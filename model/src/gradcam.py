"""
gradcam.py
────────────────────────────────────────────────────────────────────────────────
Gradient-weighted Class Activation Mapping (Grad-CAM) for NeuroScan CNN.

What is Grad-CAM?
─────────────────
Grad-CAM answers the question: "which pixels of the input MRI contributed most
to the model predicting class C?"  This is essential for clinical trust —
radiologists need to confirm the model is looking at tumour tissue, not
imaging artefacts.

Algorithm (Selvaraju et al., 2017)
───────────────────────────────────
  1. Pick a target convolutional layer L (we use backbone.blocks[-1], the last
     MBConv block in EfficientNet-B0 — richest high-level feature maps).
  2. Run a forward pass; record the layer's output activation A_L  (B,K,H',W').
  3. Backpropagate *only* the gradient of the predicted-class logit:
       ∂ logit_c / ∂ A_L   — this tells us how each activation channel
       contributed to pushing the score up.
  4. Global-average-pool the gradients over spatial dims (H', W') to get
     per-channel importance weights α_k (one scalar per feature map):
       α_k = (1/Z) Σ_{i,j}  ∂ logit_c / ∂ A^k_{ij}
  5. Weighted sum of activation maps, then ReLU:
       CAM = ReLU( Σ_k  α_k · A^k_L )
     ReLU keeps only positive influences (regions that push the score up).
  6. Upsample the (H', W') heatmap to the original image size with bilinear
     interpolation, then blend it onto the input image with OpenCV's JET
     colormap (blue→green→yellow→red, hot = most influential).

Why the last convolutional block?
──────────────────────────────────
Earlier layers encode low-level patterns (edges, textures).  The last
convolutional layer before global pooling encodes semantic concepts most
directly tied to the class decision.  For EfficientNet-B0 this is
model.backbone.blocks[-1].

Outputs (saved to models/)
──────────────────────────
  gradcam_samples.png  — grid of (original | Grad-CAM overlay) for one (or
                         more) random test image per class.

Usage
─────
    python gradcam.py                          # 1 sample per class, default model
    python gradcam.py --model best_phase2.pt
    python gradcam.py --samples 3              # 3 random samples per class
    python gradcam.py --test-dir path/to/Testing
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ── Local imports ─────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import build_model           # noqa: E402
from preprocessing import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

MODELS_DIR  = SRC_DIR / "models"
DATASET_DIR = SRC_DIR / "dataset"


# ══════════════════════════════════════════════════════════════════════════════
# GradCAM class
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Hook-based Grad-CAM implementation for any PyTorch nn.Module.

    How PyTorch hooks work
    ──────────────────────
    Hooks are callback functions you attach to a layer.  PyTorch calls them
    automatically at the right moment:

    register_forward_hook(fn)
        fn(module, input, output) is called after every forward pass through
        the layer.  We use it to cache output — the activation map A_L.

    register_full_backward_hook(fn)
        fn(module, grad_input, grad_output) is called after every backward
        pass through the layer.  grad_output[0] is d(loss)/d(A_L) — the
        gradient of the loss w.r.t. the layer's output.

    Parameters
    ──────────
    model        : full NeuroScanCNN, on the correct device
    target_layer : the nn.Module whose activations & gradients to capture
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.activations = None    # filled in forward hook
        self.gradients   = None    # filled in backward hook

        # Register hooks — they remain active until remove_hooks() is called
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ────────────────────────────────────────────────────────

    def _save_activation(self, module, input, output):
        """Forward hook: cache the layer's output (activation map A_L)."""
        # .detach() breaks the connection from the computational graph so
        # this tensor won't accidentally accumulate gradients.
        self.activations = output.detach()     # shape: (1, K, H', W')

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: cache the gradient flowing back through this layer."""
        self.gradients = grad_output[0].detach()  # shape: (1, K, H', W')

    # ── Main computation ──────────────────────────────────────────────────────

    def generate(self, img_tensor: torch.Tensor):
        """
        Run one forward + backward pass and compute the Grad-CAM heatmap.

        Parameters
        ──────────
        img_tensor : preprocessed image, shape (1, 3, H, W), on model device

        Returns
        ───────
        heatmap  : np.ndarray (H', W') in [0, 1]  — low-res, not yet upsampled
        pred_idx : int — predicted class index
        conf     : float — softmax confidence for the predicted class
        """
        self.model.eval()

        # ── Forward pass — _save_activation fires here ─────────────────────
        logits   = self.model(img_tensor)                # (1, num_classes)
        pred_idx = int(logits.argmax(dim=1).item())

        # softmax: converts raw logits to probabilities in (0, 1) summing to 1
        conf = float(torch.softmax(logits, dim=1)[0, pred_idx].item())

        # ── Backward on the predicted-class logit only ─────────────────────
        # We only care about the gradient for the top-1 class, not the loss.
        self.model.zero_grad()
        logits[0, pred_idx].backward()   # _save_gradient fires here

        # ── Compute importance weights α_k ─────────────────────────────────
        # Global average pool over spatial dims (H', W') → scalar per channel
        # α_k = mean_{i,j} ( ∂ logit_c / ∂ A^k_{i,j} )
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # ── Weighted sum + ReLU ────────────────────────────────────────────
        # cam = ReLU( Σ_k  α_k · A^k )
        cam = (weights * self.activations).sum(dim=1).squeeze()   # (H', W')
        cam = torch.relu(cam)

        # Normalise to [0, 1] so heatmap values are comparable across images
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), pred_idx, conf

    def remove_hooks(self):
        """
        Deregister both hooks to free memory.
        Always call this when finished — lingering hooks cause memory leaks
        because PyTorch keeps a reference to every registered hook.
        """
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap overlay helper
# ══════════════════════════════════════════════════════════════════════════════

def overlay_heatmap(img_pil: Image.Image, heatmap: np.ndarray,
                    img_size: int = 224, alpha: float = 0.45) -> np.ndarray:
    """
    Upsample a Grad-CAM heatmap and blend it with the original image.

    Steps
    ─────
    1. Resize the low-resolution heatmap (H', W') to (img_size, img_size)
       using bilinear interpolation (cv2.resize default).
    2. Scale to uint8 [0, 255].
    3. Apply the JET colormap via cv2.applyColorMap:
         blue  → low activation (little influence)
         green → medium activation
         red   → high activation (most influential region)
    4. Convert from OpenCV's BGR to RGB (PIL / matplotlib convention).
    5. Alpha-blend with the original image:
         blended = α · heatmap_rgb + (1-α) · original_rgb

    Parameters
    ──────────
    img_pil  : original PIL image (any size)
    heatmap  : (H', W') float array in [0, 1] from GradCAM.generate()
    img_size : output resolution in pixels
    alpha    : heatmap opacity (0 = invisible, 1 = opaque)

    Returns
    ───────
    blended : (img_size, img_size, 3) uint8 numpy array, RGB
    """
    img_np = np.array(img_pil.resize((img_size, img_size)))   # (H, W, 3) uint8

    # Upsample heatmap to display resolution
    hm = cv2.resize(heatmap, (img_size, img_size))
    hm = np.uint8(255 * hm)

    # Apply JET colormap (output is BGR) then convert to RGB
    hm_colour = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_colour = cv2.cvtColor(hm_colour, cv2.COLOR_BGR2RGB)

    # Weighted blend
    return np.uint8(hm_colour * alpha + img_np * (1.0 - alpha))


# ══════════════════════════════════════════════════════════════════════════════
# Main visualisation function
# ══════════════════════════════════════════════════════════════════════════════

def generate_gradcam_grid(
    model_path,
    test_dir=None,
    samples_per_class: int = 1,
    device=None,
    save_dir: Path = None,
):
    """
    Build and save a Grad-CAM visualisation grid.

    For each class, picks `samples_per_class` random images from the test set,
    runs Grad-CAM, and plots (original | heatmap overlay) side by side.

    Parameters
    ──────────
    model_path        : path to a .pt file saved by train.py
    test_dir          : path to the Testing/ directory
    samples_per_class : number of random images per class to visualise
    device            : torch.device (auto-detected if None)
    save_dir          : directory where gradcam_samples.png is written
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if save_dir is None:
        save_dir = MODELS_DIR
    if test_dir is None:
        test_dir = DATASET_DIR / "Testing"
    test_dir = Path(test_dir)

    print(f"\nDevice      : {device}")
    print(f"Model       : {model_path}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict  = checkpoint["model_state_dict"]
        class_names = checkpoint["class_names"]
        img_size    = checkpoint["img_size"]
        num_classes = checkpoint["num_classes"]
    else:
        # Raw state_dict — read meta from JSON
        state_dict = checkpoint
        meta_path  = MODELS_DIR / "model_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        class_names = meta["class_names"]
        img_size    = meta["img_size"]
        num_classes = meta["num_classes"]

    print(f"Classes     : {class_names}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(num_classes, freeze_backbone=False).to(device)
    model.load_state_dict(state_dict)
    print("Weights loaded ✓")

    # ── Attach Grad-CAM to last backbone block ────────────────────────────────
    # backbone.blocks[-1] is the last MBConv block of EfficientNet-B0.
    # It produces feature maps that encode high-level semantic patterns — the
    # right level of abstraction for class-discriminative visualisation.
    target_layer = model.backbone.blocks[-1]
    gradcam      = GradCAM(model, target_layer)
    print(f"Target layer: {target_layer.__class__.__name__}  (backbone.blocks[-1])")

    # ── Deterministic inference transform (no augmentation) ───────────────────
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Build the figure ──────────────────────────────────────────────────────
    n_classes = len(class_names)
    n_cols    = 2 * samples_per_class       # orig + cam per sample
    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(4.5 * n_cols, n_classes * 3.8),
        squeeze=False,
    )

    for row, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} not found — skipping")
            continue

        all_images = list(class_dir.glob("*"))
        if not all_images:
            print(f"  Warning: no images in {class_dir}")
            continue

        chosen = random.sample(all_images, min(samples_per_class, len(all_images)))

        for s, img_path in enumerate(chosen):
            img_pil    = Image.open(img_path).convert("RGB")
            img_tensor = val_tf(img_pil).unsqueeze(0).to(device)   # (1,3,H,W)

            # Run Grad-CAM
            heatmap, pred_idx, conf = gradcam.generate(img_tensor)
            overlay  = overlay_heatmap(img_pil, heatmap, img_size)
            pred_lbl = class_names[pred_idx]
            tick     = "✅" if pred_lbl == class_name else "❌"

            col_orig = s * 2
            col_cam  = s * 2 + 1

            # Original image
            axes[row, col_orig].imshow(img_pil.resize((img_size, img_size)))
            axes[row, col_orig].set_title(
                f"True: {class_name}", fontsize=10, fontweight="bold"
            )
            axes[row, col_orig].axis("off")

            # Grad-CAM overlay
            axes[row, col_cam].imshow(overlay)
            axes[row, col_cam].set_title(
                f"Pred: {pred_lbl} {tick}\nConf: {conf*100:.1f}%",
                fontsize=10,
            )
            axes[row, col_cam].axis("off")

    # Clean up hooks before saving — prevents memory leaks
    gradcam.remove_hooks()

    plt.suptitle("Grad-CAM Explanations — NeuroScan CNN",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = save_dir / "gradcam_samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Grad-CAM grid saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualisation — NeuroScan CNN"
    )
    parser.add_argument("--model",    default="neuroscan_final.pt",
                        help="Model filename in models/ (default: neuroscan_final.pt)")
    parser.add_argument("--samples",  type=int, default=1,
                        help="Random images per class to visualise (default: 1)")
    parser.add_argument("--test-dir", default=None,
                        help="Path to Testing/ directory")
    cfg = parser.parse_args(args)

    model_path = MODELS_DIR / cfg.model
    test_dir   = Path(cfg.test_dir) if cfg.test_dir else DATASET_DIR / "Testing"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generate_gradcam_grid(
        model_path        = model_path,
        test_dir          = test_dir,
        samples_per_class = cfg.samples,
        device            = device,
        save_dir          = MODELS_DIR,
    )


if __name__ == "__main__":
    main()
