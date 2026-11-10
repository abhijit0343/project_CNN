# NeuroScan AI — PyTorch Deep Learning Model

Convolutional Neural Network (CNN) pipeline for brain tumour classification and explainable feature visualization from MRI scans, built with **PyTorch** and **timm**.

---

## Directory Structure

```
model/
├── dataset/
│   ├── raw/                 # Downloaded dataset folders (git-ignored)
│   │   ├── Training/        # glioma, meningioma, notumor, pituitary
│   │   └── Testing/         # held-out validation test split
│   └── processed/           # Processed datasets (git-ignored)
├── notebooks/               # Jupyter exploration and experimentation notebooks
├── requirements.txt         # PyTorch ML dependencies
└── src/
    ├── app.py               # FastAPI microservice for direct REST model serving
    ├── evaluate.py          # Model evaluation script (test accuracy, F1, report)
    ├── gradcam.py           # Grad-CAM interpretability and heatmap visualizer
    ├── pipeline.py          # Serialized inference pipeline & pkl exporter
    ├── predict.py           # CLI prediction script with JSON output flag
    ├── preprocessing.py     # Image normalization, augmentation, PyTorch DataLoaders
    ├── train.py             # Two-phase transfer learning trainer (EfficientNet-B0)
    └── models/              # Checkpoints, charts, and metrics
        ├── neuroscan_final.pt       # PyTorch state_dict checkpoint
        ├── neuroscan_model.pkl      # Bundled self-contained inference pipeline
        ├── model_meta.json          # Normalization constants & class mappings
        ├── training_history.json    # Epoch-by-epoch loss & accuracy log
        ├── evaluation_report.txt    # Precision, recall, F1 classification metrics
        ├── confusion_matrix.png     # 4-class confusion matrix plot
        ├── training_curves.png      # Loss and accuracy curves
        ├── per_class_accuracy.png   # Class-by-class accuracy bar chart
        └── gradcam_samples.png      # Sample Grad-CAM attention visualizations
```

---

## Model Architecture

The classifier leverages **EfficientNet-B0** via `timm` with transfer learning from ImageNet-1k, customized with an adapter head designed to prevent overfitting on clinical scans:

```
Input (224 × 224 × 3 RGB Tensor)
  │
  ▼
EfficientNet-B0 Backbone (7 Inverted Residual MBConv stages, α=1.2, β=1.1)
  │
  ▼
AdaptiveAvgPool2d (1280-dimensional feature vector)
  │
  ▼
Flatten
  │
  ▼
BatchNorm1d(1280) ──► Linear(1280 ──► 256) ──► ReLU ──► Dropout(0.40)
  │
  ▼
BatchNorm1d(256)  ──► Linear(256  ──► 128) ──► ReLU ──► Dropout(0.30)
  │
  ▼
Linear(128 ──► 4) ──► Softmax
  │
  ▼
Output Probabilities: [glioma, meningioma, notumor, pituitary]
```

---

## Environment Setup

### 1. Create and Activate Virtual Environment

```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch & Dependencies

**For NVIDIA GPUs (CUDA 12.1 / 12.4 acceleration):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r model/requirements.txt
```

**For CPU-only environments:**
```bash
pip install -r model/requirements.txt
```

---

## Training Pipeline

The training procedure uses a **two-phase transfer learning strategy**:
- **Phase 1 (Adapter Warmup):** EfficientNet-B0 backbone weights are frozen. Only the newly initialized classification head is trained with AdamW (`lr=1e-3`) for 5 epochs.
- **Phase 2 (Fine-Tuning):** The top 30% of the backbone stages are unfrozen and trained end-to-end at a 10× lower learning rate (`lr=1e-4`) using Cosine Annealing learning rate scheduling and Automatic Mixed Precision (`torch.cuda.amp`).

To train the model:
```bash
python model/src/train.py --data-dir model/dataset/raw --epochs-warmup 5 --epochs-finetune 15 --batch-size 32
```

---

## Evaluation & Metrics

Evaluate the trained checkpoint against the held-out test split (1,600 balanced images):
```bash
python model/src/evaluate.py --data-dir model/dataset/raw/Testing
```

### Held-Out Test Results:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 97.86% | 80.00% | 0.8803 | 400 |
| **Meningioma** | 86.71% | 96.25% | 0.9123 | 400 |
| **No Tumor** | 91.32% | **100.00%** | **0.9547** | 400 |
| **Pituitary** | 98.21% | 96.00% | 0.9709 | 400 |
| **Overall / Macro** | **93.53%** | **93.06%** | **92.96%** | **1,600** |

> **Key Finding:** 100% recall on healthy brain scans (`notumor`), completely eliminating false negative alerts on non-pathological tissue.

---

## Grad-CAM Interpretability

Generate explainability saliency heatmaps overlaid directly onto MRI scans to visualize which anatomical structures influenced the model's decision:
```bash
python model/src/gradcam.py path/to/mri_scan.jpg --target-layer conv_head --save-path heatmap.png
```

---

## CLI Inference

Run prediction directly on an individual scan file:
```bash
# Standard console output
python model/src/predict.py path/to/mri_scan.jpg

# Structured JSON output (used by Node.js backend)
python model/src/predict.py path/to/mri_scan.jpg --json --top-k 4
```

---

## FastAPI Microservice

To serve predictions via an independent asynchronous REST microservice:
```bash
python model/src/app.py
# or
uvicorn model.src.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.
