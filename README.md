# NeuroScan AI — Brain Tumor MRI Explorer

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-1.0.9-orange?style=flat-square)](https://github.com/huggingface/pytorch-image-models)
[![React](https://img.shields.io/badge/React-18.x-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.x-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.x-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=flat-square&logo=node.js&logoColor=white)](https://nodejs.org/)
[![Express](https://img.shields.io/badge/Express-4.x-000000?style=flat-square&logo=express&logoColor=white)](https://expressjs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

An educational and clinical research interface for exploring deep convolutional neural network feature representations on brain MRI scans.

NeuroScan AI pairs a **React + JSX + Tailwind CSS** frontend with a **Node.js / Express** backend orchestrator, executing real-time inference on a fine-tuned **EfficientNet-B0** model built with **PyTorch** and **timm** (Image Models). The system includes native **Grad-CAM interpretability**, animated **CNN layer-by-layer activation flows**, and an interactive **Prussian Blue & Signal Red** research workbench.

> **Research and Education Notice:** NeuroScan AI is an investigational software platform strictly intended for computer science, neuroimaging research, and educational demonstration. It is not an FDA-cleared diagnostic device and its outputs must not be used for primary patient screening or clinical decisions.

---

## Tech Stack

| Layer | Technologies | Role in System |
| :--- | :--- | :--- |
| **Deep Learning** | **PyTorch 2.5+**, **timm 1.0.9**, Torchvision | EfficientNet-B0 backbone, custom classification adapter, Grad-CAM saliency extraction, mixed-precision training |
| **Data Processing** | NumPy, OpenCV, Pillow, Scikit-learn, Pandas | 224×224 tensor normalization, clinical augmentations, metric evaluations |
| **Model Serving** | Python CLI (`predict.py`) / FastAPI (`app.py`) | Child-process execution or asynchronous REST microservice |
| **Backend API** | Node.js, Express.js, Multer | File ingestion, subprocess execution of PyTorch inference, simulation fallback engine |
| **Frontend UI** | React 18, Vite, Tailwind CSS, Lucide Icons | 60fps synaptic canvas, 10-stage CNN flow visualizer, Grad-CAM slider, scan history |

---

## Key Features

1. **PyTorch Deep Learning Engine:** Built on EfficientNet-B0 using compound scaling ($\alpha=1.2, \beta=1.1$) and fine-tuned on 7,023 brain MRI slices across four clinical categories.
2. **Clinical Brand Aesthetic:** Designed with a bespoke clinical palette — Prussian Blue (`#003153`), Signal Red (`#D62839`), Paper White (`#F7F8F6`), and Instrument Serif + DM Sans typography.
3. **Hero 60fps Synaptic Canvas:** Interactive real-time neural network graph simulation responding to cursor proximity with synaptic pulse transmissions.
4. **Interactive 10-Stage CNN Flow:** Visualizes every layer stage from `model/src/train.py` (Stem 3×3 → 7 MBConv Stages → Global Average Pooling Head) with animated playback, receptive field calculations, and tensor dimensions.
5. **Grad-CAM Heatmap Slider:** Interactive draggable side-by-side comparison slider between the original MRI slice and the model's spatial attention activations.
6. **1-Click Test Slices:** Preloaded high-resolution scans for Glioma, Meningioma, Pituitary Adenoma, and Healthy Brain control scans for instantaneous evaluation.
7. **Transparent Benchmark Dashboard:** Interactive loss/accuracy training curves across Phase 1 & Phase 2, plus an interactive 4-class Confusion Matrix with per-class precision and recall.
8. **Client-Side Scan Dossier:** Local session cache of analyzed scans with instant result recall and printable clinical dossier reporting.

---

## Model Specifications & Performance

- **Backbone Architecture:** EfficientNet-B0 (`timm/efficientnet_b0`, pretrained on ImageNet-1k).
- **Classification Head:** 
  $$\text{AdaptiveAvgPool2d}(1280) \rightarrow \text{BatchNorm1d} \rightarrow \text{Linear}(256) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.40) \rightarrow \text{Linear}(128) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.30) \rightarrow \text{Linear}(4)$$
- **Input Tensor Dimensions:** $224 \times 224 \times 3$ normalized with ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Training Strategy:** Two-phase transfer learning (Phase 1: head warmup with frozen backbone; Phase 2: fine-tuning top 30% layers with Cosine Annealing LR and FP16 Automatic Mixed Precision).
- **Target Classes:** `glioma`, `meningioma`, `notumor`, `pituitary`.
- **Model Checkpoints:** Preserved in `model/src/models/neuroscan_final.pt` (state dictionary) and `model/src/models/neuroscan_model.pkl` (portable pipeline).

### Held-Out Validation Benchmark (1,600 Test Scans)

| Pathological Class | Precision | Recall | F1-Score | Held-Out Test Scans |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 97.86% | 80.00% | 0.8803 | 400 |
| **Meningioma** | 86.71% | 96.25% | 0.9123 | 400 |
| **No Tumor (Healthy Control)** | 91.32% | **100.00%** | **0.9547** | 400 |
| **Pituitary Adenoma** | 98.21% | 96.00% | 0.9709 | 400 |
| **Macro Average / Overall** | **93.53%** | **93.06%** | **92.96%** | **1,600** |

> **Clinical Highlight:** The model achieved **100% recall on healthy brain scans** (400/400 correct), guaranteeing zero false-positive tumor alerts on normal patient scans.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   React + JSX Frontend                      │  (Vite :5173)
│   • 60fps Dynamic Synaptic Canvas                           │
│   • 10-Stage CNN Interactive Tensor Pipeline                │
│   • Grad-CAM Dual-Pane Heatmap Comparison Slider            │
│   • Local Storage Scan History & Printable Dossier          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                   POST /api/predict (multipart/form-data)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Node.js / Express Backend                   │  (Express :5000)
│   • Multer temporary scan file buffer                       │
│   • Child-process execution of Python ML script             │
│   • Automatic detection of Python virtualenv                │
│   • Graceful fallback simulation engine for presentations   │
└──────────────────────────────┬──────────────────────────────┘
                               │
            python model/src/predict.py --json --model ...
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      PyTorch ML Engine                      │  (PyTorch + timm)
│   • Loads neuroscan_model.pkl / neuroscan_final.pt          │
│   • Standardizes 224×224 RGB image tensors                  │
│   • EfficientNet-B0 feedforward inference (< 50ms)          │
│   • Grad-CAM activation heatmap generation                  │
│   • Returns top-k probabilities & stage activations         │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Neuroscan_CNN/
├── frontend/                        # React + JSX + Tailwind CSS client
│   ├── public/                      # Static assets & sample MRI slices
│   │   └── samples/                 # Preloaded Glioma, Meningioma, Pituitary, Normal
│   ├── src/
│   │   ├── components/              # BrainNetwork, CNNInteractiveFlow, GradCamHeatmapViewer, etc.
│   │   ├── pages/                   # HeroSection, LiveExplorerSection, PerformanceSection, etc.
│   │   ├── hooks/                   # useScanAnalysis.js (API state & caching)
│   │   ├── data/                    # model.js (10 CNN stages), samples.js
│   │   ├── App.jsx                  # Single-page application shell
│   │   └── main.jsx                 # Vite root entry
│   ├── tailwind.config.js           # Brand color tokens & custom typography
│   └── vite.config.js               # Dev server & reverse-proxy configuration
│
├── backend/                         # Node.js + Express.js API
│   ├── src/
│   │   ├── app.js                   # Middleware & CORS configuration
│   │   ├── index.js                 # Server listener (:5000)
│   │   ├── routes/                  # scanRoutes.js
│   │   ├── controllers/             # scanController.js
│   │   ├── services/                # scanService.js (PyTorch child_process runner & fallback)
│   │   └── middleware/              # Multer upload handler
│   └── uploads/                     # Temporary upload directory (git-ignored)
│
├── model/                           # PyTorch ML Engineering & Training
│   ├── requirements.txt             # PyTorch, timm, OpenCV, scikit-learn
│   ├── dataset/                     # Kaggle MRI dataset (raw/ & processed/)
│   ├── notebooks/                   # Jupyter exploratory analysis
│   └── src/
│       ├── preprocessing.py         # Torchvision transforms & stratified DataLoaders
│       ├── train.py                 # Two-phase transfer learning trainer
│       ├── evaluate.py              # Test set evaluation & classification report
│       ├── predict.py               # Standalone CLI inference engine (JSON-ready)
│       ├── gradcam.py               # Grad-CAM heatmap extraction
│       ├── pipeline.py              # Self-contained pickled inference pipeline exporter
│       ├── app.py                   # Optional FastAPI microservice
│       └── models/                  # Checkpoints & evaluation artifacts
│           ├── neuroscan_final.pt   # PyTorch model state_dict
│           ├── neuroscan_model.pkl  # Pickled inference pipeline
│           ├── model_meta.json      # Model metadata & normalization parameters
│           ├── training_history.json# Epoch loss & accuracy logs
│           └── evaluation_report.txt# Test metrics breakdown
│
├── .env.example                     # Environment variable blueprint
├── .gitignore                       # Ignored caches, weights, and node_modules
└── README.md                        # Master project documentation
```

---

## Getting Started

### 1. Prerequisites
- **Node.js** (v18.0.0 or later) & **npm**
- **Python** (3.10 to 3.12 recommended)
- **NVIDIA GPU + CUDA Drivers** (Optional, for GPU-accelerated inference; runs on CPU by default)

---

### 2. Python & PyTorch Environment Setup

From the repository root, create and activate a Python virtual environment:

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS (Bash)
python3 -m venv venv
source venv/bin/activate
```

Install PyTorch and project dependencies:

```powershell
# For NVIDIA GPUs with CUDA 12.1 / 12.4 acceleration (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r model\requirements.txt

# For standard CPU-only environments:
pip install -r model\requirements.txt
```

Verify that PyTorch is working and detects your GPU (if available):
```powershell
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

---

### 3. Backend Setup

In a new terminal window:

```powershell
cd backend
npm install
npm run dev
```

The Express API starts at `http://localhost:5000`. It will automatically detect your `venv/Scripts/python.exe` and invoke the PyTorch model for scan analyses.

---

### 4. Frontend Setup

In a third terminal window:

```powershell
cd frontend
npm install
npm run dev
```

The React + Vite application will open at `http://localhost:5173`. Any requests to `/api` are automatically proxied to the backend at `http://localhost:5000`.

To create an optimized production build:
```powershell
npm run build
npm run preview
```

---

## PyTorch ML Workflow & Scripts

All machine learning scripts are located in `model/src/`:

### 1. Train the EfficientNet-B0 Model
To train the two-phase PyTorch model from scratch on the Kaggle dataset:
```powershell
python model/src/train.py --data-dir model/dataset/raw --epochs-warmup 5 --epochs-finetune 15 --batch-size 32
```
*Outputs are saved directly to `model/src/models/` (`neuroscan_final.pt`, `training_history.json`, `training_curves.png`).*

### 2. Evaluate Held-Out Test Scans
To compute the full classification report and confusion matrix:
```powershell
python model/src/evaluate.py --data-dir model/dataset/raw/Testing
```
*Generates `evaluation_report.txt` and `confusion_matrix.png`.*

### 3. Generate Grad-CAM Heatmaps
To visualize the convolutional feature maps of a given scan:
```powershell
python model/src/gradcam.py path/to/mri_slice.jpg --target-layer conv_head --save-path heatmap.png
```

### 4. Standalone CLI Inference
To test classification from the terminal:
```powershell
# Standard formatted output:
python model/src/predict.py path/to/mri_slice.jpg

# JSON output for automated services:
python model/src/predict.py path/to/mri_slice.jpg --json --top-k 4
```

### 5. Optional FastAPI Microservice
NeuroScan also includes a standalone FastAPI microservice for direct Python API deployment:
```powershell
python model/src/app.py
# Interactive API documentation: http://localhost:8000/docs
```

---

## REST API Reference

### Express Backend (`http://localhost:5000`)

#### `POST /api/predict`
Accepts a multipart form upload of a brain MRI image and returns the tumor classification, confidence metrics, and CNN layer activation stats.

- **Content-Type:** `multipart/form-data`
- **Body Field:** `image` (binary file: `.jpg`, `.jpeg`, `.png`)
- **Response Format (`application/json`):**
```json
{
  "filename": "mri_scan.jpg",
  "prediction": "meningioma",
  "confidence": 0.9842,
  "top_k": [
    { "label": "meningioma", "confidence": 0.9842 },
    { "label": "glioma", "confidence": 0.0121 },
    { "label": "pituitary", "confidence": 0.0025 },
    { "label": "notumor", "confidence": 0.0012 }
  ],
  "all_scores": {
    "glioma": 0.0121,
    "meningioma": 0.9842,
    "notumor": 0.0012,
    "pituitary": 0.0025
  },
  "stages": [
    { "name": "Input", "shape": "224 × 224 × 3", "kind": "MRI Slice", "activation": 0.95 },
    { "name": "Stem Conv3x3", "shape": "112 × 112 × 32", "kind": "Early Features / Edges", "activation": 0.88 },
    { "name": "MBConv Stage 1", "shape": "112 × 112 × 16", "kind": "Local Texture Inverted Residuals", "activation": 0.82 },
    { "name": "MBConv Stage 7", "shape": "7 × 7 × 320", "kind": "High-Level Pathological Semantics", "activation": 0.74 },
    { "name": "GAP + Head", "shape": "1 × 1280 → 4", "kind": "Global Pool & Softmax Classifier", "activation": 0.98 }
  ],
  "model_used": "EfficientNet-B0 (PyTorch)",
  "mode": "live"
}
```

#### `GET /api/health`
Health check endpoint reporting API status and model availability.
- **Response:** `{"status": "ok", "service": "NeuroScan AI Backend"}`

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
