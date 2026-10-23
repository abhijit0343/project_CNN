# NeuroScan AI — Brain Tumor MRI Explorer

An educational and clinical research interface for exploring deep convolutional neural network feature representations on brain MRI scans.

The application pairs a **React + JSX + Tailwind CSS** frontend with a **Node.js / Express** backend, which executes inference directly on a trained **EfficientNet-B0** model (`neuroscan_model.pkl` / `neuroscan_final.pt`) using PyTorch.

> **Research and Education Notice:** NeuroScan AI is an investigational software platform strictly intended for computer science, neuroimaging research, and educational demonstration. It is not an FDA-cleared diagnostic device and its outputs must not be used for primary patient screening or clinical decisions.

---

## Model Specifications

- **Architecture:** EfficientNet-B0 (Compound Scaling α=1.2, β=1.1) with 7 Inverted Residual MBConv stages.
- **Classification Head:** Global Average Pooling (1280-d) → BatchNorm1d → Linear(256) → ReLU → Dropout(0.4) → Linear(128) → ReLU → Dropout(0.3) → Linear(4) → Softmax.
- **Input Tensor:** Standardized 224 × 224 RGB image normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Target Classes:** `glioma`, `meningioma`, `notumor`, `pituitary`.
- **Test Performance:** **93.06% Accuracy** across 1,600 held-out test scans (400 balanced scans per class), **92.96% Macro F1-Score**, and **100% recall on healthy brain scans**.
- **Model Checkpoints:** Stored in `model/src/models/` (`neuroscan_model.pkl` and `neuroscan_final.pt`).

---

## Architecture & Data Flow

```
┌─────────────────────────────────┐
│     React + JSX Frontend        │ (Vite :5173)
│   - 60fps Neural Graph Hero     │
│   - 1-Click Sample Scans        │
│   - Interactive CNN Pipeline    │
│   - Grad-CAM Comparison Slider  │
└────────────────┬────────────────┘
                 │ POST /api/predict or /api/scans/upload
                 ▼
┌─────────────────────────────────┐
│    Node.js / Express Backend    │ (Express :5000)
│   - Multer file ingestion       │
│   - Child process execution     │
│   - Live/Simulation fallback    │
└────────────────┬────────────────┘
                 │ python model/src/predict.py --json
                 ▼
┌─────────────────────────────────┐
│       PyTorch ML Engine         │
│   - loads neuroscan_model.pkl   │
│   - EfficientNet-B0 inference   │
│   - Returns top-k class scores  │
└─────────────────────────────────┘
```

---

## Running Locally

### 1. Python Environment Setup

From the repository root, activate the virtual environment and ensure dependencies are installed:

```powershell
# Using the existing venv
.\venv\Scripts\Activate.ps1
pip install -r model\requirements.txt
```

### 2. Start the Backend API

```powershell
cd backend
npm install
npm run dev
```
The Express API runs at `http://localhost:5000` with endpoints:
- `POST /api/predict` — accepts uploaded MRI image (`image` or `mri` multipart field).
- `POST /api/scans/upload` — alternative upload endpoint.
- `GET /api/health` — health check.

### 3. Start the Frontend

```powershell
cd frontend
npm install
npm run dev
```
The frontend starts at `http://localhost:5173` with automated Vite proxy forwarding `/api` to the backend.

To build the production bundle:
```powershell
npm run build
```

---

## Key Features

1. **Brand Aesthetic:** Built strictly with Prussian Blue (`#003153`), Signal Red (`#D62839`), and clean off-white backgrounds with Instrument Serif + DM Sans typography. No generic neon purple AI templates.
2. **Hero 60fps Canvas:** Real-time synaptic graph simulation with dynamic pulse signals and interactive cursor proximity attraction.
3. **1-Click Test Scans:** Includes 4 pre-loaded real test slices (Glioma, Meningioma, Pituitary, Normal) for instant 1-click evaluation without needing personal scan files.
4. **Interactive CNN Diagram:** Displays all 10 real stages from `model/src/train.py` (Input → Stem 3×3 → 7 MBConv Stages → GAP Head) with playback controls (Play, Pause, Replay, 0.5× / 1× / 2× speed), tensor shapes, and receptive fields.
5. **Grad-CAM Comparison Slider:** Interactive side-by-side draggable comparison slider between original MRI and the Prussian/Red saliency attention heatmap.
6. **Performance & Benchmarks:** Interactive SVG loss and accuracy curves (Phase 1 frozen adapter vs Phase 2 fine-tuning) and interactive 4-class Confusion Matrix.
7. **Local Scan History:** Secure client-side cache of previous scans stored in `localStorage` with instant restore capability and report dossier export.
