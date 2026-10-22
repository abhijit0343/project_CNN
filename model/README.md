# NeuroScan AI — Python ML Model

CNN-based brain tumour classification from MRI scans.

## Structure

```
model/
├── dataset/
│   ├── raw/           ← Place downloaded dataset here (git-ignored)
│   └── processed/     ← Preprocessed arrays (git-ignored)
├── notebooks/         ← Jupyter exploration notebooks
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
├── saved_models/      ← Trained model files (git-ignored)
├── outputs/           ← Plots, confusion matrices (git-ignored)
└── requirements.txt
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Launch Jupyter
jupyter notebook
```

## Workflow (planned)

1. **Download** dataset → `dataset/raw/`
2. **Preprocess** → `python src/preprocessing.py`
3. **Train**      → `python src/train.py`
4. **Evaluate**   → `python src/evaluate.py`
5. **Predict**    → `python src/predict.py <image_path>`

## Dataset

This project is designed for the [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle).
Download it manually and place the folders inside `dataset/raw/`.
