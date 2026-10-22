"""
app.py
──────
FastAPI server for NeuroScan AI model inference.
"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
from pathlib import Path
import sys

# Add current directory to sys.path to allow local imports
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import Predictor

app = FastAPI(title="NeuroScan AI Model Service")

# Enable CORS for backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor (loads model once)
# Default model path is models/neuroscan_final.pt inside SRC_DIR
try:
    predictor = Predictor()
except Exception as e:
    print(f"Error initializing Predictor: {e}")
    predictor = None

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        result = predictor.predict(image, top_k=4)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
