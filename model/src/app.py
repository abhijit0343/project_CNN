# ==============================================================================
# app.py: FastAPI Microservice for Serving NeuroScan CNN Predictions
# ==============================================================================
# This script wraps the PyTorch deep learning model inside an asynchronous HTTP API.
# It allows external clients (like a Node.js backend or web frontend) to send MRI
# scans via POST requests and receive tumor classification probabilities in JSON format.
# ==============================================================================

import os
import io
import sys
from pathlib import Path
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# STEP 1: PATH RESOLUTION & LOCAL IMPORTS
# ------------------------------------------------------------------------------
# Ensure the current directory (src/) is in Python's module lookup path (sys.path).
# This guarantees that 'from predict import Predictor' succeeds regardless of which
# directory the server was launched from.
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the inference engine wrapper
from predict import Predictor

# ------------------------------------------------------------------------------
# STEP 2: FASTAPI APPLICATION INITIALIZATION & CORS SETUP
# ------------------------------------------------------------------------------
# Create the FastAPI app instance with metadata for interactive OpenAPI docs (/docs)
app = FastAPI(
    title="NeuroScan AI Model Service",
    description="Deep Learning API for Brain Tumor MRI Classification (EfficientNet-B0)",
    version="1.0.0"
)

# Cross-Origin Resource Sharing (CORS) Middleware:
# Allows browsers or backend servers running on different ports/domains (e.g.,
# localhost:5000 or localhost:5173) to send requests to this API without security blocks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (frontend/backend)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP verbs: GET, POST, OPTIONS, etc.
    allow_headers=["*"],       # Allow all headers
)

# ------------------------------------------------------------------------------
# STEP 3: MODEL PRE-LOADING (SINGLETON PATTERN)
# ------------------------------------------------------------------------------
# Loading a deep learning model into RAM/VRAM is slow (takes 1-3 seconds).
# By instantiating Predictor() once at server startup, weights stay resident in
# memory, enabling fast per-request inference (< 50ms per scan).
try:
    predictor = Predictor()
    print("✅ [NeuroScan AI Service] Predictor initialized and model resident in memory.")
except Exception as e:
    print(f"❌ [NeuroScan AI Service] Error initializing Predictor: {e}")
    predictor = None

# ------------------------------------------------------------------------------
# STEP 4: HEALTH CHECK ENDPOINT
# ------------------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Health check route used by container orchestrators (Docker/K8s) or the backend
    to verify that the server is alive and the AI model is ready for inference.
    """
    return {
        "status": "ok",
        "model_loaded": predictor is not None
    }

# ------------------------------------------------------------------------------
# STEP 5: PREDICTION INFERENCE ENDPOINT
# ------------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main inference route.
    Accepts: An uploaded MRI scan file (multipart/form-data).
    Returns: JSON containing predicted tumor class, confidence score, and
             full probability distribution across all 4 classes.
    """
    # Guard 1: Ensure the model loaded successfully at startup
    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded on server. Please check server logs."
        )
    
    # Guard 2: Validate MIME type to ensure the client submitted an image file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Uploaded file must be an image (.jpg, .jpeg, .png)."
        )

    try:
        # Read the raw binary content of the uploaded file asynchronously
        contents = await file.read()
        
        # Convert raw bytes into a PIL Image and force 3-channel RGB.
        # This converts grayscale MRI scans (1 channel) or RGBA PNGs (4 channels)
        # into standard 3-channel RGB as expected by the EfficientNet backbone.
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Execute the prediction pipeline (Preprocess -> Forward Pass -> Softmax)
        result = predictor.predict(image, top_k=4)
        return result

    except Exception as e:
        # Catch unexpected errors (corrupt image bytes, tensor shape issues) and return HTTP 500
        raise HTTPException(
            status_code=500,
            detail=f"Inference error during model prediction: {str(e)}"
        )

# ------------------------------------------------------------------------------
# STEP 6: LOCAL RUNNER ENTRY POINT
# ------------------------------------------------------------------------------
# If this file is run directly with `python app.py`, launch the Uvicorn ASGI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
