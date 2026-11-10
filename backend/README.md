# NeuroScan AI — Backend Service

Express.js REST API that bridges the frontend React client and the PyTorch deep learning engine.

## Structure

```
backend/
├── src/
│   ├── index.js           # Server entry point
│   ├── app.js             # Express application & middleware configuration
│   ├── routes/            # REST API route definitions (scanRoutes.js)
│   ├── controllers/       # Controller handlers (scanController.js)
│   ├── services/          # Model inference execution & simulation fallback (scanService.js)
│   ├── middleware/        # Multer multipart upload middleware
│   └── config/            # Environment & server settings
└── uploads/               # Temporary MRI scan upload buffer (git-ignored)
```

## Setup & Running

```bash
# 1. Install dependencies
npm install

# 2. Configure environment (optional, defaults to PORT=5000)
cp ../.env.example ../.env

# 3. Start development server with live reload
npm run dev
```

API will be available at `http://localhost:5000`.

## PyTorch Model Execution Pipeline

When an MRI scan is uploaded via `/api/predict` or `/api/scans/upload`:
1. **Multer** saves the image temporarily in `backend/uploads/`.
2. **Scan Service** executes `python model/src/predict.py` via Node.js `child_process.execFile`:
   - Points to `model/src/models/neuroscan_model.pkl` (or `neuroscan_final.pt`).
   - Automatically detects and uses the local virtual environment (`venv/Scripts/python.exe`).
   - Returns top-k probabilities, predicted class, and CNN layer activation stats in structured JSON.
3. If Python or PyTorch is not available, the backend falls back to an internal simulation engine so frontend demonstrations can proceed uninterrupted.

## Key Endpoints

| Method | Endpoint             | Description                                              |
|--------|----------------------|----------------------------------------------------------|
| `GET`  | `/api/health`        | Health check verifying backend status                    |
| `POST` | `/api/predict`       | Primary endpoint for MRI scan analysis (`image` field)   |
| `POST` | `/api/scans/upload`  | Secondary upload route (`mri` field)                     |
| `GET`  | `/api/scans/:id`     | Fetch previous scan result by ID                         |
