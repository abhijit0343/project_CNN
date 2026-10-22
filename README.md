# NeuroScan AI

> An AI-powered brain tumor detection system using MRI scans.

## Project Structure

```
NeuroScan-AI/
├── frontend/          # React + TypeScript + Vite + Tailwind CSS
├── backend/           # Node.js + Express.js
├── model/             # Python + TensorFlow/Keras (CNN model)
│   ├── dataset/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   ├── src/
│   ├── saved_models/
│   ├── outputs/
│   └── requirements.txt
├── docs/              # Project documentation
├── .gitignore
├── .env.example
└── README.md
```

## Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Frontend  | React, TypeScript, Vite, Tailwind   |
| Backend   | Node.js, Express.js                 |
| ML Model  | Python, TensorFlow / Keras          |

## Getting Started

See each sub-directory's `README.md` for setup instructions.

### 1. Frontend
```bash
cd frontend
npm install
npm run dev
```

### 2. Backend
```bash
cd backend
npm install
npm run dev
```

### 3. Model
```bash
cd model
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## License

MIT
