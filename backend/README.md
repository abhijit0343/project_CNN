# NeuroScan AI — Backend

Express.js REST API that bridges the frontend and the Python ML model.

## Structure

```
backend/
├── src/
│   ├── index.js           # Entry point
│   ├── app.js             # Express app setup
│   ├── routes/            # Route definitions
│   ├── controllers/       # Request / response handlers
│   ├── services/          # Business logic
│   ├── middleware/        # Multer, error handling, etc.
│   └── config/            # Centralised config
└── uploads/               # Temporary upload storage (git-ignored)
```

## Setup

```bash
npm install
cp ../.env.example ../.env   # then edit .env
npm run dev
```

API will be available at `http://localhost:5000`.

## Key Endpoints

| Method | Endpoint           | Description              |
|--------|--------------------|--------------------------|
| GET    | /api/health        | Health check             |
| POST   | /api/scans/upload  | Upload MRI for analysis  |
| GET    | /api/scans/:id     | Get scan result by ID    |
