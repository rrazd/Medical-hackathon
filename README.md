# DermaMatch — Explainable AD Treatment Response Predictor

DermaMatch is a browser-based prototype that helps atopic dermatitis (AD) patients
understand how likely they are to improve on a specific biologic **before** they meet
their dermatologist. A patient uploads a baseline photo plus a few details, and the tool
analyzes visual skin biomarkers, matches them against a curated reference set of real
before/after cases, and returns a transparent, explainable estimate of likely response
for **Dupixent** (dupilumab) and **Ebglyss** (lebrikizumab).

> ⚠️ **Decision-support only.** DermaMatch is a prototype for discussion with a
> dermatologist. It is **not** a diagnosis, prescription, or medical advice.

## Stack

- **Backend:** Python 3 · FastAPI · OpenCV / scikit-image (biomarker extraction) ·
  scikit-learn (similarity matching)
- **Frontend:** React 19 · Vite · TypeScript

## Getting started

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://127.0.0.1:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev                     # http://127.0.0.1:5173  (proxies /api → :8000)
```

## How it works

1. **Upload & intake** — a multi-step wizard collects a baseline photo and patient details.
2. **Biomarker extraction** — classical CV measures erythema, lesion coverage, texture,
   dryness/scaling, and inflammation from the image.
3. **Similarity matching** — the patient vector is matched against reference before/after
   cases; per-biologic response is estimated from the most similar cases.
4. **Explainable results** — the UI shows likelihood, contributing biomarkers, and the
   matched reference cases behind the estimate.

## Data & privacy

- Reference cases live under `backend/data/` (CSV metadata + `images/`).
- Only de-identified / consented images are used. Patient uploads are processed for the
  session only and are **not** persisted.

## Tests

```bash
cd backend && pytest          # backend unit tests
cd frontend && npm test       # frontend (vitest)
```
