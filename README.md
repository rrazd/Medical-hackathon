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

## Deployment & monitoring

The app is deployed as two separately hosted pieces that both **auto-deploy on
every push to `main`**:

| Piece | Host | Live URL | Dashboard to monitor |
|-------|------|----------|----------------------|
| Frontend (React/Vite) | **Vercel** | https://medical-hackathon-livid.vercel.app/ | https://vercel.com/dashboard → project `medical-hackathon` |
| Backend (FastAPI) | **Render** (free tier) | https://dermamatch-backend.onrender.com | https://dashboard.render.com/ → service `dermamatch-backend` |

**How the two connect:** `frontend/vercel.json` rewrites `/api/*` on the Vercel
site to the Render backend, so the browser only ever talks to the Vercel URL.
Backend deploy config lives in `render.yaml` (root dir `backend`, health check at
`/health`).

**What to watch on each board:**
- **Vercel → Deployments:** build/deploy status and logs for the frontend. Set the
  project **Root Directory to `frontend`** so `vercel.json` applies.
- **Render → Events / Logs:** backend deploy status, request logs, and any
  restarts. `GET /health` should return `{"ok": true}`.

**Free-tier cold start:** Render's free instance spins down after ~15 min of
inactivity, so the **first** request after idle takes ~50s while it wakes. The
frontend "Analyzing your photo…" scanner covers this; subsequent requests are
fast. (A cached copy of the reference biomarker features in
`backend/data/derived/reference_cache.json` keeps warm predictions well under a
second of compute.)

## Data & privacy

- Reference cases live under `backend/data/` (CSV metadata + `images/`).
- Only de-identified / consented images are used. Patient uploads are processed for the
  session only and are **not** persisted.

## Tests

```bash
cd backend && pytest          # backend unit tests
cd frontend && npm test       # frontend (vitest)
```
