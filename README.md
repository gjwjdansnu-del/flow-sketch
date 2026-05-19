# FlowSketch

Draw a 2D body and preview a real-time AI flow surrogate (2D U-Net trained on SU2 Euler solutions).

## Stack

- **Frontend:** React + Vite (`frontend/`)
- **Backend:** FastAPI + PyTorch U-Net (`backend/`, `models/`)
- **Site model:** `checkpoints/unet_site.pt` (rotated 2-channel input: solid mask + Mach)

## Local development

```bash
# Backend (from repo root)
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

Open the Vite URL (default port 5173). The dev server proxies `/predict` and `/health` to the API.

## Deployment

| Component | Platform |
|-----------|----------|
| Frontend | [GitHub Pages](https://pages.github.com/) (`main` branch workflow) |
| API | [Render](https://render.com/) (`render.yaml` blueprint) |

1. Push to GitHub — Pages deploys automatically.
2. In Render: **New → Blueprint** → connect this repo → deploy `flow-sketch-api`.
3. In GitHub repo **Settings → Secrets and variables → Actions → Variables**, set:
   - `VITE_API_BASE` = `https://<your-render-service>.onrender.com` (no trailing slash)

Live UI: `https://<user>.github.io/flow-sketch/`

## Training data (local)

Large CFD/npz artifacts are gitignored under `datasets/`. Regenerate with scripts in `cfd_pipeline/`.
