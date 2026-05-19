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

**`Failed to fetch` on the live site means the Render API is missing or down.** Pages alone cannot run inference.

### Checklist

1. **Deploy Render backend first** — Dashboard → New → Blueprint → this repo → `flow-sketch-api`
2. **Open** `https://flow-sketch-api.onrender.com/health` — must return `"model_loaded": true`
3. **Set GitHub Actions variable** `VITE_API_BASE` = `https://flow-sketch-api.onrender.com` (no trailing slash)
4. **Re-run** workflow **Deploy GitHub Pages**
5. Open https://podobooks-ganghwa.github.io/flow-sketch/ and test Predict

See [docs/DEPLOY.md](docs/DEPLOY.md) for details and troubleshooting.

## Training data (local)

Large CFD/npz artifacts are gitignored under `datasets/`. Regenerate with scripts in `cfd_pipeline/`.
