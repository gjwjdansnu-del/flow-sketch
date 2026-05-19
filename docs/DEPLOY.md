# FlowSketch deployment checklist

GitHub Pages serves **only the frontend**. Predict requires a separate API (Render).

## 1. Deploy Render backend first

1. Open https://dashboard.render.com → **New → Blueprint**
2. Connect repository `podobooks-ganghwa/flow-sketch`
3. Apply `render.yaml` (service name: `flow-sketch-api`)
4. Wait until deploy status is **Live** (first build may take 10–15 min because of PyTorch)

## 2. Verify API health

Open in a browser (replace host if Render assigned a different name):

```text
https://flow-sketch-api.onrender.com/health
```

Expected JSON:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "rotated2",
  "input_channels": 2
}
```

If you see `Not Found`, the backend is **not deployed yet** — GitHub Pages will show `Failed to fetch`.

## 3. Point GitHub Pages at the API

1. GitHub repo → **Settings → Secrets and variables → Actions → Variables**
2. Add or update:
   - Name: `VITE_API_BASE`
   - Value: `https://flow-sketch-api.onrender.com` (no trailing slash)
3. **Actions** → **Deploy GitHub Pages** → **Run workflow**

## 4. Verify the live site

- Frontend: https://podobooks-ganghwa.github.io/flow-sketch/
- Footer should show `API: https://flow-sketch-api.onrender.com` and `Backend: ok (...)`
- Draw a shape → **Predict**

## CORS

The API allows `https://podobooks-ganghwa.github.io` and local Vite ports. Override with env `CORS_ORIGINS` on Render if needed.

## Troubleshooting `Failed to fetch`

| Symptom | Fix |
|--------|-----|
| `/health` returns 404 on Render | Finish Blueprint deploy |
| Footer: API URL not configured | Set `VITE_API_BASE` and rerun Pages workflow |
| Health OK in browser but site fails | Hard-refresh; check browser console for CORS |
| Cold start on free tier | Wait ~30s after idle, retry Predict |
