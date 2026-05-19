# flow_sketch frontend

Simple React UI for drawing a 2D body, calling the FastAPI `/predict` endpoint, and viewing the predicted flow field colormap.

## Run

1. Start the inference backend (separate terminal):

```bash
cd /Users/apl/projects/flow_sketch
uvicorn backend.app:app --reload
```

2. Install and run the frontend:

```bash
cd /Users/apl/projects/flow_sketch/frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually http://localhost:5173).

## API proxy

During `npm run dev`, Vite proxies `/predict` and `/health` to `http://127.0.0.1:8000` so the browser avoids CORS issues.

To call the backend directly instead, set:

```bash
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

(Requires CORS enabled on the backend.)

## Usage

- Draw a closed polygon on the sketch canvas (click/drag points, double-click or click near the first point to close).
- Set Mach (1.5–5.0) and AoA (-10–10°).
- Click **Predict**.
- Switch field buttons (Mach, Pressure, Density, Temperature, Shock) after a prediction is available.
