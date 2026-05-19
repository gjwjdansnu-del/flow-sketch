"""FastAPI inference server for flow_sketch U-Net surrogate."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "unet_site.pt"
STATS_PATH = PROJECT_ROOT / "checkpoints" / "unet_site_stats.json"

NY = 128
NX = 256
OUTPUT_KEYS = ("mach", "pressure", "density", "temperature", "shock_indicator")
MODEL_TYPE = "rotated2"
INPUT_CHANNELS = 2

sys.path.insert(0, str(MODELS_DIR))
from dataset import load_output_stats  # noqa: E402
from unet import UNet  # noqa: E402


class PredictRequest(BaseModel):
    solid_mask: list[list[float]] = Field(..., description="Solid mask grid, shape 128x256 (0/1).")
    mach: float = Field(..., description="Freestream Mach number.")
    aoa: float | None = Field(
        default=None,
        description="Ignored for rotated2 model (AoA is encoded via rotated solid_mask).",
    )


class PredictResponse(BaseModel):
    mach: list[list[float]]
    pressure: list[list[float]]
    density: list[list[float]]
    temperature: list[list[float]]
    shock_indicator: list[list[float]]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_solid_mask(solid_mask: np.ndarray) -> np.ndarray:
    if solid_mask.shape != (NY, NX):
        raise HTTPException(
            status_code=422,
            detail=f"solid_mask must have shape ({NY}, {NX}), got {solid_mask.shape}.",
        )
    return solid_mask.astype(np.float32)


def build_model_input(solid_mask: np.ndarray, mach: float) -> torch.Tensor:
    mach_inf_map = np.full((NY, NX), mach, dtype=np.float32)
    stacked = np.stack([solid_mask, mach_inf_map], axis=0)
    return torch.from_numpy(stacked).unsqueeze(0)


class ModelBundle:
    def __init__(self) -> None:
        self.device = select_device()
        self.model: UNet | None = None
        self.output_mean: np.ndarray | None = None
        self.output_std: np.ndarray | None = None
        self.input_channels: int = INPUT_CHANNELS

    def load(self) -> None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        if not STATS_PATH.exists():
            raise FileNotFoundError(f"Output stats not found: {STATS_PATH}")

        stats = load_output_stats(STATS_PATH)
        self.output_mean = stats["mean"]
        self.output_std = stats["std"]

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        self.input_channels = int(config.get("input_channels", INPUT_CHANNELS))

        model = UNet(in_channels=self.input_channels, out_channels=5)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def predict(self, solid_mask: np.ndarray, mach: float) -> dict[str, list[list[float]]]:
        if self.model is None or self.output_mean is None or self.output_std is None:
            raise RuntimeError("Model is not loaded.")

        inputs = build_model_input(solid_mask, mach).to(self.device)
        predictions_norm = self.model(inputs)[0].detach().cpu().numpy()
        predictions = predictions_norm * self.output_std[:, None, None] + self.output_mean[:, None, None]

        return {key: predictions[index].tolist() for index, key in enumerate(OUTPUT_KEYS)}


bundle = ModelBundle()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    bundle.load()
    print(
        f"Loaded U-Net ({MODEL_TYPE}, in_channels={bundle.input_channels}) on device: {bundle.device}"
    )
    yield


app = FastAPI(title="flow_sketch inference", lifespan=lifespan)

_default_cors = ",".join(
    [
        "https://podobooks-ganghwa.github.io",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]
)
_cors_origins = os.environ.get("CORS_ORIGINS", _default_cors).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _cors_origins if origin.strip()],
    allow_origin_regex=r"https://.*\.github\.io",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(bundle.device),
        "checkpoint": str(CHECKPOINT_PATH),
        "model_loaded": bundle.model is not None,
        "model_type": MODEL_TYPE,
        "input_channels": bundle.input_channels,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    solid_mask = validate_solid_mask(np.asarray(request.solid_mask, dtype=np.float32))
    fields = bundle.predict(solid_mask, request.mach)
    return PredictResponse(**fields)
