#!/usr/bin/env python3
"""Export flow_sketch U-Net checkpoint to ONNX for browser inference."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

MODELS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODELS_DIR.parent
sys.path.insert(0, str(MODELS_DIR))

from unet import UNet  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "unet_site.pt"
STATS_PATH = PROJECT_ROOT / "checkpoints" / "unet_site_stats.json"
ONNX_PATH = PROJECT_ROOT / "frontend" / "public" / "models" / "unet_site.onnx"
STATS_OUT_PATH = PROJECT_ROOT / "frontend" / "public" / "models" / "unet_site_stats.json"

IN_CHANNELS = 2
OUT_CHANNELS = 5
HEIGHT = 128
WIDTH = 256
OPSET = 17


def load_model() -> UNet:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    in_channels = int(config.get("input_channels", IN_CHANNELS))
    base_channels = int(config.get("base_channels", 32))

    model = UNet(in_channels=in_channels, out_channels=OUT_CHANNELS, base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_onnx(model: UNet, output_path: Path) -> None:
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, IN_CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    # New torch exporter may write a compact proto; re-save so weights are embedded.
    onnx_model = onnx.load(str(output_path))
    onnx.save(onnx_model, str(output_path))


def verify_onnx(model: UNet, onnx_path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping numerical verification.")
        return

    rng = np.random.default_rng(42)
    sample = rng.standard_normal((1, IN_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": sample})[0]

    diff = np.abs(torch_out - onnx_out)
    print(f"Verification max abs diff: {diff.max():.6e}")
    print(f"Verification mean abs diff: {diff.mean():.6e}")


def main() -> None:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Stats not found: {STATS_PATH}")

    model = load_model()
    export_onnx(model, ONNX_PATH)
    shutil.copy2(STATS_PATH, STATS_OUT_PATH)

    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX: {ONNX_PATH}")
    print(f"ONNX size: {size_mb:.2f} MB")
    print(f"Copied stats: {STATS_OUT_PATH}")

    if size_mb > 100:
        print("WARNING: ONNX file exceeds GitHub's 100 MB per-file limit.")
        print("Do not push this file to GitHub without Git LFS or external hosting.")

    verify_onnx(model, ONNX_PATH)


if __name__ == "__main__":
    main()
