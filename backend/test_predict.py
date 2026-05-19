#!/usr/bin/env python3
"""Send one .npz sample to the local /predict endpoint and print field shapes."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_DIR = PROJECT_ROOT / "datasets" / "processed" / "small_dataset_npz"
BASE_URL = "http://127.0.0.1:8000"


def fetch_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    npz_files = sorted(NPZ_DIR.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {NPZ_DIR}", file=sys.stderr)
        sys.exit(1)

    sample_path = npz_files[0]
    with np.load(sample_path) as data:
        solid_mask = data["solid_mask"].astype(float).tolist()
        mach = float(data["mach_inf_map"][0, 0])
        aoa = float(data["aoa_map"][0, 0])

    print(f"Sample: {sample_path.name}")
    print(f"Request mach={mach}, aoa={aoa}")

    try:
        health = fetch_json(f"{BASE_URL}/health")
        print("GET /health:", health)

        result = fetch_json(
            f"{BASE_URL}/predict",
            {"solid_mask": solid_mask, "mach": mach, "aoa": aoa},
        )
    except urllib.error.URLError as exc:
        print(f"Failed to reach backend at {BASE_URL}: {exc}", file=sys.stderr)
        print("Start the server with: uvicorn backend.app:app --reload", file=sys.stderr)
        sys.exit(1)

    print("POST /predict field shapes:")
    for key in ("mach", "pressure", "density", "temperature", "shock_indicator"):
        array = np.asarray(result[key], dtype=np.float32)
        print(f"  {key}: {array.shape}")


if __name__ == "__main__":
    main()
