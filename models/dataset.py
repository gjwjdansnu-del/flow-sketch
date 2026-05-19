"""Dataset loader for flow_sketch Cartesian .npz samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_DIR = PROJECT_ROOT / "datasets" / "processed" / "small_dataset_npz"
STATS_PATH = PROJECT_ROOT / "datasets" / "processed" / "small_dataset_npz" / "output_stats.json"

INPUT_KEYS_OLD3 = ("solid_mask", "mach_inf_map", "aoa_map")
INPUT_KEYS_ROTATED2 = ("solid_mask", "mach_inf_map")
OUTPUT_KEYS = ("mach", "pressure", "density", "temperature", "shock_indicator")

InputMode = Literal["auto", "old3", "rotated2"]


def list_npz_files(data_dir: Path | None = None) -> list[Path]:
    directory = data_dir or NPZ_DIR
    return sorted(directory.glob("*.npz"))


def detect_input_keys(path: Path) -> tuple[str, ...]:
    with np.load(path) as data:
        if "aoa_map" in data.files:
            return INPUT_KEYS_OLD3
        return INPUT_KEYS_ROTATED2


def resolve_input_keys(
    paths: list[Path],
    input_mode: InputMode = "auto",
) -> tuple[str, ...]:
    if not paths:
        raise ValueError("No .npz files provided.")

    if input_mode == "old3":
        return INPUT_KEYS_OLD3
    if input_mode == "rotated2":
        return INPUT_KEYS_ROTATED2

    keys = detect_input_keys(paths[0])
    for path in paths[1:]:
        if detect_input_keys(path) != keys:
            raise ValueError(
                f"Mixed input formats detected in dataset. Example mismatch: {paths[0].name} vs {path.name}"
            )
    return keys


def load_sample_arrays(path: Path, input_keys: tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    keys = input_keys or detect_input_keys(path)
    with np.load(path) as data:
        inputs = np.stack([data[key].astype(np.float32) for key in keys], axis=0)
        outputs = np.stack([data[key].astype(np.float32) for key in OUTPUT_KEYS], axis=0)
    return inputs, outputs


def compute_output_stats(
    paths: list[Path],
    input_keys: tuple[str, ...] | None = None,
) -> dict[str, list[float]]:
    if not paths:
        raise ValueError("No .npz files provided for statistics computation.")

    sums = np.zeros(len(OUTPUT_KEYS), dtype=np.float64)
    sums_sq = np.zeros(len(OUTPUT_KEYS), dtype=np.float64)
    counts = np.zeros(len(OUTPUT_KEYS), dtype=np.float64)

    for path in paths:
        _inputs, outputs = load_sample_arrays(path, input_keys=input_keys)
        for channel_index, channel in enumerate(outputs):
            values = channel.ravel()
            sums[channel_index] += values.sum()
            sums_sq[channel_index] += np.square(values).sum()
            counts[channel_index] += values.size

    means = sums / counts
    variances = np.maximum(sums_sq / counts - np.square(means), 1e-12)
    stds = np.sqrt(variances)
    stds = np.maximum(stds, 1e-6)

    return {
        "mean": means.astype(np.float32).tolist(),
        "std": stds.astype(np.float32).tolist(),
        "keys": list(OUTPUT_KEYS),
    }


def save_output_stats(stats: dict[str, list[float]], path: Path | None = None) -> Path:
    output_path = path or STATS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return output_path


def load_output_stats(path: Path | None = None) -> dict[str, np.ndarray]:
    stats_path = path or STATS_PATH
    if not stats_path.exists():
        raise FileNotFoundError(f"Output statistics file is missing: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    return {
        "mean": np.asarray(payload["mean"], dtype=np.float32),
        "std": np.asarray(payload["std"], dtype=np.float32),
        "keys": payload.get("keys", list(OUTPUT_KEYS)),
    }


class FlowSketchDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        stats: dict[str, np.ndarray] | list | None = None,
        preload: bool = False,
        input_mode: InputMode = "auto",
        input_keys: tuple[str, ...] | None = None,
    ) -> None:
        if not paths:
            raise ValueError("Dataset requires at least one .npz file.")
        self.paths = paths
        self.preload = preload
        self.input_mode = input_mode
        self.input_keys = input_keys or resolve_input_keys(paths, input_mode=input_mode)
        self.num_input_channels = len(self.input_keys)
        self.stats = stats or load_output_stats()
        mean = np.asarray(self.stats["mean"], dtype=np.float32)
        std = np.asarray(self.stats["std"], dtype=np.float32)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

        self._cache: list[tuple[np.ndarray, np.ndarray, str]] | None = None
        if preload:
            self._cache = []
            for path in paths:
                inputs, outputs = load_sample_arrays(path, input_keys=self.input_keys)
                self._cache.append((inputs, outputs, path.stem))

    @property
    def preloaded_sample_count(self) -> int:
        return len(self._cache) if self._cache is not None else 0

    @property
    def preloaded_bytes(self) -> int:
        if self._cache is None:
            return 0
        total = 0
        for inputs, outputs, _name in self._cache:
            total += inputs.nbytes + outputs.nbytes
        return total

    def __len__(self) -> int:
        return len(self.paths)

    def _arrays_for_index(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        if self._cache is not None:
            return self._cache[index]
        path = self.paths[index]
        inputs, outputs = load_sample_arrays(path, input_keys=self.input_keys)
        return inputs, outputs, path.stem

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        inputs, outputs, name = self._arrays_for_index(index)
        inputs_tensor = torch.from_numpy(inputs)
        outputs_norm = (outputs - self.mean) / self.std
        outputs_tensor = torch.from_numpy(outputs_norm.astype(np.float32))
        return inputs_tensor, outputs_tensor, name

    def denormalize(self, outputs: torch.Tensor | np.ndarray) -> np.ndarray:
        array = outputs.detach().cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
        return array * self.std + self.mean
