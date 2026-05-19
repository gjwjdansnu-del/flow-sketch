"""Save prediction comparison images for U-Net training."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import OUTPUT_KEYS


def save_prediction_preview(
    inputs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    predictions: torch.Tensor | np.ndarray,
    output_path: Path,
    title: str,
    extent: tuple[float, float, float, float] = (-1.0, 3.0, -1.0, 1.0),
) -> Path:
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    solid_mask = inputs[0]
    field_names = list(OUTPUT_KEYS)
    n_fields = len(field_names)
    row_titles = ["Ground truth", "Prediction", "Absolute error"]

    fig, axes = plt.subplots(3, n_fields, figsize=(3.2 * n_fields, 8), sharex=True, sharey=True)
    if n_fields == 1:
        axes = np.array(axes).reshape(3, 1)

    for col, name in enumerate(field_names):
        gt = targets[col]
        pred = predictions[col]
        err = np.abs(gt - pred)
        panels = [gt, pred, err]
        cmaps = ["turbo", "turbo", "magma"]

        for row_index, (row_ax, panel, cmap, row_title) in enumerate(
            zip(axes, panels, cmaps, row_titles, strict=True)
        ):
            image = row_ax[col].imshow(panel, origin="lower", extent=extent, aspect="auto", cmap=cmap)
            if row_index == 0:
                row_ax[col].imshow(
                    solid_mask,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    cmap="gray",
                    alpha=0.25,
                    vmin=0.0,
                    vmax=1.0,
                )
            if col == 0:
                row_ax[col].set_ylabel(row_title)
            row_ax[col].set_xlabel("x")
            fig.colorbar(image, ax=row_ax[col], shrink=0.72)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
