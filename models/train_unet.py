#!/usr/bin/env python3
"""Train a U-Net on flow_sketch Cartesian .npz datasets."""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

MODELS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODELS_DIR.parent
sys.path.insert(0, str(MODELS_DIR))

from dataset import (  # noqa: E402
    INPUT_KEYS_OLD3,
    INPUT_KEYS_ROTATED2,
    FlowSketchDataset,
    OUTPUT_KEYS,
    compute_output_stats,
    list_npz_files,
    resolve_input_keys,
    save_output_stats,
)
from plot_predictions import save_prediction_preview  # noqa: E402
from unet import UNet  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "small_dataset_npz"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "unet_small_dataset.pt"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "datasets" / "previews" / "training"
LEARNING_RATE = 1e-3
SHOCK_WEIGHT = 2.0
VAL_FRACTION = 0.2
PREVIEW_EVERY = 5
SEED = 42
BASE_CHANNELS = 32


@dataclass
class EpochMetrics:
    loss: float
    epoch_seconds: float
    avg_batch_seconds: float
    num_batches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flow_sketch U-Net on Cartesian npz data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing .npz training samples",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to save model checkpoint (.pt)",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=DEFAULT_PREVIEW_DIR,
        help="Directory for loss curve and epoch preview PNGs",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="JSON path for train-set output mean/std (default: next to checkpoint)",
    )
    parser.add_argument("--preview-every", type=int, default=PREVIEW_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--shock-weight", type=float, default=SHOCK_WEIGHT)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load all .npz samples into RAM once at dataset initialization",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default 0 for macOS/MPS safety)",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "old3", "rotated2"),
        default="auto",
        help="Input channel layout: old3 (mask+mach+aoa) or rotated2 (mask+mach)",
    )
    parser.add_argument(
        "--input-channels",
        default="auto",
        help='Model input channels: "auto", 2, or 3',
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_training_input(
    all_paths: list[Path],
    input_mode: str,
    input_channels_arg: str,
) -> tuple[int, tuple[str, ...], str]:
    detected_keys = resolve_input_keys(all_paths, input_mode=input_mode)  # type: ignore[arg-type]

    if input_channels_arg == "auto":
        return len(detected_keys), detected_keys, input_mode

    requested = int(input_channels_arg)
    if requested == 2:
        return 2, INPUT_KEYS_ROTATED2, "rotated2"
    if requested == 3:
        return 3, INPUT_KEYS_OLD3, "old3"
    raise ValueError(f"Unsupported --input-channels value: {input_channels_arg}")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def split_paths(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_paths = shuffled[:val_count]
    train_paths = shuffled[val_count:]
    if not train_paths:
        raise ValueError("Train split is empty. Reduce val_fraction or add more samples.")
    return train_paths, val_paths


def weighted_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, shock_weight: float) -> torch.Tensor:
    channel_weights = torch.ones(len(OUTPUT_KEYS), device=predictions.device, dtype=predictions.dtype)
    shock_index = OUTPUT_KEYS.index("shock_indicator")
    channel_weights[shock_index] = shock_weight
    weights = channel_weights[:, None, None]
    return (torch.abs(predictions - targets) * weights).mean()


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    shock_weight: float,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    batch_seconds = 0.0
    epoch_start = time.perf_counter()

    for inputs, targets, _names in loader:
        batch_start = time.perf_counter()
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()

        predictions = model(inputs)
        loss = weighted_l1_loss(predictions, targets, shock_weight)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        batch_seconds += time.perf_counter() - batch_start

    epoch_seconds = time.perf_counter() - epoch_start
    avg_batch_seconds = batch_seconds / max(total_batches, 1)
    return EpochMetrics(
        loss=total_loss / max(total_batches, 1),
        epoch_seconds=epoch_seconds,
        avg_batch_seconds=avg_batch_seconds,
        num_batches=total_batches,
    )


def probe_batch_size(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    shock_weight: float,
) -> bool:
    """Run one training batch; return False if OOM."""
    model.train()
    try:
        inputs, targets, _names = next(iter(train_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = weighted_l1_loss(predictions, targets, shock_weight)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return True
    except RuntimeError as exc:
        if is_oom_error(exc):
            return False
        raise


def build_dataloaders(
    train_dataset: FlowSketchDataset,
    val_dataset: FlowSketchDataset,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def resolve_batch_size(
    model: nn.Module,
    train_dataset: FlowSketchDataset,
    val_dataset: FlowSketchDataset,
    requested_batch_size: int,
    num_workers: int,
    device: torch.device,
    shock_weight: float,
) -> int:
    candidates: list[int] = []
    for size in (requested_batch_size, 16, 8, 4):
        if size not in candidates and size <= len(train_dataset):
            candidates.append(size)
    if not candidates:
        candidates = [min(4, len(train_dataset))]

    initial_state = copy.deepcopy(model.state_dict())
    for batch_size in candidates:
        model.load_state_dict(initial_state)
        train_loader, _val_loader = build_dataloaders(
            train_dataset, val_dataset, batch_size, num_workers
        )
        if probe_batch_size(model, train_loader, device, shock_weight):
            if batch_size != requested_batch_size:
                print(
                    f"Batch size {requested_batch_size} failed (OOM); "
                    f"using batch_size={batch_size} instead."
                )
            return batch_size
        print(f"Batch size {batch_size} failed (OOM); trying smaller batch.")

    raise RuntimeError("Could not find a batch size that fits in device memory.")


@torch.no_grad()
def save_epoch_preview(
    model: nn.Module,
    dataset: FlowSketchDataset,
    device: torch.device,
    epoch: int,
    preview_dir: Path,
) -> Path:
    model.eval()
    inputs, targets, sample_name = dataset[0]
    inputs_batch = inputs.unsqueeze(0).to(device)
    targets_batch = targets.unsqueeze(0).to(device)
    predictions_norm = model(inputs_batch)
    predictions = dataset.denormalize(predictions_norm[0])
    targets_denorm = dataset.denormalize(targets_batch[0])

    output_path = preview_dir / f"pred_epoch_{epoch:03d}.png"
    return save_prediction_preview(
        inputs.cpu().numpy(),
        targets_denorm,
        predictions,
        output_path,
        title=f"Validation preview epoch {epoch} ({sample_name})",
    )


def plot_loss_curves(train_losses: list[float], val_losses: list[float], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted L1 loss")
    ax.set_title("U-Net training loss")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_losses: list[float],
    val_losses: list[float],
    stats: dict,
    saved_stats_path: Path,
    train_paths: list[Path],
    val_paths: list[Path],
    config: dict,
    best_val_loss: float | None = None,
    best_epoch: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "output_stats": stats,
        "output_stats_path": str(saved_stats_path),
        "train_paths": [path_item.name for path_item in train_paths],
        "val_paths": [path_item.name for path_item in val_paths],
        "config": config,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    checkpoint_path = resolve_path(args.checkpoint)
    preview_dir = resolve_path(args.preview_dir)
    stats_path = (
        resolve_path(args.stats_path)
        if args.stats_path
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_stats.json")
    )
    best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best.pt")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = select_device()
    print(f"Device: {device}")

    all_paths = list_npz_files(data_dir)
    train_paths, val_paths = split_paths(all_paths, args.val_fraction, args.seed)
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")

    in_channels, input_keys, resolved_input_mode = resolve_training_input(
        all_paths,
        args.input_mode,
        str(args.input_channels),
    )
    print(f"Input mode: {resolved_input_mode} ({in_channels} channels: {', '.join(input_keys)})")

    stats = compute_output_stats(train_paths, input_keys=input_keys)
    saved_stats_path = save_output_stats(stats, stats_path)
    print(f"Saved output stats: {saved_stats_path}")

    print("Building datasets...")
    train_dataset = FlowSketchDataset(
        train_paths,
        stats=stats,
        preload=args.preload,
        input_mode=resolved_input_mode,  # type: ignore[arg-type]
        input_keys=input_keys,
    )
    val_dataset = FlowSketchDataset(
        val_paths,
        stats=stats,
        preload=args.preload,
        input_mode=resolved_input_mode,  # type: ignore[arg-type]
        input_keys=input_keys,
    )

    if args.preload:
        total_bytes = train_dataset.preloaded_bytes + val_dataset.preloaded_bytes
        total_samples = train_dataset.preloaded_sample_count + val_dataset.preloaded_sample_count
        print(
            f"Preload enabled: {total_samples} samples in RAM "
            f"({total_bytes / (1024 ** 3):.2f} GiB float32 arrays)"
        )
    else:
        print("Preload disabled: samples read from disk each access.")

    model = UNet(
        in_channels=in_channels,
        out_channels=5,
        base_channels=args.base_channels,
    ).to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")

    initial_state = copy.deepcopy(model.state_dict())
    batch_size = resolve_batch_size(
        model,
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_workers,
        device,
        args.shock_weight,
    )
    model.load_state_dict(initial_state)
    print(f"Using batch_size={batch_size}, num_workers={args.num_workers}")

    train_loader, val_loader = build_dataloaders(
        train_dataset, val_dataset, batch_size, args.num_workers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses: list[float] = []
    val_losses: list[float] = []
    preview_paths: list[Path] = []
    loss_plot_path = preview_dir / "unet_loss.png"
    epoch_times: list[float] = []
    train_epoch_times: list[float] = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.shock_weight)
        val_metrics = run_epoch(model, val_loader, None, device, args.shock_weight)
        train_losses.append(train_metrics.loss)
        val_losses.append(val_metrics.loss)
        epoch_times.append(train_metrics.epoch_seconds + val_metrics.epoch_seconds)
        train_epoch_times.append(train_metrics.epoch_seconds)

        train_samples_per_sec = len(train_paths) / max(train_metrics.epoch_seconds, 1e-9)
        elapsed = time.perf_counter() - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_metrics.loss:.6f} | val={val_metrics.loss:.6f} | "
            f"epoch_time={format_duration(epoch_times[-1])} | "
            f"train_batch_avg={train_metrics.avg_batch_seconds * 1000:.1f}ms | "
            f"train_samples/s={train_samples_per_sec:.1f} | "
            f"ETA={format_duration(eta_seconds)}"
        )

        if epoch % args.preview_every == 0 or epoch == args.epochs:
            preview_path = save_epoch_preview(model, val_dataset, device, epoch, preview_dir)
            preview_paths.append(preview_path)

    total_time = time.perf_counter() - start_time

    config = {
        "data_dir": str(data_dir),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "shock_weight": args.shock_weight,
        "base_channels": args.base_channels,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "preload": args.preload,
        "num_workers": args.num_workers,
        "parameter_count": param_count,
        "input_channels": in_channels,
        "input_mode": resolved_input_mode,
        "input_keys": list(input_keys),
    }

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        train_losses,
        val_losses,
        stats,
        saved_stats_path,
        train_paths,
        val_paths,
        config,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
    )

    if best_state_dict is not None:
        best_model = UNet(
            in_channels=in_channels,
            out_channels=5,
            base_channels=args.base_channels,
        ).to(device)
        best_model.load_state_dict(best_state_dict)
        save_checkpoint(
            best_checkpoint_path,
            best_model,
            optimizer,
            train_losses,
            val_losses,
            stats,
            saved_stats_path,
            train_paths,
            val_paths,
            config,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
        )

    plot_loss_curves(train_losses, val_losses, loss_plot_path)

    avg_epoch_time = sum(epoch_times) / max(len(epoch_times), 1)
    avg_train_epoch_time = sum(train_epoch_times) / max(len(train_epoch_times), 1)
    avg_train_samples_per_sec = len(train_paths) / max(avg_train_epoch_time, 1e-9)
    estimate_20_epochs = avg_epoch_time * 20

    print("\nTraining complete.")
    print(f"Device: {device}")
    print(f"Preload: {args.preload}")
    print(f"Input channels: {in_channels}")
    print(f"Batch size: {batch_size}")
    print(f"Model parameters: {param_count:,}")
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    print(f"Average epoch time: {format_duration(avg_epoch_time)}")
    print(f"Average train throughput: {avg_train_samples_per_sec:.1f} samples/s")
    print(f"Total training time: {format_duration(total_time)}")
    print(f"Estimated time for 20 epochs: {format_duration(estimate_20_epochs)}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Output stats: {saved_stats_path}")
    print(f"Loss curve: {loss_plot_path}")
    print("Prediction previews:")
    for path in preview_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
