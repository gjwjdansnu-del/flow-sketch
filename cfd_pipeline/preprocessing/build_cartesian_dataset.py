#!/usr/bin/env python3
"""Convert SU2 unstructured results into fixed-size Cartesian ML samples."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "cfd_pipeline" / "postprocessing"))

from plot_single_su2_result import find_field, read_vtu, triangles_from_vtu  # noqa: E402


DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "datasets" / "raw" / "su2_runs" / "small_dataset_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "small_dataset_npz"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "datasets" / "previews" / "processed"
GEOMETRY_DIR = PROJECT_ROOT / "datasets" / "raw" / "geometries"
RUNS_DIR = PROJECT_ROOT / "datasets" / "raw" / "su2_runs"

NX = 256
NY = 128
X_RANGE = (-1.0, 3.0)
Y_RANGE = (-1.0, 1.0)
PREVIEW_LIMIT = 10


@dataclass
class CaseRow:
    geometry: str
    geometry_type: str
    mach: float
    aoa: float
    status: str


def clean_header(value: str) -> str:
    return value.strip().strip('"').strip()


def mach_label(mach: float) -> str:
    return f"{mach:.1f}"


def aoa_label(aoa: float) -> str:
    if float(aoa).is_integer():
        return str(int(aoa))
    return f"{aoa:.1f}"


def run_dir_for_case(case: CaseRow) -> Path:
    return RUNS_DIR / f"{case.geometry}_M{mach_label(case.mach)}_A{aoa_label(case.aoa)}"


def sample_name(case: CaseRow) -> str:
    return f"{case.geometry}_M{mach_label(case.mach)}_A{aoa_label(case.aoa)}"


def is_processable_status(status: str) -> bool:
    return status.strip().lower() != "failed"


def read_summary(path: Path) -> list[CaseRow]:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV is missing: {path}")

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)

    rows: list[CaseRow] = []
    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for raw_row in reader:
            row = {clean_header(key): value for key, value in raw_row.items()}
            status = row.get("status", "")
            if not is_processable_status(status):
                continue
            case = CaseRow(
                geometry=row["geometry"],
                geometry_type=row["geometry_type"],
                mach=float(row["mach"]),
                aoa=float(row["aoa"]),
                status=status,
            )
            if not (run_dir_for_case(case) / "flow.vtu").exists():
                continue
            rows.append(case)

    rows.sort(key=lambda case: (case.geometry_type, case.geometry, case.mach, case.aoa))
    if not rows:
        raise ValueError(f"No processable cases found in {path}")
    return rows


def fixed_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(X_RANGE[0], X_RANGE[1], NX, dtype=np.float32)
    y = np.linspace(Y_RANGE[0], Y_RANGE[1], NY, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    flat_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return x, y, grid_x, flat_points


def read_geometry_points(case: CaseRow) -> np.ndarray:
    geometry_path = GEOMETRY_DIR / f"{case.geometry}.csv"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry CSV is missing: {geometry_path}")

    points: list[tuple[float, float]] = []
    with geometry_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))
    return np.asarray(points, dtype=np.float32)


def create_solid_mask(geometry_points: np.ndarray, flat_grid_points: np.ndarray) -> np.ndarray:
    path = MplPath(geometry_points)
    mask = path.contains_points(flat_grid_points)
    return mask.reshape((NY, NX)).astype(np.float32)


def interpolate_field(flow: Any, values: np.ndarray, flat_grid_points: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).squeeze()
    triangles = triangles_from_vtu(flow)
    triangulation = mtri.Triangulation(flow.points[:, 0], flow.points[:, 1], triangles)
    interpolator = mtri.LinearTriInterpolator(triangulation, values)
    grid_values = interpolator(flat_grid_points[:, 0], flat_grid_points[:, 1])
    array = np.asarray(grid_values.filled(np.nan), dtype=np.float32).reshape((NY, NX))

    if np.isnan(array).any():
        array = fill_nan_nearest(flow.points[:, :2], values, flat_grid_points, array)
    return array.astype(np.float32)


def fill_nan_nearest(
    source_points: np.ndarray,
    source_values: np.ndarray,
    flat_grid_points: np.ndarray,
    array: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return fill_nan_nearest_numpy(source_points, source_values, flat_grid_points, array)

    missing = np.isnan(array.ravel())
    if not np.any(missing):
        return array

    tree = cKDTree(source_points[:, :2])
    _distances, indices = tree.query(flat_grid_points[missing])
    filled = array.ravel()
    filled[missing] = source_values[indices]
    return filled.reshape((NY, NX))


def fill_nan_nearest_numpy(
    source_points: np.ndarray,
    source_values: np.ndarray,
    flat_grid_points: np.ndarray,
    array: np.ndarray,
) -> np.ndarray:
    missing = np.isnan(array.ravel())
    if not np.any(missing):
        return array

    query_points = flat_grid_points[missing]
    nearest_values = np.empty(len(query_points), dtype=np.float32)
    chunk_size = 2048
    source_xy = source_points[:, :2].astype(np.float32)
    source_values = source_values.astype(np.float32)

    for start in range(0, len(query_points), chunk_size):
        stop = min(start + chunk_size, len(query_points))
        chunk = query_points[start:stop].astype(np.float32)
        diff = chunk[:, None, :] - source_xy[None, :, :]
        distances_sq = np.sum(diff * diff, axis=2)
        nearest_indices = np.argmin(distances_sq, axis=1)
        nearest_values[start:stop] = source_values[nearest_indices]

    filled = array.ravel()
    filled[missing] = nearest_values
    return filled.reshape((NY, NX))


def get_required_field(flow: Any, label: str, candidates: list[str]) -> np.ndarray:
    match = find_field(flow.point_data, candidates)
    if match is None:
        raise ValueError(f"{label} field missing. Available fields: {list(flow.point_data.keys())}")
    return np.asarray(match[1]).squeeze()


def shock_indicator_from_density(density: np.ndarray, solid_mask: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(density)
    magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)
    magnitude[solid_mask > 0.5] = 0.0
    scale = float(np.percentile(magnitude[magnitude > 0], 99)) if np.any(magnitude > 0) else 1.0
    if scale <= 0.0:
        scale = 1.0
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)


def write_npz(
    output_path: Path,
    case: CaseRow,
    solid_mask: np.ndarray,
    mach: np.ndarray,
    pressure: np.ndarray,
    density: np.ndarray,
    temperature: np.ndarray,
    shock_indicator: np.ndarray,
) -> None:
    mach_inf_map = np.full((NY, NX), case.mach, dtype=np.float32)
    aoa_map = np.full((NY, NX), case.aoa, dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        solid_mask=solid_mask.astype(np.float32),
        mach_inf_map=mach_inf_map,
        aoa_map=aoa_map,
        mach=mach.astype(np.float32),
        pressure=pressure.astype(np.float32),
        density=density.astype(np.float32),
        temperature=temperature.astype(np.float32),
        shock_indicator=shock_indicator.astype(np.float32),
        geometry=np.asarray(case.geometry),
        geometry_type=np.asarray(case.geometry_type),
        mach_inf=np.asarray(case.mach, dtype=np.float32),
        aoa=np.asarray(case.aoa, dtype=np.float32),
        status=np.asarray(case.status),
    )


def save_preview(
    preview_path: Path,
    case: CaseRow,
    solid_mask: np.ndarray,
    mach: np.ndarray,
    pressure: np.ndarray,
    density: np.ndarray,
    shock_indicator: np.ndarray,
) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        ("solid_mask", solid_mask, "gray"),
        ("mach", mach, "turbo"),
        ("pressure", pressure, "turbo"),
        ("density", density, "turbo"),
        ("shock_indicator", shock_indicator, "magma"),
    ]

    fig, axes = plt.subplots(1, len(fields), figsize=(16, 3.5), sharex=True, sharey=True)
    extent = [X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]]
    for ax, (title, values, cmap) in zip(axes, fields, strict=True):
        image = ax.imshow(values, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        fig.colorbar(image, ax=ax, shrink=0.72)
    axes[0].set_ylabel("y")
    fig.suptitle(f"{case.geometry} M={mach_label(case.mach)} AoA={aoa_label(case.aoa)}")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)


def process_case(
    case: CaseRow,
    flat_grid_points: np.ndarray,
    output_dir: Path,
    preview_dir: Path,
    preview_index: int | None,
) -> Path:
    run_dir = run_dir_for_case(case)
    flow_path = run_dir / "flow.vtu"
    if not flow_path.exists():
        raise FileNotFoundError(f"flow.vtu is missing: {flow_path}")

    flow = read_vtu(flow_path)
    geometry_points = read_geometry_points(case)
    solid_mask = create_solid_mask(geometry_points, flat_grid_points)

    mach_values = get_required_field(flow, "Mach", ["Mach", "Mach_Number", "MachNumber"])
    pressure_values = get_required_field(flow, "Pressure", ["Pressure", "Static_Pressure", "Pressure_Pa"])
    density_values = get_required_field(flow, "Density", ["Density", "Rho", "rho"])
    temperature_values = get_required_field(flow, "Temperature", ["Temperature", "Static_Temperature"])

    mach = interpolate_field(flow, mach_values, flat_grid_points)
    pressure = interpolate_field(flow, pressure_values, flat_grid_points)
    density = interpolate_field(flow, density_values, flat_grid_points)
    temperature = interpolate_field(flow, temperature_values, flat_grid_points)

    for field in (mach, pressure, density, temperature):
        field[solid_mask > 0.5] = 0.0

    shock_indicator = shock_indicator_from_density(density, solid_mask)
    output_path = output_dir / f"{sample_name(case)}.npz"
    write_npz(output_path, case, solid_mask, mach, pressure, density, temperature, shock_indicator)

    if preview_index is not None:
        preview_path = preview_dir / f"{preview_index:03d}_{sample_name(case)}.png"
        save_preview(preview_path, case, solid_mask, mach, pressure, density, shock_indicator)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cartesian ML samples from SU2 VTU results.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"SU2 summary CSV (default: {DEFAULT_SUMMARY_CSV.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output .npz directory (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=DEFAULT_PREVIEW_DIR,
        help=f"Preview PNG directory (default: {DEFAULT_PREVIEW_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=PREVIEW_LIMIT,
        help=f"Number of preview PNGs to write (default: {PREVIEW_LIMIT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = args.summary if args.summary.is_absolute() else PROJECT_ROOT / args.summary
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    preview_dir = args.preview_dir if args.preview_dir.is_absolute() else PROJECT_ROOT / args.preview_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    _x, _y, _grid_x, flat_grid_points = fixed_grid()
    cases = read_summary(summary_csv)

    processed: list[Path] = []
    failed: list[tuple[str, str]] = []
    for index, case in enumerate(cases):
        try:
            preview_index = index if index < args.preview_limit else None
            output_path = process_case(
                case,
                flat_grid_points,
                output_dir,
                preview_dir,
                preview_index,
            )
            processed.append(output_path)
            print(f"[ok] {sample_name(case)} -> {output_path}")
        except Exception as exc:
            failed.append((sample_name(case), str(exc)))
            print(f"[failed] {sample_name(case)}: {exc}")

    print("\nCartesian dataset build complete.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Processed samples: {len(processed)}")
    print(f"Failed samples: {len(failed)}")
    print(f"Output folder: {output_dir}")
    print(f"Preview folder: {preview_dir}")
    if failed:
        print("Failures:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
