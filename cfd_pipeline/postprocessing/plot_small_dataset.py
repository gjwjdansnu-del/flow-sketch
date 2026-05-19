#!/usr/bin/env python3
"""Plot all successful cases in the small SU2 dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from plot_single_su2_result import find_field, overlay_surface, read_vtu, triangles_from_vtu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SU2_RUNS_DIR = PROJECT_ROOT / "datasets" / "raw" / "su2_runs"
SUMMARY_CSV = SU2_RUNS_DIR / "small_dataset_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "previews" / "small_dataset"
GALLERY_DIR = OUTPUT_DIR / "galleries"


@dataclass
class DatasetCase:
    geometry: str
    geometry_type: str
    mach: float
    aoa: float
    run_dir: Path
    flow: object
    surface: object | None
    mach_values: np.ndarray
    pressure_values: np.ndarray
    density_values: np.ndarray


def clean_header(value: str) -> str:
    return value.strip().strip('"').strip()


def read_successful_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Small dataset summary CSV is missing: {path}")

    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [{clean_header(key): value for key, value in row.items()} for row in reader]

    successful = [row for row in rows if row.get("status") != "failed"]
    successful.sort(key=lambda row: (row["geometry_type"], row["geometry"], float(row["mach"])))
    if not successful:
        raise ValueError(f"No successful cases found in {path}")
    return successful


def mach_label(mach: float) -> str:
    return f"{mach:.1f}"


def aoa_label(aoa: float) -> str:
    if float(aoa).is_integer():
        return str(int(aoa))
    return f"{aoa:.1f}"


def run_dir_for_row(row: dict[str, str]) -> Path:
    mach = float(row["mach"])
    aoa = float(row["aoa"])
    return SU2_RUNS_DIR / f"{row['geometry']}_M{mach_label(mach)}_A{aoa_label(aoa)}"


def load_cases(rows: list[dict[str, str]]) -> list[DatasetCase]:
    cases: list[DatasetCase] = []

    for row in rows:
        run_dir = run_dir_for_row(row)
        flow_path = run_dir / "flow.vtu"
        surface_path = run_dir / "surface_flow.vtu"
        if not flow_path.exists():
            print(f"Skipping {run_dir.name}: missing flow.vtu")
            continue

        flow = read_vtu(flow_path)
        surface = read_vtu(surface_path) if surface_path.exists() else None

        mach_field = find_field(flow.point_data, ["Mach", "Mach_Number", "MachNumber"])
        pressure_field = find_field(flow.point_data, ["Pressure", "Static_Pressure", "Pressure_Pa"])
        density_field = find_field(flow.point_data, ["Density", "Rho", "rho"])
        missing = [
            name
            for name, field in (
                ("Mach", mach_field),
                ("Pressure", pressure_field),
                ("Density", density_field),
            )
            if field is None
        ]
        if missing:
            print(f"Skipping {run_dir.name}: missing fields {missing}")
            print(f"Available fields: {list(flow.point_data.keys())}")
            continue

        cases.append(
            DatasetCase(
                geometry=row["geometry"],
                geometry_type=row["geometry_type"],
                mach=float(row["mach"]),
                aoa=float(row["aoa"]),
                run_dir=run_dir,
                flow=flow,
                surface=surface,
                mach_values=np.asarray(mach_field[1]).squeeze(),
                pressure_values=np.asarray(pressure_field[1]).squeeze(),
                density_values=np.asarray(density_field[1]).squeeze(),
            )
        )

    if not cases:
        raise RuntimeError("No small dataset cases could be loaded for plotting.")
    return cases


def global_limits(cases: list[DatasetCase], values_getter: Callable[[DatasetCase], np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(values_getter(case)).ravel() for case in cases])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def plot_case_on_axis(
    ax: plt.Axes,
    case: DatasetCase,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
):
    triangles = triangles_from_vtu(case.flow)
    triangulation = mtri.Triangulation(case.flow.points[:, 0], case.flow.points[:, 1], triangles)
    plot = ax.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    overlay_surface(ax, case.surface)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, linewidth=0.25, alpha=0.25)
    ax.set_xlabel("x / chord")
    return plot


def plot_individual_field(
    case: DatasetCase,
    field_name: str,
    values: np.ndarray,
    limits: tuple[float, float],
) -> Path:
    output_dir = OUTPUT_DIR / f"{case.geometry}_M{mach_label(case.mach)}_A{aoa_label(case.aoa)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{field_name.lower()}.png"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = plot_case_on_axis(
        ax,
        case,
        values,
        limits[0],
        limits[1],
        f"{case.geometry} M={mach_label(case.mach)} {field_name}",
    )
    ax.set_ylabel("y / chord")
    fig.colorbar(plot, ax=ax, label=field_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def group_by_geometry_type(cases: list[DatasetCase]) -> dict[str, list[DatasetCase]]:
    grouped: dict[str, list[DatasetCase]] = {}
    for case in cases:
        grouped.setdefault(case.geometry_type, []).append(case)
    for group_cases in grouped.values():
        group_cases.sort(key=lambda case: case.mach)
    return dict(sorted(grouped.items()))


def plot_gallery(
    geometry_type: str,
    cases: list[DatasetCase],
    field_name: str,
    values_getter: Callable[[DatasetCase], np.ndarray],
    limits: tuple[float, float],
) -> Path:
    output_path = GALLERY_DIR / f"{geometry_type}_{field_name.lower()}_gallery.png"
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(cases), figsize=(4.0 * len(cases), 4.1), sharex=True, sharey=True)
    if len(cases) == 1:
        axes = [axes]

    last_plot = None
    for ax, case in zip(axes, cases, strict=True):
        last_plot = plot_case_on_axis(
            ax,
            case,
            values_getter(case),
            limits[0],
            limits[1],
            f"M={mach_label(case.mach)}",
        )

    axes[0].set_ylabel("y / chord")
    fig.suptitle(f"{geometry_type} {field_name}")
    if last_plot is not None:
        fig.colorbar(last_plot, ax=axes, shrink=0.82, label=field_name)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_successful_rows(SUMMARY_CSV)
    cases = load_cases(rows)

    field_specs = {
        "Mach": (
            lambda case: case.mach_values,
            global_limits(cases, lambda case: case.mach_values),
        ),
        "Pressure": (
            lambda case: case.pressure_values,
            global_limits(cases, lambda case: case.pressure_values),
        ),
        "Density": (
            lambda case: case.density_values,
            global_limits(cases, lambda case: case.density_values),
        ),
    }

    individual_outputs: list[Path] = []
    for case in cases:
        for field_name, (values_getter, limits) in field_specs.items():
            individual_outputs.append(plot_individual_field(case, field_name, values_getter(case), limits))

    gallery_outputs: list[Path] = []
    for geometry_type, geometry_cases in group_by_geometry_type(cases).items():
        for field_name, (values_getter, limits) in field_specs.items():
            gallery_outputs.append(plot_gallery(geometry_type, geometry_cases, field_name, values_getter, limits))

    print(f"Successful plotted cases: {len(cases)}")
    print(f"Individual plot directory: {OUTPUT_DIR}")
    print(f"Gallery directory: {GALLERY_DIR}")
    print(f"Individual images written: {len(individual_outputs)}")
    print(f"Gallery images written: {len(gallery_outputs)}")


if __name__ == "__main__":
    main()
