#!/usr/bin/env python3
"""Create comparison plots for the diamond airfoil Mach sweep."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from plot_single_su2_result import find_field, overlay_surface, read_vtu, triangles_from_vtu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SU2_RUNS_DIR = PROJECT_ROOT / "datasets" / "raw" / "su2_runs"
SUMMARY_CSV = SU2_RUNS_DIR / "mach_sweep_summary.csv"
PREVIEW_DIR = PROJECT_ROOT / "datasets" / "previews" / "su2_results"
GEOMETRY_NAME = "diamond_airfoil_000"
AOA_LABEL = "A0"


@dataclass
class SweepCase:
    mach: float
    run_dir: Path
    flow: object
    surface: object | None
    mach_values: np.ndarray
    pressure_values: np.ndarray


def clean_header(value: str) -> str:
    return value.strip().strip('"').strip()


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Mach sweep summary CSV is missing: {path}")

    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [{clean_header(key): value for key, value in row.items()} for row in reader]

    rows = [row for row in rows if row.get("geometry") == GEOMETRY_NAME]
    rows.sort(key=lambda row: float(row["mach"]))
    if not rows:
        raise ValueError(f"No rows for geometry {GEOMETRY_NAME} in {path}")
    return rows


def mach_label(mach: float) -> str:
    return f"{mach:.1f}"


def run_dir_for_mach(mach: float) -> Path:
    return SU2_RUNS_DIR / f"{GEOMETRY_NAME}_M{mach_label(mach)}_{AOA_LABEL}"


def load_cases(rows: list[dict[str, str]]) -> list[SweepCase]:
    cases: list[SweepCase] = []
    for row in rows:
        mach = float(row["mach"])
        run_dir = run_dir_for_mach(mach)
        flow_path = run_dir / "flow.vtu"
        surface_path = run_dir / "surface_flow.vtu"
        if not flow_path.exists():
            print(f"Skipping Mach {mach_label(mach)}: missing {flow_path}")
            continue

        flow = read_vtu(flow_path)
        surface = read_vtu(surface_path) if surface_path.exists() else None

        mach_field = find_field(flow.point_data, ["Mach", "Mach_Number", "MachNumber"])
        pressure_field = find_field(flow.point_data, ["Pressure", "Static_Pressure", "Pressure_Pa"])
        if mach_field is None:
            print(f"Skipping Mach {mach_label(mach)}: Mach field not found.")
            continue
        if pressure_field is None:
            print(f"Skipping Mach {mach_label(mach)}: Pressure field not found.")
            continue

        cases.append(
            SweepCase(
                mach=mach,
                run_dir=run_dir,
                flow=flow,
                surface=surface,
                mach_values=np.asarray(mach_field[1]).squeeze(),
                pressure_values=np.asarray(pressure_field[1]).squeeze(),
            )
        )

    if not cases:
        raise RuntimeError("No Mach sweep cases could be loaded.")
    return cases


def robust_limits(values: list[np.ndarray], lower: float = 1.0, upper: float = 99.0) -> tuple[float, float]:
    combined = np.concatenate([np.asarray(value).ravel() for value in values])
    vmin, vmax = np.percentile(combined, [lower, upper])
    if np.isclose(vmin, vmax):
        vmin = float(np.min(combined))
        vmax = float(np.max(combined))
    return float(vmin), float(vmax)


def plot_field_sweep(
    cases: list[SweepCase],
    field_name: str,
    values_getter,
    output_path: Path,
    colorbar_label: str,
) -> Path:
    values_list = [values_getter(case) for case in cases]
    vmin, vmax = robust_limits(values_list)

    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4.2), sharex=True, sharey=True)
    if len(cases) == 1:
        axes = [axes]

    last_plot = None
    for ax, case in zip(axes, cases, strict=True):
        triangles = triangles_from_vtu(case.flow)
        triangulation = mtri.Triangulation(case.flow.points[:, 0], case.flow.points[:, 1], triangles)
        last_plot = ax.tripcolor(
            triangulation,
            values_getter(case),
            shading="gouraud",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        overlay_surface(ax, case.surface)
        ax.set_title(f"M={mach_label(case.mach)}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.0, 3.0)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, linewidth=0.25, alpha=0.25)
        ax.set_xlabel("x / chord")

    axes[0].set_ylabel("y / chord")
    fig.suptitle(field_name)
    if last_plot is not None:
        fig.colorbar(last_plot, ax=axes, shrink=0.82, label=colorbar_label)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_coefficients(rows: list[dict[str, str]], output_path: Path) -> Path:
    mach_values: list[float] = []
    cd_values: list[float] = []
    cl_values: list[float] = []

    for row in rows:
        if not row.get("final_CD") or not row.get("final_CL"):
            continue
        mach_values.append(float(row["mach"]))
        cd_values.append(float(row["final_CD"]))
        cl_values.append(float(row["final_CL"]))

    if not mach_values:
        raise ValueError("No CL/CD values found in Mach sweep summary CSV.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mach_values, cd_values, marker="o", label="CD")
    ax.plot(mach_values, cl_values, marker="s", label="CL")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Mach")
    ax.set_ylabel("Coefficient")
    ax.set_title("Diamond airfoil coefficients vs Mach")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_summary_rows(SUMMARY_CSV)
    cases = load_cases(rows)

    mach_output = plot_field_sweep(
        cases,
        "Mach field sweep",
        lambda case: case.mach_values,
        PREVIEW_DIR / "mach_sweep_mach.png",
        "Mach",
    )
    pressure_output = plot_field_sweep(
        cases,
        "Pressure field sweep",
        lambda case: case.pressure_values,
        PREVIEW_DIR / "mach_sweep_pressure.png",
        "Pressure",
    )
    coefficients_output = plot_coefficients(rows, PREVIEW_DIR / "mach_sweep_coefficients.png")

    print("Wrote Mach sweep plots:")
    print(f"  {mach_output}")
    print(f"  {pressure_output}")
    print(f"  {coefficients_output}")


if __name__ == "__main__":
    main()
