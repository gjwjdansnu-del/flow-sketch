#!/usr/bin/env python3
"""Plot representative overnight SU2 cases (one geometry per family, four Mach/AoA combos)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_single_su2_result import (  # noqa: E402
    find_field,
    overlay_surface,
    read_vtu,
    triangles_from_vtu,
)


GEOMETRY_DIR = PROJECT_ROOT / "datasets" / "raw" / "geometries"
SU2_RUNS_DIR = PROJECT_ROOT / "datasets" / "raw" / "su2_runs"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "previews" / "representative_cases"

FAMILIES = [
    "biconvex_airfoil",
    "blunt_capsule",
    "diamond_airfoil",
    "ellipse",
    "naca_00xx",
    "wedge",
    "random_angular_body",
    "random_asymmetric_body",
    "random_bluff_body",
    "random_smooth_body",
    "random_star_body",
    "random_sharp_star_body",
    "random_thin_body",
    "weird_random",
]

PLOT_CONDITIONS: list[tuple[float, float, str]] = [
    (1.5, 0.0, "M1.5_A0"),
    (1.5, 10.0, "M1.5_A10"),
    (5.0, 0.0, "M5.0_A0"),
    (5.0, 10.0, "M5.0_A10"),
]

FIELD_SPECS: dict[str, list[str]] = {
    "mach": ["Mach", "Mach_Number", "MachNumber"],
    "pressure": ["Pressure", "Static_Pressure", "Pressure_Pa"],
    "density": ["Density", "Rho", "rho"],
}


@dataclass
class LoadedCase:
    geometry: str
    family: str
    mach: float
    aoa: float
    condition_label: str
    run_dir: Path
    flow: object
    surface: object | None
    mach_values: np.ndarray
    pressure_values: np.ndarray
    density_values: np.ndarray


def mach_label(mach: float) -> str:
    return f"{mach:.1f}"


def aoa_label(aoa: float) -> str:
    if float(aoa).is_integer():
        return str(int(aoa))
    return f"{aoa:.1f}"


def run_dir_for(geometry: str, mach: float, aoa: float) -> Path:
    return SU2_RUNS_DIR / f"{geometry}_M{mach_label(mach)}_A{aoa_label(aoa)}"


def geometries_for_family(family: str) -> list[str]:
    return sorted(path.stem for path in GEOMETRY_DIR.glob(f"{family}_*.csv"))


def pick_representative_geometry(family: str) -> str | None:
    names = geometries_for_family(family)
    if not names:
        return None
    preferred = f"{family}_000"
    if preferred in names:
        return preferred
    return names[0]


def values_limits(cases: list[LoadedCase], values_getter: Callable[[LoadedCase], np.ndarray]) -> tuple[float, float]:
    arrays = [np.asarray(values_getter(case)).ravel() for case in cases]
    if not arrays:
        return 0.0, 1.0
    values = np.concatenate(arrays)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def load_case(geometry: str, family: str, mach: float, aoa: float, condition_label: str) -> LoadedCase | None:
    run_dir = run_dir_for(geometry, mach, aoa)
    flow_path = run_dir / "flow.vtu"
    if not flow_path.exists():
        return None

    flow = read_vtu(flow_path)
    surface_path = run_dir / "surface_flow.vtu"
    surface = read_vtu(surface_path) if surface_path.exists() else None

    mach_field = find_field(flow.point_data, FIELD_SPECS["mach"])
    pressure_field = find_field(flow.point_data, FIELD_SPECS["pressure"])
    density_field = find_field(flow.point_data, FIELD_SPECS["density"])
    if mach_field is None or pressure_field is None or density_field is None:
        missing = [
            name
            for name, field in (
                ("Mach", mach_field),
                ("Pressure", pressure_field),
                ("Density", density_field),
            )
            if field is None
        ]
        print(f"[skip] {run_dir.name}: missing fields {missing}")
        return None

    return LoadedCase(
        geometry=geometry,
        family=family,
        mach=mach,
        aoa=aoa,
        condition_label=condition_label,
        run_dir=run_dir,
        flow=flow,
        surface=surface,
        mach_values=np.asarray(mach_field[1]).squeeze(),
        pressure_values=np.asarray(pressure_field[1]).squeeze(),
        density_values=np.asarray(density_field[1]).squeeze(),
    )


def plot_on_axis(
    ax: plt.Axes,
    case: LoadedCase,
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


def save_individual_plot(
    case: LoadedCase,
    field_key: str,
    field_label: str,
    values: np.ndarray,
    limits: tuple[float, float],
) -> Path:
    case_dir = OUTPUT_DIR / f"{case.geometry}_M{mach_label(case.mach)}_A{aoa_label(case.aoa)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_dir / f"{field_key}.png"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = plot_on_axis(
        ax,
        case,
        values,
        limits[0],
        limits[1],
        f"{case.geometry} {case.condition_label} {field_label}",
    )
    ax.set_ylabel("y / chord")
    fig.colorbar(plot, ax=ax, label=field_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_gallery(
    geometry: str,
    cases: list[LoadedCase],
    field_key: str,
    field_label: str,
    values_getter: Callable[[LoadedCase], np.ndarray],
    limits: tuple[float, float],
) -> Path:
    output_path = OUTPUT_DIR / f"{geometry}_{field_key}_gallery.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    last_plot = None
    for ax, case in zip(axes_flat, cases, strict=True):
        last_plot = plot_on_axis(
            ax,
            case,
            values_getter(case),
            limits[0],
            limits[1],
            case.condition_label,
        )

    axes[0, 0].set_ylabel("y / chord")
    axes[1, 0].set_ylabel("y / chord")
    fig.suptitle(f"{geometry} — {field_label}")
    if last_plot is not None:
        fig.colorbar(last_plot, ax=axes, shrink=0.82, label=field_label)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def process_geometry(geometry: str, family: str) -> tuple[list[LoadedCase], list[str]]:
    loaded: list[LoadedCase] = []
    missing: list[str] = []

    for mach, aoa, condition_label in PLOT_CONDITIONS:
        case = load_case(geometry, family, mach, aoa, condition_label)
        if case is None:
            missing.append(f"{geometry}_{condition_label} ({run_dir_for(geometry, mach, aoa)})")
            print(f"[missing] {geometry} {condition_label}: no usable flow.vtu")
            continue
        loaded.append(case)

    if not loaded:
        return loaded, missing

    value_getters: dict[str, tuple[Callable[[LoadedCase], np.ndarray], str]] = {
        "mach": (lambda case: case.mach_values, "Mach"),
        "pressure": (lambda case: case.pressure_values, "Pressure"),
        "density": (lambda case: case.density_values, "Density"),
    }

    for field_key, (getter, field_label) in value_getters.items():
        limits = values_limits(loaded, getter)
        for case in loaded:
            save_individual_plot(case, field_key, field_label, getter(case), limits)
        if len(loaded) == len(PLOT_CONDITIONS):
            save_gallery(geometry, loaded, field_key, field_label, getter, limits)
        else:
            print(f"[skip gallery] {geometry} {field_key}: only {len(loaded)}/4 cases available")

    return loaded, missing


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, str]] = []
    all_loaded: list[LoadedCase] = []
    all_missing: list[str] = []

    for family in FAMILIES:
        geometry = pick_representative_geometry(family)
        if geometry is None:
            print(f"[skip family] {family}: no geometry CSV found")
            continue
        selected.append((family, geometry))
        loaded, missing = process_geometry(geometry, family)
        all_loaded.extend(loaded)
        all_missing.extend(missing)

    print("\n=== Representative case plots ===")
    print("Selected representative geometries:")
    for family, geometry in selected:
        print(f"  - {family}: {geometry}")
    print(f"Plotted cases (flow fields saved): {len(all_loaded)}")
    print(f"Expected per geometry (if complete): {len(PLOT_CONDITIONS)} conditions × 3 fields")
    print(f"Representative geometries: {len(selected)}")
    if all_missing:
        print(f"Missing cases: {len(all_missing)}")
        for item in all_missing:
            print(f"  - {item}")
    else:
        print("Missing cases: none")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
