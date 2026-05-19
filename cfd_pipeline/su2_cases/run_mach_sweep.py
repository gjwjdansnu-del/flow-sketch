#!/usr/bin/env python3
"""Run a small SU2 Mach sweep for one geometry."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from run_single_su2 import convert_mesh_to_su2, parse_convergence_status, parse_history


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_NAME = "diamond_airfoil_000"
INPUT_MESH = PROJECT_ROOT / "datasets" / "raw" / "meshes" / f"{GEOMETRY_NAME}.msh"
SU2_RUNS_DIR = PROJECT_ROOT / "datasets" / "raw" / "su2_runs"
SUMMARY_CSV = SU2_RUNS_DIR / "mach_sweep_summary.csv"
MACH_NUMBERS = [1.5, 2.0, 3.0, 4.0, 5.0]
AOA_DEGREES = 0.0


SUMMARY_COLUMNS = [
    "geometry",
    "mach",
    "aoa",
    "status",
    "su2_runtime_s",
    "total_runtime_s",
    "iterations_completed",
    "final_CL",
    "final_CD",
    "final_rms_Rho",
    "final_rms_RhoU",
    "final_rms_RhoV",
    "final_rms_RhoE",
]


def mach_label(mach: float) -> str:
    return f"{mach:.1f}"


def aoa_label(aoa: float) -> str:
    if float(aoa).is_integer():
        return str(int(aoa))
    return f"{aoa:.1f}"


def run_dir_for_case(mach: float, aoa: float) -> Path:
    return SU2_RUNS_DIR / f"{GEOMETRY_NAME}_M{mach_label(mach)}_A{aoa_label(aoa)}"


def config_text(mesh_filename: str, mach: float, aoa: float) -> str:
    return f"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% flow_sketch Mach sweep SU2 Euler test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

MACH_NUMBER= {mach}
AOA= {aoa}
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15
GAMMA_VALUE= 1.4

REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.0
REF_AREA= 1.0

MARKER_EULER= ( wall )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( wall )
MARKER_MONITORING= ( wall )

NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 0.5
CFL_ADAPT= NO
ITER= 2000

LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= LU_SGS
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 5

CONV_NUM_METHOD_FLOW= JST
JST_SENSOR_COEFF= ( 0.5, 0.02 )
TIME_DISCRE_FLOW= EULER_IMPLICIT

CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= -8
CONV_STARTITER= 10
SCREEN_OUTPUT= (INNER_ITER, ITER_TIME, RMS_DENSITY, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, RMS_ENERGY)
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)

MESH_FILENAME= {mesh_filename}
MESH_FORMAT= SU2
MESH_OUT_FILENAME= mesh_out
SOLUTION_FILENAME= solution_flow
TABULAR_FORMAT= CSV
CONV_FILENAME= history
RESTART_FILENAME= restart_flow
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
OUTPUT_WRT_FREQ= 100
"""


def run_su2(run_dir: Path, config_path: Path) -> subprocess.CompletedProcess[str]:
    su2_executable = shutil.which("SU2_CFD")
    if su2_executable is None:
        raise RuntimeError("SU2_CFD was not found on PATH.")

    result = subprocess.run(
        [su2_executable, config_path.name],
        cwd=run_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    (run_dir / "log_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (run_dir / "log_stderr.txt").write_text(result.stderr, encoding="utf-8")
    return result


def blank_summary_row(mach: float, aoa: float, status: str) -> dict[str, Any]:
    return {
        "geometry": GEOMETRY_NAME,
        "mach": mach,
        "aoa": aoa,
        "status": status,
        "su2_runtime_s": "",
        "total_runtime_s": "",
        "iterations_completed": "",
        "final_CL": "",
        "final_CD": "",
        "final_rms_Rho": "",
        "final_rms_RhoU": "",
        "final_rms_RhoV": "",
        "final_rms_RhoE": "",
    }


def make_summary_row(
    mach: float,
    aoa: float,
    status: str,
    timing: dict[str, float],
    history_summary: dict[str, Any],
) -> dict[str, Any]:
    coefficients = history_summary.get("final_coefficients", {}) or {}
    residuals = history_summary.get("final_residuals", {}) or {}
    return {
        "geometry": GEOMETRY_NAME,
        "mach": mach,
        "aoa": aoa,
        "status": status,
        "su2_runtime_s": timing["su2_runtime_seconds"],
        "total_runtime_s": timing["total_script_runtime_seconds"],
        "iterations_completed": history_summary.get("iterations_completed", ""),
        "final_CL": coefficients.get("CL", ""),
        "final_CD": coefficients.get("CD", ""),
        "final_rms_Rho": residuals.get("rms[Rho]", ""),
        "final_rms_RhoU": residuals.get("rms[RhoU]", ""),
        "final_rms_RhoV": residuals.get("rms[RhoV]", ""),
        "final_rms_RhoE": residuals.get("rms[RhoE]", ""),
    }


def write_timing(
    run_dir: Path,
    timing: dict[str, float],
    history_summary: dict[str, Any],
    status: str,
    mach: float,
    aoa: float,
) -> None:
    payload = {
        "geometry": GEOMETRY_NAME,
        "mach": mach,
        "aoa": aoa,
        "mesh_conversion_seconds": timing["mesh_conversion_seconds"],
        "config_writing_seconds": timing["config_writing_seconds"],
        "su2_runtime_seconds": timing["su2_runtime_seconds"],
        "total_script_runtime_seconds": timing["total_script_runtime_seconds"],
        "convergence_status": status,
        "history_summary": history_summary,
    }
    (run_dir / "timing.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_case(mach: float, aoa: float = AOA_DEGREES) -> dict[str, Any]:
    if not INPUT_MESH.exists():
        raise FileNotFoundError(
            f"Input mesh is missing: {INPUT_MESH}\n"
            "Run cfd_pipeline/meshing/mesh_single_geometry.py first."
        )

    run_dir = run_dir_for_case(mach, aoa)
    run_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    mesh_copy = run_dir / INPUT_MESH.name
    su2_mesh = run_dir / f"{GEOMETRY_NAME}.su2"
    config_path = run_dir / "config.cfg"
    shutil.copy2(INPUT_MESH, mesh_copy)

    conversion_start = time.perf_counter()
    convert_mesh_to_su2(mesh_copy, su2_mesh)
    mesh_conversion_seconds = time.perf_counter() - conversion_start

    config_start = time.perf_counter()
    config_path.write_text(config_text(su2_mesh.name, mach, aoa), encoding="utf-8")
    config_writing_seconds = time.perf_counter() - config_start

    print(f"\n=== Running {GEOMETRY_NAME}, Mach {mach_label(mach)}, AoA {aoa_label(aoa)} ===")
    print(f"Run directory: {run_dir}")

    su2_start = time.perf_counter()
    result = run_su2(run_dir, config_path)
    su2_runtime_seconds = time.perf_counter() - su2_start
    total_script_runtime_seconds = time.perf_counter() - total_start

    timing = {
        "mesh_conversion_seconds": mesh_conversion_seconds,
        "config_writing_seconds": config_writing_seconds,
        "su2_runtime_seconds": su2_runtime_seconds,
        "total_script_runtime_seconds": total_script_runtime_seconds,
    }

    if result.returncode != 0:
        status = "failed"
        write_timing(run_dir, timing, {"history_found": False}, status, mach, aoa)
        print(f"Case failed with SU2_CFD exit code {result.returncode}. Continuing sweep.")
        return blank_summary_row(mach, aoa, status)

    history_summary = parse_history(run_dir / "history.csv")
    status = parse_convergence_status(run_dir / "log_stdout.txt")
    write_timing(run_dir, timing, history_summary, status, mach, aoa)

    print(f"Status: {status}")
    print(f"SU2 runtime: {su2_runtime_seconds:.3f} s")
    print(f"Total runtime: {total_script_runtime_seconds:.3f} s")
    return make_summary_row(mach, aoa, status, timing, history_summary)


def write_summary(rows: list[dict[str, Any]]) -> None:
    SU2_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    wall_start = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for mach in MACH_NUMBERS:
        try:
            rows.append(run_case(mach))
        except Exception as exc:
            print(f"Case Mach {mach_label(mach)} failed before/during setup: {exc}")
            rows.append(blank_summary_row(mach, AOA_DEGREES, "failed"))

    total_wall_time = time.perf_counter() - wall_start
    write_summary(rows)

    print("\n=== Mach sweep summary ===")
    for row in rows:
        print(
            f"M={row['mach']}: status={row['status']}, "
            f"su2_runtime_s={row['su2_runtime_s']}, total_runtime_s={row['total_runtime_s']}"
        )
    print(f"Total wall time: {total_wall_time:.3f} s")
    print(f"Summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
