#!/usr/bin/env python3
"""
Overnight SU2 dataset runner for flow_sketch.

Target design (not started automatically by this module):
  - about 100 normalized geometries
  - Mach = [1.5, 2.0, 3.0, 5.0]
  - AoA = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
  - 100 x 4 x 11 = 4400 cases (~10-12 h on Apple Silicon)

Run manually when ready:
  python cfd_pipeline/su2_cases/run_overnight_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def raise_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


raise_csv_field_limit()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "cfd_pipeline" / "meshing"))
sys.path.append(str(PROJECT_ROOT / "cfd_pipeline" / "su2_cases"))

from mesh_single_geometry import generate_mesh  # noqa: E402
from run_single_su2 import convert_mesh_to_su2, parse_convergence_status, parse_history  # noqa: E402


MACH_NUMBERS = [1.5, 2.0, 3.0, 5.0]
AOA_VALUES = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
MAX_GEOMETRIES = 100

GEOMETRY_DIR = PROJECT_ROOT / "datasets" / "raw" / "geometries"
MESH_DIR = PROJECT_ROOT / "datasets" / "raw" / "meshes"
SU2_RUNS_DIR = PROJECT_ROOT / "datasets" / "raw" / "su2_runs"
SUMMARY_CSV = SU2_RUNS_DIR / "overnight_dataset_summary.csv"
SMOKE_SUMMARY_CSV = SU2_RUNS_DIR / "smoke_test_summary.csv"

SMOKE_MACH = 5.0
SMOKE_AOA_DEFAULT = 0.0
COMPLEX_GEOMETRY_TYPES = ("random_sharp_star_body", "random_star_body")

SUMMARY_COLUMNS = [
    "geometry",
    "geometry_type",
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


def discover_geometry_names(limit: int | None = MAX_GEOMETRIES) -> list[str]:
    csv_paths = sorted(GEOMETRY_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No geometry CSV files found in {GEOMETRY_DIR}")
    names = [path.stem for path in csv_paths]
    if limit is None:
        return names
    return names[:limit]


def planned_cases(geometry_names: list[str]) -> list[tuple[str, float, float]]:
    cases: list[tuple[str, float, float]] = []
    for geometry_name in geometry_names:
        for mach in MACH_NUMBERS:
            for aoa in AOA_VALUES:
                cases.append((geometry_name, mach, aoa))
    return cases


def group_geometries_by_type(geometry_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for geometry_name in sorted(geometry_names):
        grouped.setdefault(geometry_type(geometry_name), []).append(geometry_name)
    for names in grouped.values():
        names.sort()
    return grouped


def pick_complex_geometry_type(grouped: dict[str, list[str]]) -> str | None:
    for candidate in COMPLEX_GEOMETRY_TYPES:
        if candidate in grouped:
            return candidate
    return None


def smoke_test_cases(geometry_names: list[str]) -> list[tuple[str, float, float]]:
    """
    One representative geometry per family at Mach=5, AoA=0.
    The most complex family also gets a full AoA sweep at Mach=5.
    """
    grouped = group_geometries_by_type(geometry_names)
    complex_type = pick_complex_geometry_type(grouped)
    cases: list[tuple[str, float, float]] = []

    for family in sorted(grouped):
        geometry_name = grouped[family][0]
        if family == complex_type:
            for aoa in AOA_VALUES:
                cases.append((geometry_name, SMOKE_MACH, aoa))
        else:
            cases.append((geometry_name, SMOKE_MACH, SMOKE_AOA_DEFAULT))

    return cases


def geometry_type(geometry_name: str) -> str:
    match = re.match(r"^(.+)_(\d{3})$", geometry_name)
    if match:
        return match.group(1)
    return geometry_name


def mach_label(mach: float) -> str:
    return f"{mach:.1f}"


def aoa_label(aoa: float) -> str:
    if float(aoa).is_integer():
        return str(int(aoa))
    return f"{aoa:.1f}"


def run_dir_for_case(geometry_name: str, mach: float, aoa: float) -> Path:
    return SU2_RUNS_DIR / f"{geometry_name}_M{mach_label(mach)}_A{aoa_label(aoa)}"


def case_is_complete(run_dir: Path) -> bool:
    return (run_dir / "flow.vtu").exists() and (run_dir / "history.csv").exists()


def load_completed_summary_row(geometry_name: str, mach: float, aoa: float) -> dict[str, Any]:
    run_dir = run_dir_for_case(geometry_name, mach, aoa)
    history_summary: dict[str, Any] = {"history_found": False}
    timing_path = run_dir / "timing.json"
    if timing_path.exists():
        timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
        timing = {
            "mesh_generation_seconds": float(timing_payload.get("mesh_generation_seconds", 0.0)),
            "mesh_conversion_seconds": float(timing_payload.get("mesh_conversion_seconds", 0.0)),
            "config_writing_seconds": float(timing_payload.get("config_writing_seconds", 0.0)),
            "su2_runtime_seconds": float(timing_payload.get("su2_runtime_seconds", 0.0)),
            "total_script_runtime_seconds": float(
                timing_payload.get("total_script_runtime_seconds", 0.0)
            ),
        }
        status = str(timing_payload.get("convergence_status", "existing"))
        embedded_history = timing_payload.get("history_summary")
        if isinstance(embedded_history, dict) and embedded_history:
            history_summary = embedded_history
    else:
        status = "existing"
        if (run_dir / "log_stdout.txt").exists():
            status = parse_convergence_status(run_dir / "log_stdout.txt")
        timing = {
            "mesh_generation_seconds": 0.0,
            "mesh_conversion_seconds": 0.0,
            "config_writing_seconds": 0.0,
            "su2_runtime_seconds": 0.0,
            "total_script_runtime_seconds": 0.0,
        }

    if not history_summary.get("history_found"):
        history_summary = parse_history(run_dir / "history.csv")

    return make_summary_row(geometry_name, mach, aoa, status, timing, history_summary)


def config_text(mesh_filename: str, mach: float, aoa: float) -> str:
    return f"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% flow_sketch overnight dataset SU2 Euler case
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


def ensure_mesh(geometry_name: str) -> tuple[Path, float]:
    mesh_path = MESH_DIR / f"{geometry_name}.msh"
    if mesh_path.exists():
        return mesh_path, 0.0

    geometry_path = GEOMETRY_DIR / f"{geometry_name}.csv"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry CSV is missing: {geometry_path}")

    start = time.perf_counter()
    generated_mesh, _preview = generate_mesh(geometry_path)
    return generated_mesh, time.perf_counter() - start


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


def blank_summary_row(
    geometry_name: str,
    mach: float,
    aoa: float,
    status: str,
    timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "geometry": geometry_name,
        "geometry_type": geometry_type(geometry_name),
        "mach": mach,
        "aoa": aoa,
        "status": status,
        "su2_runtime_s": "" if timing is None else timing.get("su2_runtime_seconds", ""),
        "total_runtime_s": "" if timing is None else timing.get("total_script_runtime_seconds", ""),
        "iterations_completed": "",
        "final_CL": "",
        "final_CD": "",
        "final_rms_Rho": "",
        "final_rms_RhoU": "",
        "final_rms_RhoV": "",
        "final_rms_RhoE": "",
    }


def make_summary_row(
    geometry_name: str,
    mach: float,
    aoa: float,
    status: str,
    timing: dict[str, float],
    history_summary: dict[str, Any],
) -> dict[str, Any]:
    coefficients = history_summary.get("final_coefficients", {}) or {}
    residuals = history_summary.get("final_residuals", {}) or {}
    return {
        "geometry": geometry_name,
        "geometry_type": geometry_type(geometry_name),
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
    geometry_name: str,
    mach: float,
    aoa: float,
    timing: dict[str, float],
    history_summary: dict[str, Any],
    status: str,
) -> None:
    payload = {
        "geometry": geometry_name,
        "geometry_type": geometry_type(geometry_name),
        "mach": mach,
        "aoa": aoa,
        "mesh_generation_seconds": timing["mesh_generation_seconds"],
        "mesh_conversion_seconds": timing["mesh_conversion_seconds"],
        "config_writing_seconds": timing["config_writing_seconds"],
        "su2_runtime_seconds": timing["su2_runtime_seconds"],
        "total_script_runtime_seconds": timing["total_script_runtime_seconds"],
        "convergence_status": status,
        "history_summary": history_summary,
    }
    (run_dir / "timing.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_case(geometry_name: str, mach: float, aoa: float) -> dict[str, Any]:
    total_start = time.perf_counter()
    run_dir = run_dir_for_case(geometry_name, mach, aoa)
    run_dir.mkdir(parents=True, exist_ok=True)

    mesh_path, mesh_generation_seconds = ensure_mesh(geometry_name)
    mesh_copy = run_dir / mesh_path.name
    su2_mesh = run_dir / f"{geometry_name}.su2"
    config_path = run_dir / "config.cfg"
    shutil.copy2(mesh_path, mesh_copy)

    conversion_start = time.perf_counter()
    convert_mesh_to_su2(mesh_copy, su2_mesh)
    mesh_conversion_seconds = time.perf_counter() - conversion_start

    config_start = time.perf_counter()
    config_path.write_text(config_text(su2_mesh.name, mach, aoa), encoding="utf-8")
    config_writing_seconds = time.perf_counter() - config_start

    print(
        f"\n=== Running {geometry_name}, Mach {mach_label(mach)}, AoA {aoa_label(aoa)} ===",
        flush=True,
    )
    print(f"Run directory: {run_dir}", flush=True)

    su2_start = time.perf_counter()
    result = run_su2(run_dir, config_path)
    su2_runtime_seconds = time.perf_counter() - su2_start
    total_script_runtime_seconds = time.perf_counter() - total_start

    timing = {
        "mesh_generation_seconds": mesh_generation_seconds,
        "mesh_conversion_seconds": mesh_conversion_seconds,
        "config_writing_seconds": config_writing_seconds,
        "su2_runtime_seconds": su2_runtime_seconds,
        "total_script_runtime_seconds": total_script_runtime_seconds,
    }

    if result.returncode != 0:
        status = "failed"
        write_timing(run_dir, geometry_name, mach, aoa, timing, {"history_found": False}, status)
        print(
            f"Case failed with SU2_CFD exit code {result.returncode}. Continuing.",
            flush=True,
        )
        return blank_summary_row(geometry_name, mach, aoa, status, timing)

    history_summary = parse_history(run_dir / "history.csv")
    status = parse_convergence_status(run_dir / "log_stdout.txt")
    write_timing(run_dir, geometry_name, mach, aoa, timing, history_summary, status)

    print(f"Status: {status}", flush=True)
    print(f"SU2 runtime: {su2_runtime_seconds:.3f} s", flush=True)
    print(f"Total runtime: {total_script_runtime_seconds:.3f} s", flush=True)
    return make_summary_row(geometry_name, mach, aoa, status, timing, history_summary)


def write_summary(rows: list[dict[str, Any]], summary_path: Path = SUMMARY_CSV) -> None:
    SU2_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_smoke_plan(cases: list[tuple[str, float, float]], complex_type: str | None) -> None:
    families = sorted({geometry_type(geometry) for geometry, _, _ in cases})
    print("=== Smoke test plan ===")
    print(f"Mach fixed at: {SMOKE_MACH}")
    print(f"Default AoA: {SMOKE_AOA_DEFAULT}")
    print(f"Geometry families: {len(families)}")
    print(f"Complex family (full AoA sweep): {complex_type or 'none'}")
    print(f"Total cases: {len(cases)}")
    print(f"Summary CSV: {SMOKE_SUMMARY_CSV}")
    print("\nCases:")
    for index, (geometry_name, mach, aoa) in enumerate(cases, start=1):
        print(
            f"  {index:2d}. {geometry_name}  "
            f"(family={geometry_type(geometry_name)})  M={mach}  AoA={aoa}"
        )


def print_plan(geometry_names: list[str], case_limit: int | None = None) -> None:
    all_cases = planned_cases(geometry_names)
    total_cases = len(all_cases) if case_limit is None else min(case_limit, len(all_cases))
    print("=== Overnight dataset plan ===")
    print(f"Geometries: {len(geometry_names)} (max {MAX_GEOMETRIES})")
    print(f"Mach values: {MACH_NUMBERS}")
    print(f"AoA values: {AOA_VALUES}")
    print(f"Total planned cases: {len(all_cases)}")
    if case_limit is not None:
        print(f"Case limit: {case_limit}")
        print(f"Cases to run: {total_cases}")
    else:
        print(f"Cases to run: {total_cases}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print("This script does not auto-start overnight runs from imports.")


def print_run_summary(
    rows: list[dict[str, Any]],
    total_wall_time: float,
    summary_path: Path = SUMMARY_CSV,
    skipped_count: int = 0,
) -> None:
    success_count = sum(1 for row in rows if row["status"] != "failed")
    failure_count = len(rows) - success_count
    failed_geometries = sorted(
        {str(row["geometry"]) for row in rows if row["status"] == "failed"}
    )

    runtimes: list[float] = []
    for row in rows:
        value = row.get("total_runtime_s")
        if value == "" or value is None:
            continue
        runtimes.append(float(value))

    average_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0

    successful_runtimes = [
        float(row["total_runtime_s"])
        for row in rows
        if row["status"] != "failed" and row.get("total_runtime_s") not in ("", None)
    ]
    average_success_runtime = (
        sum(successful_runtimes) / len(successful_runtimes) if successful_runtimes else 0.0
    )

    print("\n=== Overnight dataset summary ===", flush=True)
    print(f"Total cases: {len(rows)}", flush=True)
    print(f"Success count: {success_count}", flush=True)
    print(f"Failure count: {failure_count}", flush=True)
    print(f"Skipped count: {skipped_count}", flush=True)
    if failed_geometries:
        print(f"Failed geometries: {', '.join(failed_geometries)}")
    else:
        print("Failed geometries: none")
    print(f"Average runtime (all rows with timing): {average_runtime:.3f} s", flush=True)
    print(f"Average runtime (successful cases): {average_success_runtime:.3f} s", flush=True)
    print(f"Total wall time: {total_wall_time:.3f} s", flush=True)
    print(f"Summary CSV: {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the overnight SU2 dataset batch.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the planned geometry/Mach/AoA counts without running SU2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N full-plan cases (geometry x Mach x AoA order).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one geometry per family at Mach=5 and AoA=0, plus a full AoA sweep "
            "for the most complex family."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun cases even when flow.vtu and history.csv already exist.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    if args.smoke_test and args.limit is not None:
        raise ValueError("Use either --smoke-test or --limit, not both.")

    if args.smoke_test:
        geometry_names = discover_geometry_names(limit=None)
        grouped = group_geometries_by_type(geometry_names)
        complex_type = pick_complex_geometry_type(grouped)
        cases = smoke_test_cases(geometry_names)
        print_smoke_plan(cases, complex_type)
        if args.plan_only:
            return
        summary_path = SMOKE_SUMMARY_CSV
    else:
        geometry_names = discover_geometry_names()
        print_plan(geometry_names, case_limit=args.limit)
        if args.plan_only:
            return
        cases = planned_cases(geometry_names)
        if args.limit is not None:
            cases = cases[: args.limit]
        summary_path = SUMMARY_CSV

    wall_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    skipped_count = 0

    for case_index, (geometry_name, mach, aoa) in enumerate(cases, start=1):
        run_dir = run_dir_for_case(geometry_name, mach, aoa)
        if not args.overwrite and case_is_complete(run_dir):
            print(
                f"\n=== Skipping existing case {case_index}/{len(cases)}: "
                f"{geometry_name}, Mach {mach_label(mach)}, AoA {aoa_label(aoa)} ===",
                flush=True,
            )
            rows.append(load_completed_summary_row(geometry_name, mach, aoa))
            skipped_count += 1
            continue

        try:
            rows.append(run_case(geometry_name, mach, aoa))
        except Exception as exc:
            print(
                "Case failed during setup/run: "
                f"geometry={geometry_name}, Mach={mach_label(mach)}, AoA={aoa_label(aoa)}",
                flush=True,
            )
            print(f"Error: {exc}", flush=True)
            rows.append(blank_summary_row(geometry_name, mach, aoa, "failed"))

    total_wall_time = time.perf_counter() - wall_start
    write_summary(rows, summary_path=summary_path)
    print_run_summary(
        rows,
        total_wall_time,
        summary_path=summary_path,
        skipped_count=skipped_count,
    )


if __name__ == "__main__":
    main()
