#!/usr/bin/env python3
"""Run one steady 2D compressible Euler test case with SU2."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def raise_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


raise_csv_field_limit()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_MESH = PROJECT_ROOT / "datasets" / "raw" / "meshes" / "diamond_airfoil_000.msh"
RUN_DIR = PROJECT_ROOT / "datasets" / "raw" / "su2_runs" / "diamond_airfoil_000_M2_A0"
SU2_MESH_NAME = "diamond_airfoil_000.su2"


def config_text(mesh_filename: str) -> str:
    return f"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% flow_sketch single SU2 Euler test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION ------------%
SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% ----------- COMPRESSIBLE AND INCOMPRESSIBLE FREE-STREAM DEFINITION ----------%
MACH_NUMBER= 2.0
AOA= 0.0
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15
GAMMA_VALUE= 1.4

% ---------------------- REFERENCE VALUE DEFINITION ---------------------------%
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.0
REF_AREA= 1.0

% ----------------------- BOUNDARY CONDITION DEFINITION -----------------------%
MARKER_EULER= ( wall )
MARKER_FAR= ( farfield )

% ------------------------ SURFACES IDENTIFICATION ----------------------------%
MARKER_PLOTTING= ( wall )
MARKER_MONITORING= ( wall )

% ------------- COMMON PARAMETERS TO DEFINE THE NUMERICAL METHOD --------------%
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 0.5
CFL_ADAPT= NO
ITER= 2000

% ------------------------ LINEAR SOLVER DEFINITION ---------------------------%
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= LU_SGS
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 5

% -------------------- FLOW NUMERICAL METHOD DEFINITION -----------------------%
CONV_NUM_METHOD_FLOW= JST
JST_SENSOR_COEFF= ( 0.5, 0.02 )
TIME_DISCRE_FLOW= EULER_IMPLICIT

% --------------------------- CONVERGENCE PARAMETERS --------------------------%
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= -8
CONV_STARTITER= 10
SCREEN_OUTPUT= (INNER_ITER, ITER_TIME, RMS_DENSITY, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, RMS_ENERGY)
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)

% ------------------------- INPUT/OUTPUT INFORMATION --------------------------%
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


def try_su2_msh_converter(input_mesh: Path, output_mesh: Path) -> bool:
    su2_msh = shutil.which("SU2_MSH")
    if su2_msh is None:
        print("SU2_MSH not found; using meshio converter.")
        return False

    attempts = [
        [su2_msh, str(input_mesh), str(output_mesh)],
        [su2_msh, "-i", str(input_mesh), "-o", str(output_mesh)],
    ]
    for command in attempts:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0 and output_mesh.exists():
            print(f"Converted mesh with SU2_MSH: {' '.join(command)}")
            return True

        print(
            "SU2_MSH attempt failed:\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    print("All SU2_MSH attempts failed; using meshio converter.")
    return False


def convert_msh_to_su2_with_meshio(input_mesh: Path, output_mesh: Path) -> None:
    try:
        import meshio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "meshio is required for .msh -> .su2 conversion when SU2_MSH is unavailable. "
            "Install it with: conda install -y meshio"
        ) from exc

    mesh = meshio.read(input_mesh)
    points = mesh.points[:, :2]
    physical_name_by_id = {int(value[0]): name for name, value in mesh.field_data.items()}

    triangles: list[list[int]] = []
    marker_edges: dict[str, list[list[int]]] = {"farfield": [], "wall": []}
    physical_data = mesh.cell_data_dict.get("gmsh:physical", {})

    for cell_block in mesh.cells:
        cells = cell_block.data
        if cell_block.type == "triangle":
            triangles.extend(cells.tolist())
            continue

        if cell_block.type != "line":
            continue

        physical_tags = physical_data.get(cell_block.type)
        if physical_tags is None:
            continue

        if len(physical_tags) != len(cells):
            raise ValueError(
                f"Physical tag count mismatch for line cells: {len(physical_tags)} tags, "
                f"{len(cells)} cells."
            )

        for edge, physical_tag in zip(cells, physical_tags, strict=True):
            marker_name = physical_name_by_id.get(int(physical_tag))
            if marker_name in marker_edges:
                marker_edges[marker_name].append(edge.tolist())

    if not triangles:
        raise ValueError(f"No triangle elements found in gmsh mesh: {input_mesh}")
    for marker_name, edges in marker_edges.items():
        if not edges:
            raise ValueError(f"No line elements found for required marker: {marker_name}")

    with output_mesh.open("w", encoding="utf-8") as file:
        file.write("NDIME= 2\n")
        file.write(f"NELEM= {len(triangles)}\n")
        for element_id, triangle in enumerate(triangles):
            file.write(f"5\t{triangle[0]}\t{triangle[1]}\t{triangle[2]}\t{element_id}\n")

        file.write(f"NPOIN= {len(points)}\n")
        for point_id, (x, y) in enumerate(points):
            file.write(f"{x:.16e}\t{y:.16e}\t{point_id}\n")

        file.write(f"NMARK= {len(marker_edges)}\n")
        for marker_name in ("farfield", "wall"):
            edges = marker_edges[marker_name]
            file.write(f"MARKER_TAG= {marker_name}\n")
            file.write(f"MARKER_ELEMS= {len(edges)}\n")
            for edge in edges:
                file.write(f"3\t{edge[0]}\t{edge[1]}\n")

    print(f"Converted mesh with meshio: {output_mesh}")


def convert_mesh_to_su2(input_mesh: Path, output_mesh: Path) -> None:
    if output_mesh.exists():
        output_mesh.unlink()

    if try_su2_msh_converter(input_mesh, output_mesh):
        return

    convert_msh_to_su2_with_meshio(input_mesh, output_mesh)


def write_config(config_path: Path, su2_mesh: Path) -> None:
    config_path.write_text(config_text(su2_mesh.name), encoding="utf-8")


def prepare_run_directory() -> tuple[Path, Path, Path]:
    if not INPUT_MESH.exists():
        raise FileNotFoundError(
            f"Input mesh is missing: {INPUT_MESH}\n"
            "Run cfd_pipeline/meshing/mesh_single_geometry.py first."
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    mesh_copy = RUN_DIR / INPUT_MESH.name
    su2_mesh = RUN_DIR / SU2_MESH_NAME
    shutil.copy2(INPUT_MESH, mesh_copy)

    config_path = RUN_DIR / "config.cfg"
    return config_path, mesh_copy, su2_mesh


def run_su2(config_path: Path) -> subprocess.CompletedProcess[str]:
    su2_executable = shutil.which("SU2_CFD")
    if su2_executable is None:
        raise RuntimeError("SU2_CFD was not found on PATH.")

    result = subprocess.run(
        [su2_executable, config_path.name],
        cwd=RUN_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    (RUN_DIR / "log_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (RUN_DIR / "log_stderr.txt").write_text(result.stderr, encoding="utf-8")
    return result


def clean_header(value: str) -> str:
    return value.strip().strip('"').strip()


def parse_float_cell(value: str) -> float:
    cleaned = value.replace("\x00", "").strip().strip('"').strip()
    if not cleaned:
        raise ValueError("empty cell")
    return float(cleaned)


def parse_history(history_path: Path) -> dict[str, object]:
    if not history_path.exists():
        return {"history_found": False}

    with history_path.open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = [clean_header(value) for value in next(reader)]
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            try:
                rows.append([parse_float_cell(value) for value in row])
            except ValueError:
                continue

    if not rows:
        return {"history_found": True, "rows": 0}

    final_row = rows[-1]
    final_by_column = dict(zip(header, final_row, strict=True))
    residuals = {name: final_by_column[name] for name in header if name.startswith("rms[")}
    coefficients = {
        name: final_by_column[name] for name in ("CL", "CD") if name in final_by_column
    }

    iteration_column = "Inner_Iter" if "Inner_Iter" in final_by_column else header[0]
    return {
        "history_found": True,
        "rows": len(rows),
        "iteration_column": iteration_column,
        "iterations_completed": int(final_by_column[iteration_column]),
        "final_residuals": residuals,
        "final_coefficients": coefficients,
    }


def parse_convergence_status(stdout_path: Path) -> str:
    if not stdout_path.exists():
        return "unknown"

    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    if "Exit Success (SU2_CFD)" not in stdout:
        return "failed"
    if "Maximum number of iterations reached" in stdout:
        return "max_iterations_reached"
    if "Convergence criteria met" in stdout or "Converged" in stdout:
        return "converged"
    return "completed"


def write_timing(timing: dict[str, float], history_summary: dict[str, object], status: str) -> Path:
    timing_path = RUN_DIR / "timing.json"
    payload = {
        "mesh_conversion_seconds": timing["mesh_conversion_seconds"],
        "config_writing_seconds": timing["config_writing_seconds"],
        "su2_runtime_seconds": timing["su2_runtime_seconds"],
        "total_script_runtime_seconds": timing["total_script_runtime_seconds"],
        "convergence_status": status,
        "history_summary": history_summary,
    }
    timing_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return timing_path


def print_history_summary(summary: dict[str, object]) -> None:
    if not summary.get("history_found"):
        print("history.csv not found.")
        return

    print(f"History rows: {summary.get('rows')}")
    if "iterations_completed" in summary:
        print(f"Iterations completed: {summary['iterations_completed']}")
    if summary.get("final_residuals"):
        print("Final residual values:")
        for name, value in dict(summary["final_residuals"]).items():
            print(f"  {name}: {value}")
    if summary.get("final_coefficients"):
        print("Final aerodynamic coefficients:")
        for name, value in dict(summary["final_coefficients"]).items():
            print(f"  {name}: {value}")


def main() -> None:
    total_start = time.perf_counter()
    config_path, mesh_copy, su2_mesh = prepare_run_directory()

    conversion_start = time.perf_counter()
    convert_mesh_to_su2(mesh_copy, su2_mesh)
    mesh_conversion_seconds = time.perf_counter() - conversion_start

    config_start = time.perf_counter()
    write_config(config_path, su2_mesh)
    config_writing_seconds = time.perf_counter() - config_start

    print(f"Run directory: {RUN_DIR}")
    print(f"Config: {config_path}")
    print(f"Mesh: {su2_mesh}")

    su2_start = time.perf_counter()
    result = run_su2(config_path)
    su2_runtime_seconds = time.perf_counter() - su2_start
    total_script_runtime_seconds = time.perf_counter() - total_start

    print(f"SU2_CFD exit code: {result.returncode}")
    print(f"stdout log: {RUN_DIR / 'log_stdout.txt'}")
    print(f"stderr log: {RUN_DIR / 'log_stderr.txt'}")
    print(f"Mesh conversion time: {mesh_conversion_seconds:.3f} s")
    print(f"Config writing time: {config_writing_seconds:.3f} s")
    print(f"SU2_CFD runtime: {su2_runtime_seconds:.3f} s")
    print(f"Total script runtime: {total_script_runtime_seconds:.3f} s")

    if result.returncode != 0:
        raise RuntimeError(
            "SU2_CFD failed. See log_stdout.txt and log_stderr.txt in the run directory."
        )

    history_summary = parse_history(RUN_DIR / "history.csv")
    convergence_status = parse_convergence_status(RUN_DIR / "log_stdout.txt")
    timing_path = write_timing(
        {
            "mesh_conversion_seconds": mesh_conversion_seconds,
            "config_writing_seconds": config_writing_seconds,
            "su2_runtime_seconds": su2_runtime_seconds,
            "total_script_runtime_seconds": total_script_runtime_seconds,
        },
        history_summary,
        convergence_status,
    )

    print_history_summary(history_summary)
    print(f"Convergence status: {convergence_status}")
    print(f"Timing JSON: {timing_path}")
    print("SU2_CFD completed successfully.")


if __name__ == "__main__":
    main()
