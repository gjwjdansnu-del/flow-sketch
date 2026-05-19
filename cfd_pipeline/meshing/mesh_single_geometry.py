#!/usr/bin/env python3
"""Generate a first 2D external-flow mesh for one closed body geometry."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY = PROJECT_ROOT / "datasets" / "raw" / "geometries" / "diamond_airfoil_000.csv"
MESH_DIR = PROJECT_ROOT / "datasets" / "raw" / "meshes"
PREVIEW_DIR = PROJECT_ROOT / "datasets" / "previews" / "meshes"

DOMAIN = {
    "x_min": -3.0,
    "x_max": 6.0,
    "y_min": -3.0,
    "y_max": 3.0,
}
NEAR_BODY_SIZE = 0.02
FARFIELD_SIZE = 0.2


def read_geometry_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Geometry CSV does not exist: {path}")

    points: list[tuple[float, float]] = []
    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames != ["x", "y"]:
            raise ValueError(f"Expected CSV header ['x', 'y'], got {reader.fieldnames}")
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))

    geometry = np.asarray(points, dtype=float)
    validate_geometry(geometry)
    return geometry


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def validate_geometry(points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Geometry must be an Nx2 coordinate array.")
    if len(points) < 50:
        raise ValueError(f"Geometry needs at least 50 points, got {len(points)}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Geometry contains NaN or infinite values.")
    if not np.allclose(points[0], points[-1]):
        raise ValueError("Geometry must be a closed loop: first point must equal last point.")
    if polygon_area(points) <= 0.0:
        raise ValueError("Geometry must have positive polygon area.")


def build_with_gmsh_api(points: np.ndarray, mesh_path: Path) -> None:
    try:
        import gmsh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("gmsh Python API is not available.") from exc

    gmsh.initialize()
    try:
        gmsh.model.add("single_external_flow_mesh")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        x_min = DOMAIN["x_min"]
        x_max = DOMAIN["x_max"]
        y_min = DOMAIN["y_min"]
        y_max = DOMAIN["y_max"]

        farfield_points = [
            gmsh.model.geo.addPoint(x_min, y_min, 0.0, FARFIELD_SIZE),
            gmsh.model.geo.addPoint(x_max, y_min, 0.0, FARFIELD_SIZE),
            gmsh.model.geo.addPoint(x_max, y_max, 0.0, FARFIELD_SIZE),
            gmsh.model.geo.addPoint(x_min, y_max, 0.0, FARFIELD_SIZE),
        ]
        farfield_lines = [
            gmsh.model.geo.addLine(farfield_points[0], farfield_points[1]),
            gmsh.model.geo.addLine(farfield_points[1], farfield_points[2]),
            gmsh.model.geo.addLine(farfield_points[2], farfield_points[3]),
            gmsh.model.geo.addLine(farfield_points[3], farfield_points[0]),
        ]

        open_body = points[:-1]
        body_points = [
            gmsh.model.geo.addPoint(float(x), float(y), 0.0, NEAR_BODY_SIZE) for x, y in open_body
        ]
        body_lines = [
            gmsh.model.geo.addLine(body_points[i], body_points[(i + 1) % len(body_points)])
            for i in range(len(body_points))
        ]

        farfield_loop = gmsh.model.geo.addCurveLoop(farfield_lines)
        body_loop = gmsh.model.geo.addCurveLoop(body_lines)
        fluid_surface = gmsh.model.geo.addPlaneSurface([farfield_loop, body_loop])

        gmsh.model.geo.synchronize()

        farfield_group = gmsh.model.addPhysicalGroup(1, farfield_lines)
        gmsh.model.setPhysicalName(1, farfield_group, "farfield")
        wall_group = gmsh.model.addPhysicalGroup(1, body_lines)
        gmsh.model.setPhysicalName(1, wall_group, "wall")
        fluid_group = gmsh.model.addPhysicalGroup(2, [fluid_surface])
        gmsh.model.setPhysicalName(2, fluid_group, "fluid")

        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_path))
    finally:
        gmsh.finalize()


def geo_file_contents(points: np.ndarray, mesh_path: Path) -> str:
    lines: list[str] = [
        'SetFactory("Built-in");',
        "Mesh.MshFileVersion = 2.2;",
        f"Point(1) = {{{DOMAIN['x_min']}, {DOMAIN['y_min']}, 0, {FARFIELD_SIZE}}};",
        f"Point(2) = {{{DOMAIN['x_max']}, {DOMAIN['y_min']}, 0, {FARFIELD_SIZE}}};",
        f"Point(3) = {{{DOMAIN['x_max']}, {DOMAIN['y_max']}, 0, {FARFIELD_SIZE}}};",
        f"Point(4) = {{{DOMAIN['x_min']}, {DOMAIN['y_max']}, 0, {FARFIELD_SIZE}}};",
        "Line(1) = {1, 2};",
        "Line(2) = {2, 3};",
        "Line(3) = {3, 4};",
        "Line(4) = {4, 1};",
    ]

    first_body_point_id = 1001
    first_body_line_id = 1001
    open_body = points[:-1]
    for index, (x, y) in enumerate(open_body):
        point_id = first_body_point_id + index
        lines.append(f"Point({point_id}) = {{{float(x)}, {float(y)}, 0, {NEAR_BODY_SIZE}}};")

    for index in range(len(open_body)):
        line_id = first_body_line_id + index
        start = first_body_point_id + index
        end = first_body_point_id + ((index + 1) % len(open_body))
        lines.append(f"Line({line_id}) = {{{start}, {end}}};")

    farfield_line_ids = "1, 2, 3, 4"
    body_line_ids = ", ".join(str(first_body_line_id + i) for i in range(len(open_body)))
    lines.extend(
        [
            f"Curve Loop(1) = {{{farfield_line_ids}}};",
            f"Curve Loop(2) = {{{body_line_ids}}};",
            "Plane Surface(1) = {1, 2};",
            f'Physical Curve("farfield") = {{{farfield_line_ids}}};',
            f'Physical Curve("wall") = {{{body_line_ids}}};',
            'Physical Surface("fluid") = {1};',
            f'Save "{mesh_path}";',
        ]
    )
    return "\n".join(lines) + "\n"


def build_with_gmsh_cli(points: np.ndarray, mesh_path: Path) -> None:
    gmsh_executable = shutil.which("gmsh")
    if gmsh_executable is None:
        raise RuntimeError("gmsh CLI executable was not found on PATH.")

    with tempfile.TemporaryDirectory() as temporary_directory:
        geo_path = Path(temporary_directory) / "single_external_flow_mesh.geo"
        geo_path.write_text(geo_file_contents(points, mesh_path), encoding="utf-8")
        result = subprocess.run(
            [gmsh_executable, str(geo_path), "-2", "-format", "msh2", "-o", str(mesh_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "gmsh CLI mesh generation failed.\n"
                f"Command: {gmsh_executable} {geo_path} -2 -format msh2 -o {mesh_path}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )


def parse_msh2_triangles(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = mesh_path.read_text(encoding="utf-8", errors="replace").splitlines()
    nodes: dict[int, tuple[float, float]] = {}
    triangles: list[list[int]] = []

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line == "$Nodes":
            count = int(lines[index + 1])
            for offset in range(count):
                parts = lines[index + 2 + offset].split()
                node_id = int(parts[0])
                nodes[node_id] = (float(parts[1]), float(parts[2]))
            index += count + 3
            continue
        if line == "$Elements":
            count = int(lines[index + 1])
            for offset in range(count):
                parts = lines[index + 2 + offset].split()
                element_type = int(parts[1])
                tag_count = int(parts[2])
                node_ids = [int(value) for value in parts[3 + tag_count :]]
                if element_type == 2 and len(node_ids) == 3:
                    triangles.append(node_ids)
            index += count + 3
            continue
        index += 1

    if not nodes:
        raise ValueError(f"No nodes found in mesh file: {mesh_path}")
    if not triangles:
        raise ValueError(f"No triangular 2D elements found in mesh file: {mesh_path}")

    sorted_node_ids = sorted(nodes)
    node_index = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
    coordinates = np.asarray([nodes[node_id] for node_id in sorted_node_ids], dtype=float)
    triangle_indices = np.asarray(
        [[node_index[node_id] for node_id in triangle] for triangle in triangles], dtype=int
    )
    return coordinates, triangle_indices


def save_mesh_preview(mesh_path: Path, preview_path: Path, body_points: np.ndarray) -> None:
    coordinates, triangles = parse_msh2_triangles(mesh_path)

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.triplot(
        coordinates[:, 0],
        coordinates[:, 1],
        triangles,
        color="0.45",
        linewidth=0.25,
        alpha=0.7,
    )
    ax.fill(body_points[:, 0], body_points[:, 1], color="#D62728", alpha=0.65, label="wall")
    ax.plot(body_points[:, 0], body_points[:, 1], color="#8C1D18", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(DOMAIN["x_min"], DOMAIN["x_max"])
    ax.set_ylim(DOMAIN["y_min"], DOMAIN["y_max"])
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    ax.set_title(mesh_path.name)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.tight_layout()
    fig.savefig(preview_path, dpi=180)
    plt.close(fig)


def generate_mesh(geometry_path: Path = DEFAULT_GEOMETRY) -> tuple[Path, Path]:
    points = read_geometry_csv(geometry_path)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    stem = geometry_path.stem
    mesh_path = MESH_DIR / f"{stem}.msh"
    preview_path = PREVIEW_DIR / f"{stem}_mesh.png"

    try:
        print("Trying gmsh Python API...")
        build_with_gmsh_api(points, mesh_path)
        print("Mesh generated with gmsh Python API.")
    except Exception as api_error:
        print(f"gmsh Python API failed: {api_error}")
        print("Falling back to gmsh CLI...")
        build_with_gmsh_cli(points, mesh_path)
        print("Mesh generated with gmsh CLI.")

    if not mesh_path.exists():
        raise RuntimeError(f"Mesh generation reported success, but file is missing: {mesh_path}")

    save_mesh_preview(mesh_path, preview_path, points)
    return mesh_path, preview_path


def main() -> None:
    mesh_path, preview_path = generate_mesh()
    print(f"Mesh output: {mesh_path}")
    print(f"Preview output: {preview_path}")


if __name__ == "__main__":
    main()
