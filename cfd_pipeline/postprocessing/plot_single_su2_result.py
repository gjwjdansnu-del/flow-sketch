#!/usr/bin/env python3
"""Create quick-look plots for one SU2 Euler run."""

from __future__ import annotations

import csv
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "datasets" / "raw" / "su2_runs" / "diamond_airfoil_000_M2_A0"
FLOW_VTU = RUN_DIR / "flow.vtu"
SURFACE_VTU = RUN_DIR / "surface_flow.vtu"
HISTORY_CSV = RUN_DIR / "history.csv"
PREVIEW_DIR = PROJECT_ROOT / "datasets" / "previews" / "su2_results"


VTK_DTYPE_TO_NUMPY = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt8": np.uint8,
    "UInt32": np.uint32,
    "UInt64": np.uint64,
}


@dataclass
class VtuData:
    points: np.ndarray
    cell_types: np.ndarray
    cell_connectivity: list[np.ndarray]
    point_data: dict[str, np.ndarray]


def read_vtu(path: Path) -> VtuData:
    """Read VTU with meshio first, then fall back to SU2 raw-appended parsing."""
    try:
        import meshio  # type: ignore[import-not-found]

        mesh = meshio.read(path)
        cell_types: list[int] = []
        cell_connectivity: list[np.ndarray] = []
        vtk_type_by_meshio_type = {"line": 3, "triangle": 5, "quad": 9}
        for cell_block in mesh.cells:
            vtk_type = vtk_type_by_meshio_type.get(cell_block.type)
            if vtk_type is None:
                continue
            for cell in cell_block.data:
                cell_types.append(vtk_type)
                cell_connectivity.append(np.asarray(cell, dtype=np.int64))
        print(f"Read {path.name} with meshio.")
        return VtuData(
            points=np.asarray(mesh.points[:, :2], dtype=float),
            cell_types=np.asarray(cell_types, dtype=np.uint8),
            cell_connectivity=cell_connectivity,
            point_data={name: np.asarray(values) for name, values in mesh.point_data.items()},
        )
    except Exception as exc:
        print(f"meshio could not read {path.name}: {exc}")
        print("Falling back to raw appended VTU parser.")
        return read_raw_appended_vtu(path)


def read_raw_appended_vtu(path: Path) -> VtuData:
    data = path.read_bytes()
    appended_tag = b"<AppendedData"
    appended_index = data.index(appended_tag)
    xml_prefix = data[:appended_index] + b"</VTKFile>\n"
    xml_text = xml_prefix.decode("utf-8", errors="replace")
    root = ET.fromstring(xml_text)
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"No Piece element found in VTU file: {path}")

    underscore_index = data.index(b"_", appended_index)
    appended_data = data[underscore_index + 1 :]

    points_element = piece.find("Points")
    cells_element = piece.find("Cells")
    point_data_element = piece.find("PointData")
    if points_element is None or cells_element is None or point_data_element is None:
        raise ValueError(f"Missing Points, Cells, or PointData section in {path}")

    point_array = points_element.find("DataArray")
    if point_array is None:
        raise ValueError(f"Missing point coordinate DataArray in {path}")
    points = read_data_array(point_array, appended_data)[:, :2]

    cell_arrays = {array.attrib.get("Name", ""): array for array in cells_element.findall("DataArray")}
    connectivity = read_data_array(cell_arrays["connectivity"], appended_data).ravel()
    offsets = read_data_array(cell_arrays["offsets"], appended_data).ravel()
    cell_types = read_data_array(cell_arrays["types"], appended_data).ravel().astype(np.uint8)

    cell_connectivity = split_cells(connectivity, offsets)
    point_data: dict[str, np.ndarray] = {}
    for array in point_data_element.findall("DataArray"):
        name = array.attrib.get("Name", "").strip()
        if not name:
            continue
        point_data[name] = read_data_array(array, appended_data)

    return VtuData(
        points=np.asarray(points, dtype=float),
        cell_types=cell_types,
        cell_connectivity=cell_connectivity,
        point_data=point_data,
    )


def read_data_array(array: ET.Element, appended_data: bytes) -> np.ndarray:
    vtk_type = array.attrib["type"]
    dtype = VTK_DTYPE_TO_NUMPY[vtk_type]
    components = int(array.attrib.get("NumberOfComponents", "1"))
    offset = int(array.attrib["offset"])
    n_bytes = struct.unpack_from("<Q", appended_data, offset)[0]
    start = offset + 8
    raw = appended_data[start : start + n_bytes]
    values = np.frombuffer(raw, dtype=dtype)
    if components > 1:
        values = values.reshape((-1, components))
    return values


def split_cells(connectivity: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    cells: list[np.ndarray] = []
    start = 0
    for offset in offsets:
        end = int(offset)
        cells.append(np.asarray(connectivity[start:end], dtype=np.int64))
        start = end
    return cells


def triangles_from_vtu(data: VtuData) -> np.ndarray:
    triangles = [
        cell
        for cell_type, cell in zip(data.cell_types, data.cell_connectivity, strict=True)
        if int(cell_type) == 5 and len(cell) == 3
    ]
    if not triangles:
        raise ValueError("No triangle cells found for field plotting.")
    return np.asarray(triangles, dtype=np.int64)


def surface_segments_from_vtu(data: VtuData) -> list[np.ndarray]:
    return [
        cell
        for cell_type, cell in zip(data.cell_types, data.cell_connectivity, strict=True)
        if int(cell_type) == 3 and len(cell) == 2
    ]


def find_field(point_data: dict[str, np.ndarray], candidates: list[str]) -> tuple[str, np.ndarray] | None:
    normalized = {normalize_name(name): name for name in point_data}
    for candidate in candidates:
        actual_name = normalized.get(normalize_name(candidate))
        if actual_name is not None:
            return actual_name, np.asarray(point_data[actual_name]).squeeze()
    return None


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def overlay_surface(ax: plt.Axes, surface: VtuData | None) -> None:
    if surface is None:
        return
    segments = surface_segments_from_vtu(surface)
    if segments:
        for segment in segments:
            xy = surface.points[segment]
            ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=0.6)
        return

    # Fallback for surface files with ordered boundary points but no line cells.
    ax.plot(surface.points[:, 0], surface.points[:, 1], color="black", linewidth=0.8)


def plot_field(flow: VtuData, surface: VtuData | None, field_name: str, values: np.ndarray) -> Path:
    triangles = triangles_from_vtu(flow)
    triangulation = mtri.Triangulation(flow.points[:, 0], flow.points[:, 1], triangles)

    output_path = PREVIEW_DIR / f"{normalize_name(field_name)}.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    plot = ax.tripcolor(triangulation, values, shading="gouraud", cmap="turbo")
    overlay_surface(ax, surface)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    ax.set_title(field_name)
    fig.colorbar(plot, ax=ax, label=field_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def clean_header(value: str) -> str:
    return value.strip().strip('"').strip()


def read_history(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = [clean_header(value) for value in next(reader)]
        rows = [[float(value) for value in row] for row in reader if row]
    return header, np.asarray(rows, dtype=float)


def plot_history(path: Path) -> Path | None:
    if not path.exists():
        print(f"History CSV missing: {path}")
        return None

    header, rows = read_history(path)
    if rows.size == 0:
        print(f"History CSV has no data rows: {path}")
        return None

    print(f"History columns: {header}")
    iteration_column = "Inner_Iter" if "Inner_Iter" in header else header[0]
    x = rows[:, header.index(iteration_column)]
    residual_columns = [name for name in header if name.startswith("rms[")]
    if not residual_columns:
        residual_columns = header[1 : min(len(header), 5)]

    output_path = PREVIEW_DIR / "history_residuals.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    for column in residual_columns:
        ax.plot(x, rows[:, header.index(column)], label=column)
    ax.set_xlabel(iteration_column)
    ax.set_ylabel("Residual / logged value")
    ax.set_title("SU2 residual history")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    flow = read_vtu(FLOW_VTU)
    surface = read_vtu(SURFACE_VTU) if SURFACE_VTU.exists() else None

    available_fields = list(flow.point_data.keys())
    print(f"Available flow point fields: {available_fields}")

    requested_fields = {
        "Mach": ["Mach", "Mach_Number", "MachNumber"],
        "Pressure": ["Pressure", "Static_Pressure", "Pressure_Pa"],
        "Density": ["Density", "Rho", "rho"],
    }

    written: list[Path] = []
    for label, candidates in requested_fields.items():
        match = find_field(flow.point_data, candidates)
        if match is None:
            print(f"{label} field not found. Candidates tried: {candidates}")
            continue
        field_name, values = match
        output_path = plot_field(flow, surface, field_name, values)
        written.append(output_path)
        print(f"Wrote {label} plot from field '{field_name}': {output_path}")

    history_plot = plot_history(HISTORY_CSV)
    if history_plot is not None:
        written.append(history_plot)
        print(f"Wrote history plot: {history_plot}")

    print(f"Generated {len(written)} preview plots in {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
