#!/usr/bin/env python3
"""Generate simple closed 2D body polygons for later CFD cases."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from geometry_utils import (  # noqa: E402
    close_loop,
    ensure_ccw,
    normalize_geometry,
    validate_geometry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw" / "geometries"
PREVIEW_DIR = PROJECT_ROOT / "datasets" / "previews" / "geometries"
GALLERY_PATH = PREVIEW_DIR / "geometry_gallery.png"

SAMPLES_PER_DETERMINISTIC_TYPE = 10
SAMPLES_PER_RANDOM_TYPE = 8
POINTS_PER_SHAPE = 160
RANDOM_SEED = 7
GALLERY_COLUMNS = 12

RANDOM_SAMPLE_COUNTS: dict[str, int] = {
    "random_smooth_body": 8,
    "random_angular_body": 8,
    "random_bluff_body": 10,
    "random_asymmetric_body": 8,
    "random_thin_body": 8,
    "random_star_body": 12,
    "random_sharp_star_body": 12,
}


def prepare_shape(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = normalize_geometry(points, target_chord=1.0)
    points = ensure_ccw(points)
    validate_geometry(points, min_points=50)
    return points


def write_csv(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        writer.writerows(points)


def save_preview(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(points[:, 0], points[:, 1], color="black", linewidth=1.8)
    ax.fill(points[:, 0], points[:, 1], color="#4C78A8", alpha=0.35)
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def resample_polyline(vertices: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a closed polygon boundary to a fixed number of points."""
    vertices = close_loop(np.asarray(vertices, dtype=float))
    segments = vertices[1:] - vertices[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 0.0:
        raise ValueError("cannot resample a zero-perimeter polygon")

    targets = np.linspace(0.0, perimeter, n_points, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    sampled: list[np.ndarray] = []

    for target in targets:
        index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
        local_length = target - cumulative[index]
        weight = 0.0 if lengths[index] == 0.0 else local_length / lengths[index]
        sampled.append(vertices[index] + weight * segments[index])

    return close_loop(np.asarray(sampled))


def cosine_x(n: int) -> np.ndarray:
    beta = np.linspace(0.0, math.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


def upper_lower_to_polygon(x: np.ndarray, y_upper: np.ndarray, y_lower: np.ndarray) -> np.ndarray:
    upper = np.column_stack([x, y_upper])
    lower = np.column_stack([x[::-1], y_lower[::-1]])
    return close_loop(np.vstack([upper, lower[1:-1]]))


def generate_wedge(sample_index: int) -> np.ndarray:
    half_angle = math.radians(6.0 + 1.5 * sample_index)
    half_height = 0.5 * math.tan(half_angle)
    vertices = np.array(
        [
            [-0.5, 0.0],
            [0.5, half_height],
            [0.5, -half_height],
        ]
    )
    return prepare_shape(resample_polyline(vertices, POINTS_PER_SHAPE))


def generate_diamond_airfoil(sample_index: int) -> np.ndarray:
    thickness = 0.06 + 0.01 * sample_index
    camber_shift = 0.02 * math.sin(sample_index)
    vertices = np.array(
        [
            [-0.5, 0.0],
            [0.0, 0.5 * thickness + camber_shift],
            [0.5, 0.0],
            [0.0, -0.5 * thickness + camber_shift],
        ]
    )
    return prepare_shape(resample_polyline(vertices, POINTS_PER_SHAPE))


def generate_biconvex_airfoil(sample_index: int) -> np.ndarray:
    thickness = 0.06 + 0.012 * sample_index
    x = cosine_x(POINTS_PER_SHAPE // 2)
    y = 2.0 * thickness * np.sqrt(np.clip(x * (1.0 - x), 0.0, None))
    return prepare_shape(upper_lower_to_polygon(x, y, -y))


def generate_naca_00xx(sample_index: int) -> np.ndarray:
    thickness_ratio = 0.06 + 0.01 * sample_index
    x = cosine_x(POINTS_PER_SHAPE // 2)
    y_t = 5.0 * thickness_ratio * (
        0.2969 * np.sqrt(np.clip(x, 0.0, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    y_t[0] = 0.0
    y_t[-1] = 0.0
    return prepare_shape(upper_lower_to_polygon(x, y_t, -y_t))


def generate_ellipse(sample_index: int) -> np.ndarray:
    height = 0.12 + 0.018 * sample_index
    theta = np.linspace(0.0, 2.0 * math.pi, POINTS_PER_SHAPE, endpoint=False)
    x = 0.5 * np.cos(theta)
    y = 0.5 * height * np.sin(theta)
    return prepare_shape(close_loop(np.column_stack([x, y])))


def generate_blunt_capsule(sample_index: int) -> np.ndarray:
    radius = 0.055 + 0.008 * sample_index
    half_body = 0.5 - radius
    n_arc = POINTS_PER_SHAPE // 4
    n_line = POINTS_PER_SHAPE // 4

    right_arc = np.linspace(-math.pi / 2.0, math.pi / 2.0, n_arc, endpoint=False)
    left_arc = np.linspace(math.pi / 2.0, 3.0 * math.pi / 2.0, n_arc, endpoint=False)
    top_x = np.linspace(half_body, -half_body, n_line, endpoint=False)
    bottom_x = np.linspace(-half_body, half_body, n_line, endpoint=False)

    points = np.vstack(
        [
            np.column_stack([half_body + radius * np.cos(right_arc), radius * np.sin(right_arc)]),
            np.column_stack([top_x, np.full_like(top_x, radius)]),
            np.column_stack([-half_body + radius * np.cos(left_arc), radius * np.sin(left_arc)]),
            np.column_stack([bottom_x, np.full_like(bottom_x, -radius)]),
        ]
    )
    return prepare_shape(close_loop(points))


def generate_random_smooth_body(sample_index: int, rng: np.random.Generator) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, POINTS_PER_SHAPE, endpoint=False)
    phase_1 = rng.uniform(0.0, 2.0 * math.pi)
    phase_2 = rng.uniform(0.0, 2.0 * math.pi)
    phase_3 = rng.uniform(0.0, 2.0 * math.pi)
    radial = (
        1.0
        + (0.08 + 0.01 * sample_index) * np.cos(3.0 * theta + phase_1)
        + 0.05 * np.sin(5.0 * theta + phase_2)
        + 0.03 * np.cos(7.0 * theta + phase_3)
    )
    radial = np.clip(radial, 0.72, 1.28)
    aspect = 0.18 + 0.02 * (sample_index % 5)
    x = 0.5 * radial * np.cos(theta)
    y = aspect * radial * np.sin(theta)
    return prepare_shape(close_loop(np.column_stack([x, y])))


def generate_random_angular_body(sample_index: int, rng: np.random.Generator) -> np.ndarray:
    n_sides = 5 + (sample_index % 6)
    angles = np.linspace(0.0, 2.0 * math.pi, n_sides, endpoint=False)
    angle_jitter = rng.uniform(-0.12, 0.12, size=n_sides)
    radii = 0.42 + 0.14 * rng.random(n_sides) + 0.01 * sample_index
    aspect = 0.16 + 0.025 * (sample_index % 4)
    x = radii * np.cos(angles + angle_jitter)
    y = aspect * radii * np.sin(angles + angle_jitter)
    vertices = np.column_stack([x, y])
    return prepare_shape(resample_polyline(vertices, POINTS_PER_SHAPE))


def generate_random_bluff_body(sample_index: int, rng: np.random.Generator) -> np.ndarray:
    """Single outer contour: rounded nose on the left, straight/tapered afterbody on the right."""
    radius = 0.085 + 0.015 * (sample_index % 5)
    half_body = 0.5 - radius
    n_arc = 36
    n_line = 22

    top_height = radius * (1.0 + rng.uniform(-0.06, 0.14))
    bottom_height = -radius * (1.0 + rng.uniform(-0.06, 0.14))
    tail_x = 0.5 - 0.04 * (sample_index % 2)

    left_arc = np.linspace(math.pi / 2.0, 3.0 * math.pi / 2.0, n_arc, endpoint=False)
    nose = np.column_stack(
        [
            -half_body + radius * np.cos(left_arc),
            radius * np.sin(left_arc),
        ]
    )

    bottom = np.column_stack(
        [
            np.linspace(-half_body, tail_x, n_line, endpoint=False),
            np.full(n_line, bottom_height),
        ]
    )

    if sample_index % 3 == 2:
        n_tail = n_line // 2
        right_lower = np.column_stack(
            [
                np.full(n_tail, tail_x),
                np.linspace(bottom_height, 0.0, n_tail, endpoint=False),
            ]
        )
        right_upper = np.column_stack(
            [
                np.full(n_tail, tail_x),
                np.linspace(0.0, top_height, n_tail, endpoint=False),
            ]
        )
        top = np.column_stack(
            [
                np.linspace(tail_x, -half_body, n_line, endpoint=False),
                np.full(n_line, top_height),
            ]
        )
        points = np.vstack([nose, bottom[1:], right_lower[1:], right_upper[1:], top[1:]])
    else:
        right = np.column_stack(
            [
                np.full(n_line, tail_x),
                np.linspace(bottom_height, top_height, n_line, endpoint=False),
            ]
        )
        top = np.column_stack(
            [
                np.linspace(tail_x, -half_body, n_line, endpoint=False),
                np.full(n_line, top_height),
            ]
        )
        points = np.vstack([nose, bottom[1:], right[1:], top[1:]])

    return prepare_shape(resample_polyline(points, POINTS_PER_SHAPE))


def _radial_star_polygon(
    sample_index: int,
    rng: np.random.Generator,
    n_lobes_range: tuple[int, int],
    n_points: int,
    sharpness: float,
    asymmetric: bool,
) -> np.ndarray:
    n_lobes = int(rng.integers(n_lobes_range[0], n_lobes_range[1] + 1))
    theta = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    amp = 0.12 + 0.03 * (sample_index % 6) + 0.05 * rng.random()
    base = 0.86 + 0.08 * rng.random()

    if sharpness < 1.0:
        lobes = np.abs(np.cos(0.5 * n_lobes * theta + phase)) ** sharpness
    else:
        lobes = 0.5 + 0.5 * np.cos(n_lobes * theta + phase)
        lobes = np.clip(lobes, 0.0, 1.0) ** sharpness

    radial = np.clip(base + amp * lobes, 0.58, 1.32)
    aspect = 0.14 + 0.08 * rng.random() if asymmetric else 0.18 + 0.03 * (sample_index % 4)
    theta_offset = rng.uniform(-0.18, 0.18) if asymmetric else 0.0
    x_stretch = 0.9 + 0.14 * rng.random() if asymmetric else 1.0

    x = 0.5 * x_stretch * radial * np.cos(theta + theta_offset)
    y = aspect * radial * np.sin(theta + theta_offset)
    return np.column_stack([x, y])


def generate_random_star_body(sample_index: int, rng: np.random.Generator) -> np.ndarray:
    n_points = int(np.clip(120 + sample_index * 6, 120, 200))
    asymmetric = (sample_index % 2 == 1) or rng.random() < 0.35
    points = _radial_star_polygon(
        sample_index=sample_index,
        rng=rng,
        n_lobes_range=(5, 9),
        n_points=n_points,
        sharpness=2.2,
        asymmetric=asymmetric,
    )
    return prepare_shape(close_loop(points))


def generate_random_sharp_star_body(sample_index: int, rng: np.random.Generator) -> np.ndarray:
    n_points = int(np.clip(80 + sample_index * 6, 80, 160))
    asymmetric = (sample_index % 3 != 0) or rng.random() < 0.4
    sharpness = 0.18 + 0.04 * (sample_index % 4)
    points = _radial_star_polygon(
        sample_index=sample_index,
        rng=rng,
        n_lobes_range=(5, 10),
        n_points=n_points,
        sharpness=sharpness,
        asymmetric=asymmetric,
    )
    return prepare_shape(close_loop(points))


def generate_random_asymmetric_body(sample_index: int, rng: np.random.Generator) -> np.ndarray:
    x = cosine_x(POINTS_PER_SHAPE // 2)
    thickness = 0.05 + 0.012 * (sample_index % 6)
    camber_amp = 0.03 + 0.01 * (sample_index % 5)
    camber_phase = rng.uniform(0.0, 2.0 * math.pi)
    camber = camber_amp * np.sin(math.pi * x + camber_phase) * x * (1.0 - x)
    thickness_profile = thickness * np.sqrt(np.clip(x * (1.0 - x), 0.0, None))
    upper = thickness_profile + camber + 0.5 * camber_amp * x
    lower = -thickness_profile + 0.35 * camber
    return prepare_shape(upper_lower_to_polygon(x, upper, lower))


def generate_random_thin_body(sample_index: int, rng: np.random.Generator) -> np.ndarray:
    mode = sample_index % 3
    if mode == 0:
        half_height = 0.012 + 0.004 * (sample_index % 5)
        vertices = np.array(
            [
                [-0.5, -half_height],
                [0.5, -half_height],
                [0.5, half_height],
                [-0.5, half_height],
            ]
        )
        return prepare_shape(resample_polyline(vertices, POINTS_PER_SHAPE))

    if mode == 1:
        half_height = 0.01 + 0.003 * (sample_index % 4)
        notch = 0.08 + 0.03 * rng.random()
        vertices = np.array(
            [
                [-0.5, -half_height],
                [0.5 - notch, -half_height],
                [0.5, 0.0],
                [0.5 - notch, half_height],
                [-0.5, half_height],
            ]
        )
        return prepare_shape(resample_polyline(vertices, POINTS_PER_SHAPE))

    half_height = 0.008 + 0.003 * (sample_index % 4)
    theta = np.linspace(0.0, 2.0 * math.pi, POINTS_PER_SHAPE, endpoint=False)
    x = 0.5 * np.cos(theta)
    y = half_height * np.sin(theta)
    return prepare_shape(close_loop(np.column_stack([x, y])))


def shape_generators() -> dict[str, object]:
    return {
        "wedge": generate_wedge,
        "diamond_airfoil": generate_diamond_airfoil,
        "biconvex_airfoil": generate_biconvex_airfoil,
        "naca_00xx": generate_naca_00xx,
        "ellipse": generate_ellipse,
        "blunt_capsule": generate_blunt_capsule,
    }


def random_generators() -> dict[str, object]:
    return {
        "random_smooth_body": generate_random_smooth_body,
        "random_angular_body": generate_random_angular_body,
        "random_bluff_body": generate_random_bluff_body,
        "random_asymmetric_body": generate_random_asymmetric_body,
        "random_thin_body": generate_random_thin_body,
        "random_star_body": generate_random_star_body,
        "random_sharp_star_body": generate_random_sharp_star_body,
    }


def random_sample_count(shape_type: str) -> int:
    return RANDOM_SAMPLE_COUNTS.get(shape_type, SAMPLES_PER_RANDOM_TYPE)


def planned_shape_names() -> list[str]:
    names: list[str] = []
    for shape_type in shape_generators():
        for sample_index in range(SAMPLES_PER_DETERMINISTIC_TYPE):
            names.append(f"{shape_type}_{sample_index:03d}")
    for shape_type in random_generators():
        for sample_index in range(random_sample_count(shape_type)):
            names.append(f"{shape_type}_{sample_index:03d}")
    return names


def collect_gallery_previews() -> list[Path]:
    return sorted(
        path
        for path in PREVIEW_DIR.glob("*.png")
        if path.name != GALLERY_PATH.name
    )


def count_existing_geometries() -> int:
    return len(list(RAW_DIR.glob("*.csv")))


def should_skip(shape_name: str, overwrite: bool) -> bool:
    if overwrite:
        return False
    csv_path = RAW_DIR / f"{shape_name}.csv"
    png_path = PREVIEW_DIR / f"{shape_name}.png"
    return csv_path.exists() and png_path.exists()


def save_shape(shape_name: str, points: np.ndarray) -> tuple[Path, Path]:
    csv_path = RAW_DIR / f"{shape_name}.csv"
    png_path = PREVIEW_DIR / f"{shape_name}.png"
    write_csv(csv_path, points)
    save_preview(png_path, points)
    return csv_path, png_path


def save_geometry_gallery(preview_paths: list[Path], output_path: Path, columns: int = GALLERY_COLUMNS) -> None:
    if not preview_paths:
        raise ValueError("No preview images available for gallery generation.")

    preview_paths = sorted(preview_paths, key=lambda path: path.name)
    columns = max(1, columns)
    rows = math.ceil(len(preview_paths) / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 1.6))
    axes_array = np.atleast_1d(axes).ravel()

    for axis in axes_array:
        axis.axis("off")

    for index, preview_path in enumerate(preview_paths):
        axis = axes_array[index]
        image = plt.imread(preview_path)
        axis.imshow(image)
        axis.set_title(preview_path.stem, fontsize=7)
        axis.axis("off")

    fig.suptitle("flow_sketch geometry gallery", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate normalized 2D geometry CSV/PNG files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing geometry CSV/PNG files with the same name.",
    )
    parser.add_argument(
        "--only-types",
        nargs="*",
        default=None,
        help="Generate only the listed shape type prefixes (e.g. random_bluff_body).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    only_types = set(args.only_types) if args.only_types else None
    existing_before = count_existing_geometries()
    added = 0
    skipped = 0
    generated_by_type: dict[str, int] = {}

    deterministic = shape_generators()
    random_types = random_generators()

    if only_types is not None:
        deterministic = {key: value for key, value in deterministic.items() if key in only_types}
        random_types = {key: value for key, value in random_types.items() if key in only_types}

    for shape_type, generator in deterministic.items():
        generated_by_type[shape_type] = 0
        for sample_index in range(SAMPLES_PER_DETERMINISTIC_TYPE):
            shape_name = f"{shape_type}_{sample_index:03d}"
            if should_skip(shape_name, args.overwrite):
                skipped += 1
                continue
            points = generator(sample_index)
            save_shape(shape_name, points)
            generated_by_type[shape_type] += 1
            added += 1

    for shape_type, generator in random_types.items():
        generated_by_type[shape_type] = 0
        sample_count = random_sample_count(shape_type)
        for sample_index in range(sample_count):
            shape_name = f"{shape_type}_{sample_index:03d}"
            if should_skip(shape_name, args.overwrite):
                skipped += 1
                continue
            points = generator(sample_index, rng)
            save_shape(shape_name, points)
            generated_by_type[shape_type] += 1
            added += 1

    gallery_previews = collect_gallery_previews()
    save_geometry_gallery(gallery_previews, GALLERY_PATH)
    final_total = count_existing_geometries()

    target_total = len(planned_shape_names())
    print("=== Geometry generation summary ===")
    print(f"Existing geometry count before generation: {existing_before}")
    print(f"New geometries added: {added}")
    print(f"Skipped existing geometries: {skipped}")
    print(f"Target planned geometries: {target_total}")
    print(f"Final total geometry CSV count: {final_total}")
    print(f"random_star_body generated: {generated_by_type.get('random_star_body', 0)}")
    print(f"random_sharp_star_body generated: {generated_by_type.get('random_sharp_star_body', 0)}")
    print(f"random_bluff_body regenerated: {generated_by_type.get('random_bluff_body', 0)}")
    print(f"CSV output: {RAW_DIR}")
    print(f"Preview output: {PREVIEW_DIR}")
    print(f"Gallery path: {GALLERY_PATH}")


if __name__ == "__main__":
    main()
