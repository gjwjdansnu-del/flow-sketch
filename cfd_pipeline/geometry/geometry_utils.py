"""Shared geometry normalization utilities for flow_sketch."""

from __future__ import annotations

import numpy as np

DOMAIN_X = (-1.0, 3.0)
DOMAIN_Y = (-1.0, 1.0)
DOMAIN_MARGIN = 0.05


def close_loop(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points
    if not np.allclose(points[0], points[-1]):
        return np.vstack([points, points[0]])
    return points


def open_loop(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        return points[:-1].copy()
    return points.copy()


def polygon_area(points: np.ndarray) -> float:
    """Signed polygon area for a closed or open polygon."""
    closed = close_loop(np.asarray(points, dtype=float))
    x = closed[:, 0]
    y = closed[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def polygon_centroid(points: np.ndarray) -> tuple[float, float]:
    """Return the area-weighted centroid of a polygon."""
    open_points = open_loop(points)
    if len(open_points) < 3:
        return float(np.mean(open_points[:, 0])), float(np.mean(open_points[:, 1]))

    x = open_points[:, 0]
    y = open_points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area2 = float(np.sum(cross))
    if abs(signed_area2) < 1e-12:
        return float(np.mean(x)), float(np.mean(y))

    centroid_x = float(np.sum((x + x_next) * cross) / (3.0 * signed_area2))
    centroid_y = float(np.sum((y + y_next) * cross) / (3.0 * signed_area2))
    return centroid_x, centroid_y


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    """Return a closed polygon with positive signed area."""
    closed = close_loop(points)
    if polygon_area(closed) < 0.0:
        closed = close_loop(open_loop(closed)[::-1])
    return closed


def chord_length(points: np.ndarray) -> float:
    open_points = open_loop(points)
    if len(open_points) == 0:
        return 0.0
    return float(np.max(open_points[:, 0]) - np.min(open_points[:, 0]))


def normalize_geometry(points: np.ndarray, target_chord: float = 1.0) -> np.ndarray:
    """
    Center a polygon at the origin and scale its x-extent to target_chord.

    Area/thickness variation is preserved; only center and chord are normalized.
    """
    if target_chord <= 0.0:
        raise ValueError("target_chord must be positive")

    open_points = open_loop(np.asarray(points, dtype=float))
    if len(open_points) < 3:
        raise ValueError("polygon must contain at least 3 unique points")

    centroid_x, centroid_y = polygon_centroid(open_points)
    centered = open_points.copy()
    centered[:, 0] -= centroid_x
    centered[:, 1] -= centroid_y

    current_chord = chord_length(centered)
    if current_chord <= 0.0:
        raise ValueError("polygon has non-positive chord length")

    scale = target_chord / current_chord
    scaled = centered * scale
    normalized = ensure_ccw(scaled)

    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized polygon contains NaN or infinite coordinates")

    validate_domain_fit(normalized)
    return normalized


def validate_domain_fit(
    points: np.ndarray,
    x_range: tuple[float, float] = DOMAIN_X,
    y_range: tuple[float, float] = DOMAIN_Y,
    margin: float = DOMAIN_MARGIN,
) -> None:
    open_points = open_loop(points)
    min_x = float(np.min(open_points[:, 0]))
    max_x = float(np.max(open_points[:, 0]))
    min_y = float(np.min(open_points[:, 1]))
    max_y = float(np.max(open_points[:, 1]))

    if (
        min_x < x_range[0] + margin
        or max_x > x_range[1] - margin
        or min_y < y_range[0] + margin
        or max_y > y_range[1] - margin
    ):
        raise ValueError(
            "normalized polygon does not fit inside the CFD domain "
            f"x={x_range}, y={y_range} with margin={margin}"
        )


def validate_geometry(points: np.ndarray, min_points: int = 20) -> None:
    closed = close_loop(points)
    if not np.all(np.isfinite(closed)):
        raise ValueError("shape contains NaN or infinite coordinates")
    if len(open_loop(closed)) < min_points:
        raise ValueError(f"shape has too few points: {len(open_loop(closed))}")
    if not np.allclose(closed[0], closed[-1]):
        raise ValueError("shape is not closed")
    if polygon_area(closed) <= 0.0:
        raise ValueError(f"shape has non-positive polygon area: {polygon_area(closed)}")
    validate_domain_fit(closed)
