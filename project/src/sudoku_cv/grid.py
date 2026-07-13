"""Sudoku board detection, perspective correction, and digit-cell extraction.

The public functions intentionally preserve the API used by ``pipeline.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class GridNotFoundError(RuntimeError):
    """Raised when no plausible Sudoku boundary can be found."""


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def preprocess_for_contours(image: np.ndarray) -> np.ndarray:
    """Create a binary image in which grid lines are foreground."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a BGR image with three channels")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    return cv2.morphologyEx(
        thresholded, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )


def order_points(points: np.ndarray) -> np.ndarray:
    """Return corners in top-left, top-right, bottom-right, bottom-left order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _warp(image: np.ndarray, corners: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Warp using corners already ordered TL, TR, BR, BL."""
    destination = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (size, size)), matrix


def _fit_component_boundaries(x_coordinates: np.ndarray, y_coordinates: np.ndarray,
                              width: int, height: int) -> np.ndarray | None:
    """Fit and intersect the four outer lines of a connected lattice."""
    min_x = np.full(height, width, dtype=np.int32)
    max_x = np.full(height, -1, dtype=np.int32)
    min_y = np.full(width, height, dtype=np.int32)
    max_y = np.full(width, -1, dtype=np.int32)
    np.minimum.at(min_x, y_coordinates, x_coordinates)
    np.maximum.at(max_x, y_coordinates, x_coordinates)
    np.minimum.at(min_y, x_coordinates, y_coordinates)
    np.maximum.at(max_y, x_coordinates, y_coordinates)
    rows = np.flatnonzero(max_x >= 0)
    columns = np.flatnonzero(max_y >= 0)
    if len(rows) < 20 or len(columns) < 20:
        return None

    left = np.column_stack((min_x[rows], rows))
    right = np.column_stack((max_x[rows], rows))
    top = np.column_stack((columns, min_y[columns]))
    bottom = np.column_stack((columns, max_y[columns]))

    def fit_line(points: np.ndarray) -> np.ndarray:
        vx, vy, x0, y0 = cv2.fitLine(
            points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).ravel()
        return np.array([vy, -vx, -(vy * x0 - vx * y0)])

    def intersect(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
        point = np.cross(first, second)
        if abs(point[2]) < 1e-6:
            return None
        return point[:2] / point[2]

    left_line, right_line = fit_line(left), fit_line(right)
    top_line, bottom_line = fit_line(top), fit_line(bottom)
    corners = (
        intersect(top_line, left_line), intersect(top_line, right_line),
        intersect(bottom_line, right_line), intersect(bottom_line, left_line),
    )
    if any(point is None for point in corners):
        return None
    result = np.asarray(corners, dtype=np.float32)
    margin = 0.1 * max(width, height)
    if not np.all(np.isfinite(result)):
        return None
    if (
        np.any(result[:, 0] < -margin)
        or np.any(result[:, 0] > width - 1 + margin)
        or np.any(result[:, 1] < -margin)
        or np.any(result[:, 1] > height - 1 + margin)
    ):
        return None
    return result


def _line_grid_candidates(binary: np.ndarray, min_area: float) -> list[tuple[np.ndarray, float]]:
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 35), 1)),
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 35))),
    )
    lines = cv2.bitwise_or(horizontal, vertical)
    intersections = cv2.bitwise_and(horizontal, vertical)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines)
    _, _, intersection_stats, centers = cv2.connectedComponentsWithStats(intersections)
    valid = intersection_stats[1:, cv2.CC_STAT_AREA] >= 2
    centers = np.rint(centers[1:][valid]).astype(int)
    if len(centers):
        xs = np.clip(centers[:, 0], 0, width - 1)
        ys = np.clip(centers[:, 1], 0, height - 1)
        intersection_counts = np.bincount(labels[ys, xs], minlength=count)
    else:
        intersection_counts = np.zeros(count, dtype=int)

    indices = sorted(
        range(1, count),
        key=lambda i: stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT],
        reverse=True,
    )
    candidates = []
    for index in indices[:10]:
        component_width = stats[index, cv2.CC_STAT_WIDTH]
        component_height = stats[index, cv2.CC_STAT_HEIGHT]
        box_area = component_width * component_height
        aspect = component_width / max(component_height, 1)
        if box_area < min_area or not 0.45 <= aspect <= 2.2:
            continue
        intersections_count = int(intersection_counts[index])
        if intersections_count < 45:
            continue
        component_y, component_x = np.where(labels == index)
        corners = _fit_component_boundaries(component_x, component_y, width, height)
        if corners is not None and cv2.contourArea(corners) >= min_area:
            candidates.append((corners, 0.12 * min(intersections_count / 100.0, 1.0)))
    return candidates


def _score_candidate(binary: np.ndarray, points: np.ndarray, area_ratio: float) -> float:
    warped, _ = _warp(binary, points, 270)
    binary_float = warped.astype(np.float32) / 255.0
    projections = (binary_float.mean(axis=1), binary_float.mean(axis=0))
    expected = np.linspace(0, 269, 10)
    radius = max(2, 270 // 55)

    def line_score(projection: np.ndarray) -> float:
        strengths = []
        for position in expected:
            center = int(round(position))
            strengths.append(float(projection[max(0, center - radius):min(270, center + radius + 1)].max()))
        return float(np.mean(strengths))

    grid_score = min(line_score(projections[0]), line_score(projections[1]))
    top_left, top_right, bottom_right, bottom_left = points
    widths = (np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    heights = (np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    aspect = np.mean(widths) / max(np.mean(heights), 1.0)
    square_score = float(np.exp(-abs(np.log(max(aspect, 1e-6)))))
    return 0.72 * grid_score + 0.18 * square_score + 0.10 * min(area_ratio, 1.0)


def find_board_contour(binary_image: np.ndarray, gray_image: np.ndarray | None = None) -> np.ndarray:
    """Find the quadrilateral most likely to contain the Sudoku lattice."""
    if binary_image.ndim != 2:
        raise ValueError("binary_image must be a single-channel image")
    height, width = binary_image.shape
    image_area = float(height * width)
    min_area = image_area * 0.08
    candidates: list[tuple[float, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()

    for points, bonus in _line_grid_candidates(binary_image, min_area):
        area = cv2.contourArea(points)
        candidates.append((_score_candidate(binary_image, points, area / image_area) + bonus, points))

    # Fix: Use grayscale for Canny if provided; otherwise fallback to binary to protect the API
    canny_source = gray_image if gray_image is not None else binary_image
    canny = cv2.Canny(cv2.GaussianBlur(canny_source, (5, 5), 0), 50, 150)
    
    for source in (binary_image, canny):
        contours, _ = cv2.findContours(source, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
            if cv2.contourArea(contour) < min_area:
                break
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            for epsilon_ratio in (0.01, 0.02, 0.03, 0.04):
                polygon = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue
                points = order_points(polygon.reshape(4, 2))
                key = tuple(np.round(points / 4).astype(int).ravel())
                if key in seen:
                    continue
                seen.add(key)
                area = cv2.contourArea(points)
                if area >= min_area:
                    candidates.append((_score_candidate(binary_image, points, area / image_area), points))
                break

    if not candidates:
        raise GridNotFoundError("No four-corner contour large enough to be a Sudoku grid was found")
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, points = candidates[0]
    if score < 0.22:
        raise GridNotFoundError("Four-corner contours were found, but none had Sudoku-like grid lines")
    return points


def warp_board(image: np.ndarray, contour: np.ndarray, size: int = 450) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a public contour, normalizing its input corner order first."""
    return _warp(image, order_points(contour), size)


def extract_board(image: np.ndarray, size: int = 450) -> Tuple[np.ndarray, np.ndarray]:
    preprocessed = preprocess_for_contours(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = find_board_contour(preprocessed, gray_image=gray)
    return warp_board(image, contour, size=size)


def split_cells(board_image: np.ndarray) -> List[np.ndarray]:
    """Split a warped board into 81 crops while removing grid-line borders."""
    boundaries_y = np.rint(np.linspace(0, board_image.shape[0], 10)).astype(int)
    boundaries_x = np.rint(np.linspace(0, board_image.shape[1], 10)).astype(int)
    cells: List[np.ndarray] = []
    for row in range(9):
        for column in range(9):
            y1, y2 = boundaries_y[row], boundaries_y[row + 1]
            x1, x2 = boundaries_x[column], boundaries_x[column + 1]
            # Keep almost the entire cell.  Grid removal is performed after
            # thresholding, so a large crop here could clip a real stroke.
            margin_y = max(1, round((y2 - y1) * 0.015))
            margin_x = max(1, round((x2 - x1) * 0.015))
            cells.append(board_image[y1 + margin_y:y2 - margin_y, x1 + margin_x:x2 - margin_x].copy())
    return cells


def _select_digit_component(mask: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """Select the most digit-like component rather than simply the largest blob."""
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if component_count <= 1:
        return None

    height, width = mask.shape
    image_area = float(height * width)
    center = np.array([width / 2.0, height / 2.0])
    candidates: list[tuple[float, tuple[int, int, int, int, int]]] = []
    for index in range(1, component_count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if component_width < 3 or component_height < 3:
            continue
        if area < max(12, round(image_area * 0.002)):
            continue
        aspect = component_width / max(component_height, 1)
        if not 0.15 <= aspect <= 3.5:
            continue

        # A genuine digit can touch the original cell edge, but a component
        # occupying most of the padded border is usually leftover grid.
        touches_border = x == 0 or y == 0 or x + component_width == width or y + component_height == height
        if touches_border and (
            component_width > width * 0.4 or component_height > height * 0.4
        ):
            continue

        fill_ratio = area / float(component_width * component_height)
        distance = float(np.linalg.norm(centroids[index] - center) / max(width, height))
        area_score = np.log1p(area) / np.log1p(image_area)
        compactness_bonus = min(fill_ratio, 1.0)
        score = 0.62 * area_score + 0.22 * compactness_bonus - 0.35 * distance
        candidates.append((score, (index, x, y, component_width, component_height)))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def extract_digit_image(cell_image: np.ndarray, output_size: int = 28) -> np.ndarray:
    """Extract one digit as a centered white-on-black image."""
    empty = np.zeros((output_size, output_size), dtype=np.uint8)
    if cell_image is None or cell_image.size == 0:
        return empty

    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY) if cell_image.ndim == 3 else np.asarray(cell_image)
    gray = gray.astype(np.uint8, copy=False)
    gray = cv2.copyMakeBorder(gray, 3, 3, 3, 3, cv2.BORDER_REPLICATE)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    height, width = binary.shape
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, height // 3)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 3), 1))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    grid = cv2.bitwise_or(vertical, horizontal)
    digit_mask = cv2.subtract(binary, grid)
    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    digit_mask = cv2.dilate(digit_mask, np.ones((2, 2), np.uint8), iterations=1)

    # Distance-transform thresholding suppresses thin remnants of grid lines.
    distance = cv2.distanceTransform(digit_mask, cv2.DIST_L2, 5)
    if distance.max() > 0:
        _, distance_mask = cv2.threshold(distance, 0.25 * distance.max(), 255, cv2.THRESH_BINARY)
        distance_mask = distance_mask.astype(np.uint8)
        if cv2.countNonZero(distance_mask) >= max(12, round(digit_mask.size * 0.002)):
            digit_mask = distance_mask

    if cv2.countNonZero(digit_mask) < max(18, round(digit_mask.size * 0.003)):
        return empty
    bounds = _select_digit_component(digit_mask)
    if bounds is None:
        return empty

    component_index, x, y, component_width, component_height = bounds
    _, labels, _, _ = cv2.connectedComponentsWithStats(digit_mask, connectivity=8)
    component = (labels[y:y + component_height, x:x + component_width] == component_index).astype(np.uint8) * 255
    padding = 2
    component = cv2.copyMakeBorder(component, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    target = max(1, int(round(output_size * 0.75)))
    scale = target / max(component.shape)
    resized_width = max(1, int(round(component.shape[1] * scale)))
    resized_height = max(1, int(round(component.shape[0] * scale)))
    resized = cv2.resize(component, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    result = np.zeros((output_size, output_size), dtype=np.uint8)
    start_y = (output_size - resized_height) // 2
    start_x = (output_size - resized_width) // 2
    result[start_y:start_y + resized_height, start_x:start_x + resized_width] = resized
    return result