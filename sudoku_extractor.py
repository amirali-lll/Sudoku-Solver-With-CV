"""Reusable Sudoku grid detection, perspective correction, and cell extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


Image = NDArray[np.uint8]


class GridNotFoundError(RuntimeError):
    """Raised when no plausible Sudoku boundary can be found."""


@dataclass(frozen=True)
class ExtractorConfig:
    """Parameters that control grid and cell extraction."""

    output_size: int = 450
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    min_grid_area_ratio: float = 0.08
    cell_margin_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.output_size < 90:
            raise ValueError("output_size must be at least 90 pixels")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be an odd integer >= 3")
        if not 0 < self.min_grid_area_ratio <= 1:
            raise ValueError("min_grid_area_ratio must be in (0, 1]")
        if not 0 <= self.cell_margin_ratio < 0.4:
            raise ValueError("cell_margin_ratio must be in [0, 0.4)")


@dataclass
class ExtractionResult:
    """All useful outputs from one successful extraction."""

    original: Image
    thresholded: Image
    contour_preview: Image
    warped_grid: Image
    warped_gray: Image
    corners: NDArray[np.float32]
    transform: NDArray[np.float32]
    cells: list[Image]


class SudokuExtractor:
    """Extract a perspective-corrected Sudoku grid and its 81 cells."""

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.config = config or ExtractorConfig()

    def extract_file(self, image_path: str | Path) -> ExtractionResult:
        """Read and extract one image."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.extract(image)

    def extract(self, image: Image) -> ExtractionResult:
        """Extract a Sudoku grid from an OpenCV BGR image."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input must be a BGR image with three channels")

        gray, thresholded = self._preprocess(image)
        corners = self._find_grid_corners(gray, thresholded)
        warped_grid, transform = self._warp(image, corners)
        warped_gray = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)
        cells = self._slice_cells(warped_gray)
        contour_preview = self._draw_contour(image, corners)

        return ExtractionResult(
            original=image.copy(),
            thresholded=thresholded,
            contour_preview=contour_preview,
            warped_grid=warped_grid,
            warped_gray=warped_gray,
            corners=corners,
            transform=transform,
            cells=cells,
        )

    def save_debug_outputs(
        self,
        result: ExtractionResult,
        output_dir: str | Path,
    ) -> None:
        """Save presentation images and the 81 clean cell crops."""
        output_dir = Path(output_dir)
        cells_dir = output_dir / "cells"
        cells_dir.mkdir(parents=True, exist_ok=True)

        self._write_image(output_dir / "1_original.jpg", result.original)
        self._write_image(output_dir / "2_thresholded.jpg", result.thresholded)
        self._write_image(
            output_dir / "3_contour_drawn.jpg", result.contour_preview
        )
        self._write_image(output_dir / "4_warped_grid.jpg", result.warped_grid)
        self._write_image(
            output_dir / "5_cells_compilation.jpg",
            self.make_cell_compilation(result.cells),
        )

        for index, cell in enumerate(result.cells):
            row, column = divmod(index, 9)
            self._write_image(
                cells_dir / f"cell_{row + 1}_{column + 1}.png",
                cell,
            )

    @staticmethod
    def make_cell_compilation(cells: list[Image], gap: int = 2) -> Image:
        """Lay 81 cells out in row-major order for quick visual inspection."""
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells, received {len(cells)}")

        cell_height = max(cell.shape[0] for cell in cells)
        cell_width = max(cell.shape[1] for cell in cells)
        canvas_height = 9 * cell_height + 10 * gap
        canvas_width = 9 * cell_width + 10 * gap
        canvas = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)

        for index, cell in enumerate(cells):
            row, column = divmod(index, 9)
            y = gap + row * (cell_height + gap)
            x = gap + column * (cell_width + gap)
            canvas[y : y + cell.shape[0], x : x + cell.shape[1]] = cell
        return canvas

    def _preprocess(self, image: Image) -> tuple[Image, Image]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # A small blur suppresses print/paper texture without erasing thin grid lines.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresholded = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )
        # Closing reconnects grid lines broken by glare or low-contrast printing.
        thresholded = cv2.morphologyEx(
            thresholded,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        return gray, thresholded

    def _find_grid_corners(
        self,
        gray: Image,
        thresholded: Image,
    ) -> NDArray[np.float32]:
        height, width = gray.shape
        image_area = float(height * width)
        min_area = image_area * self.config.min_grid_area_ratio

        # Canny supplies a useful fallback when adaptive thresholding joins the
        # grid to nearby text or page edges.
        canny = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        sources = (thresholded, canny)
        candidates: list[tuple[float, NDArray[np.float32]]] = []
        seen: set[tuple[int, ...]] = set()

        # A line-component fallback separates a 9x9 lattice from large nearby
        # rectangles (for example, a colored newspaper panel joined to the grid).
        for points, confidence_bonus in self._line_grid_candidates(
            thresholded, min_area
        ):
            polygon_area = cv2.contourArea(points)
            score = self._score_candidate(
                thresholded,
                points,
                polygon_area / image_area,
            )
            candidates.append((score + confidence_bonus, points))

        for source in sources:
            contours, _ = cv2.findContours(
                source,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
                area = cv2.contourArea(contour)
                if area < min_area:
                    break

                hull = cv2.convexHull(contour)
                perimeter = cv2.arcLength(hull, True)
                for epsilon_ratio in (0.01, 0.02, 0.03, 0.04):
                    polygon = cv2.approxPolyDP(
                        hull, epsilon_ratio * perimeter, True
                    )
                    if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                        continue

                    points = self._order_corners(
                        polygon.reshape(4, 2).astype(np.float32)
                    )
                    key = tuple(np.round(points / 4).astype(int).ravel())
                    if key in seen:
                        continue
                    seen.add(key)

                    polygon_area = cv2.contourArea(points)
                    if polygon_area < min_area:
                        continue
                    score = self._score_candidate(
                        thresholded,
                        points,
                        polygon_area / image_area,
                    )
                    candidates.append((score, points))
                    break

        if not candidates:
            raise GridNotFoundError(
                "No four-corner contour large enough to be a Sudoku grid was found"
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_points = candidates[0]
        if best_score < 0.22:
            raise GridNotFoundError(
                "Four-corner contours were found, but none had Sudoku-like grid lines"
            )
        return best_points

    def _line_grid_candidates(
        self,
        thresholded: Image,
        min_area: float,
    ) -> list[tuple[NDArray[np.float32], float]]:
        """Find dense connected lattices using horizontal/vertical morphology."""
        height, width = thresholded.shape
        horizontal = cv2.morphologyEx(
            thresholded,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(15, width // 35), 1)
            ),
        )
        vertical = cv2.morphologyEx(
            thresholded,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, max(15, height // 35))
            ),
        )
        grid_lines = cv2.bitwise_or(horizontal, vertical)
        intersections = cv2.bitwise_and(horizontal, vertical)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            grid_lines
        )
        _, _, intersection_stats, intersection_centers = (
            cv2.connectedComponentsWithStats(intersections)
        )
        valid_intersections = (
            intersection_stats[1:, cv2.CC_STAT_AREA] >= 2
        )
        centers = np.rint(intersection_centers[1:][valid_intersections]).astype(int)
        if len(centers):
            center_x = np.clip(centers[:, 0], 0, width - 1)
            center_y = np.clip(centers[:, 1], 0, height - 1)
            intersection_counts = np.bincount(
                labels[center_y, center_x],
                minlength=component_count,
            )
        else:
            intersection_counts = np.zeros(component_count, dtype=int)

        candidates: list[tuple[NDArray[np.float32], float]] = []
        component_indices = sorted(
            range(1, component_count),
            key=lambda index: stats[index, cv2.CC_STAT_WIDTH]
            * stats[index, cv2.CC_STAT_HEIGHT],
            reverse=True,
        )

        for component_index in component_indices[:10]:
            x = stats[component_index, cv2.CC_STAT_LEFT]
            y = stats[component_index, cv2.CC_STAT_TOP]
            component_width = stats[component_index, cv2.CC_STAT_WIDTH]
            component_height = stats[component_index, cv2.CC_STAT_HEIGHT]
            box_area = component_width * component_height
            aspect = component_width / max(component_height, 1)
            if box_area < min_area or not 0.45 <= aspect <= 2.2:
                continue

            intersection_count = int(intersection_counts[component_index])
            if intersection_count < 45:
                continue

            component_y, component_x = np.where(labels == component_index)
            corners = self._fit_component_boundaries(
                component_x, component_y, width, height
            )
            if corners is None or cv2.contourArea(corners) < min_area:
                continue

            # A real Sudoku normally contributes about 100 line intersections.
            bonus = 0.12 * min(intersection_count / 100.0, 1.0)
            candidates.append((corners, bonus))
        return candidates

    @staticmethod
    def _fit_component_boundaries(
        x_coordinates: NDArray[np.int64],
        y_coordinates: NDArray[np.int64],
        width: int,
        height: int,
    ) -> NDArray[np.float32] | None:
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

        def fit_line(points: NDArray[np.int32]) -> NDArray[np.float64]:
            vx, vy, x0, y0 = cv2.fitLine(
                points.astype(np.float32),
                cv2.DIST_HUBER,
                0,
                0.01,
                0.01,
            ).ravel()
            # Homogeneous line ax + by + c = 0.
            return np.array([vy, -vx, -(vy * x0 - vx * y0)])

        def intersect(
            first: NDArray[np.float64],
            second: NDArray[np.float64],
        ) -> NDArray[np.float64] | None:
            point = np.cross(first, second)
            if abs(point[2]) < 1e-6:
                return None
            return point[:2] / point[2]

        left_line = fit_line(left)
        right_line = fit_line(right)
        top_line = fit_line(top)
        bottom_line = fit_line(bottom)
        corner_list = (
            intersect(top_line, left_line),
            intersect(top_line, right_line),
            intersect(bottom_line, right_line),
            intersect(bottom_line, left_line),
        )
        if any(point is None for point in corner_list):
            return None

        corners = np.asarray(corner_list, dtype=np.float32)
        if not np.all(np.isfinite(corners)):
            return None
        margin = 0.1 * max(width, height)
        if (
            np.any(corners[:, 0] < -margin)
            or np.any(corners[:, 0] > width - 1 + margin)
            or np.any(corners[:, 1] < -margin)
            or np.any(corners[:, 1] > height - 1 + margin)
        ):
            return None
        return corners

    def _score_candidate(
        self,
        thresholded: Image,
        points: NDArray[np.float32],
        area_ratio: float,
    ) -> float:
        """Prefer quadrilaterals with ten regularly spaced lines per axis."""
        size = 270
        warped, _ = self._warp(thresholded, points, size=size)
        binary = warped.astype(np.float32) / 255.0
        horizontal_projection = binary.mean(axis=1)
        vertical_projection = binary.mean(axis=0)

        expected_positions = np.linspace(0, size - 1, 10)
        radius = max(2, size // 55)

        def line_score(projection: NDArray[np.float32]) -> float:
            strengths = []
            for position in expected_positions:
                center = int(round(position))
                start = max(0, center - radius)
                stop = min(size, center + radius + 1)
                strengths.append(float(projection[start:stop].max()))
            return float(np.mean(strengths))

        grid_score = min(
            line_score(horizontal_projection),
            line_score(vertical_projection),
        )

        top_left, top_right, bottom_right, bottom_left = points
        widths = (
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )
        heights = (
            np.linalg.norm(bottom_left - top_left),
            np.linalg.norm(bottom_right - top_right),
        )
        aspect_ratio = np.mean(widths) / max(np.mean(heights), 1.0)
        square_score = float(np.exp(-abs(np.log(max(aspect_ratio, 1e-6)))))

        # Line evidence dominates; area only breaks ties between real grids.
        return 0.72 * grid_score + 0.18 * square_score + 0.10 * min(area_ratio, 1.0)

    def _warp(
        self,
        image: Image,
        corners: NDArray[np.float32],
        size: int | None = None,
    ) -> tuple[Image, NDArray[np.float32]]:
        size = size or self.config.output_size
        destination = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners, destination)
        warped = cv2.warpPerspective(image, transform, (size, size))
        return warped, transform

    def _slice_cells(self, warped_gray: Image) -> list[Image]:
        boundaries = np.rint(
            np.linspace(0, warped_gray.shape[0], 10)
        ).astype(int)
        cells: list[Image] = []

        for row in range(9):
            for column in range(9):
                y1, y2 = boundaries[row], boundaries[row + 1]
                x1, x2 = boundaries[column], boundaries[column + 1]
                margin_y = max(1, round((y2 - y1) * self.config.cell_margin_ratio))
                margin_x = max(1, round((x2 - x1) * self.config.cell_margin_ratio))
                cell = warped_gray[
                    y1 + margin_y : y2 - margin_y,
                    x1 + margin_x : x2 - margin_x,
                ]
                cells.append(cell.copy())
        return cells

    @staticmethod
    def _order_corners(points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return corners in top-left, top-right, bottom-right, bottom-left order."""
        ordered = np.empty((4, 2), dtype=np.float32)
        coordinate_sum = points.sum(axis=1)
        coordinate_difference = np.diff(points, axis=1).ravel()
        ordered[0] = points[np.argmin(coordinate_sum)]
        ordered[2] = points[np.argmax(coordinate_sum)]
        ordered[1] = points[np.argmin(coordinate_difference)]
        ordered[3] = points[np.argmax(coordinate_difference)]
        return ordered

    @staticmethod
    def _draw_contour(
        image: Image,
        corners: NDArray[np.float32],
    ) -> Image:
        preview = image.copy()
        polygon = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [polygon], True, (0, 255, 0), 3, cv2.LINE_AA)
        for index, point in enumerate(polygon.reshape(4, 2), start=1):
            center = tuple(point)
            cv2.circle(preview, center, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                preview,
                str(index),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return preview

    @staticmethod
    def _write_image(path: Path, image: Image) -> None:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write image: {path}")
