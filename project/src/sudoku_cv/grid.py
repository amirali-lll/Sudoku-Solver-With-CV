from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def preprocess_for_contours(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def order_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def warp_board(image: np.ndarray, contour: np.ndarray, size: int = 450) -> Tuple[np.ndarray, np.ndarray]:
    ordered = order_points(contour)
    destination = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    return warped, matrix


def find_board_contour(binary_image: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in Sudoku image")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx

    return cv2.boxPoints(cv2.minAreaRect(contours[0])).astype(np.float32)


def extract_board(image: np.ndarray, size: int = 450) -> Tuple[np.ndarray, np.ndarray]:
    preprocessed = preprocess_for_contours(image)
    contour = find_board_contour(preprocessed)
    warped, matrix = warp_board(image, contour, size=size)
    return warped, matrix


def split_cells(board_image: np.ndarray) -> List[np.ndarray]:
    cell_height = board_image.shape[0] // 9
    cell_width = board_image.shape[1] // 9
    cells: List[np.ndarray] = []
    for row_index in range(9):
        for col_index in range(9):
            top = row_index * cell_height
            left = col_index * cell_width
            cell = board_image[top : top + cell_height, left : left + cell_width]
            cells.append(cell)
    return cells


def extract_digit_image(cell_image: np.ndarray, output_size: int = 28) -> np.ndarray:
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    height, width = thresh.shape
    margin = int(min(height, width) * 0.08)
    thresh[:margin, :] = 0
    thresh[-margin:, :] = 0
    thresh[:, :margin] = 0
    thresh[:, -margin:] = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    digit = thresh[y : y + h, x : x + w]

    if digit.size == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    square_size = max(w, h)
    square = np.zeros((square_size, square_size), dtype=np.uint8)
    x_offset = (square_size - w) // 2
    y_offset = (square_size - h) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = digit
    resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized
