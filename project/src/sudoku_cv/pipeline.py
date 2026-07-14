from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from .config import SudokuConfig
from .digits import classify_digit, load_digit_model
from .grid import (
    extract_board,
    extract_digit_image,
    find_board_contour,
    load_image,
    preprocess_for_contours,
    split_cells,
)
from .helper import make_grid_compilation
from .solver import solve_sudoku

LOGGER = logging.getLogger(__name__)


class SudokuPipeline:
    def __init__(self, model_path: str, config: SudokuConfig | None = None):
        self.config = config or SudokuConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading digit model: %s (device=%s)", model_path, self.device)
        self.model = load_digit_model(model_path, self.device)
        LOGGER.info("Digit model loaded successfully")

    def detect_board(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        LOGGER.info("Detecting Sudoku board in image with shape %s", image.shape)
        warped_board, matrix = extract_board(image, size=self.config.board_size)
        LOGGER.info("Board detected and warped to %s", warped_board.shape)
        return warped_board, matrix

    def recognize_board(
        self,
        warped_board: np.ndarray,
        debug_dir: Path | None = None,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        LOGGER.info("Cleaning whole board before splitting")

        
        LOGGER.info("Splitting warped board into 81 cells")
        cells = split_cells(warped_board)
        board: List[List[int]] = []
        confidences: List[List[float]] = []


        raw_cells_for_debug: List[np.ndarray] = []
        processed_digits_for_debug: List[np.ndarray] = []
        
        if debug_dir is not None:
            cells_dir = debug_dir / "cells"
            digits_dir = debug_dir / "digits"
            cells_dir.mkdir(parents=True, exist_ok=True)
            digits_dir.mkdir(parents=True, exist_ok=True)

        for row_index in range(9):
            row: List[int] = []
            row_confidence: List[float] = []
            for col_index in range(9):
                cell = cells[row_index * 9 + col_index]
                digit_image = extract_digit_image(cell, output_size=self.config.cell_size)
                if debug_dir is not None:
                    cell_name = f"r{row_index + 1}c{col_index + 1}.png"
                    cv2.imwrite(str(debug_dir / "cells" / cell_name), cell)
                    cv2.imwrite(str(debug_dir / "digits" / cell_name), digit_image)
                    raw_cells_for_debug.append(cell)
                    processed_digits_for_debug.append(digit_image)
                prediction, confidence = classify_digit(
                    self.model,
                    digit_image,
                    self.device,
                    empty_threshold=self.config.min_cell_foreground_ratio,
                )

                if confidence < self.config.digit_confidence_threshold:
                    row.append(0)
                else:
                    row.append(prediction)
                row_confidence.append(confidence)

            board.append(row)
            confidences.append(row_confidence)
            LOGGER.info(
                "Recognized row %d/9: %s | confidences: %s",
                row_index + 1,
                row,
                " ".join(f"{value:.2f}" for value in row_confidence),
            )
            
        # --- NEW DEBUG SAVING LOGIC ---
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate the 9x9 visual grids
            raw_compilation = make_grid_compilation(raw_cells_for_debug)
            digit_compilation = make_grid_compilation(processed_digits_for_debug)
            
            # Save them
            cv2.imwrite(str(debug_dir / "06_raw_cells_grid.jpg"), raw_compilation)
            cv2.imwrite(str(debug_dir / "07_processed_digits_grid.jpg"), digit_compilation)
            LOGGER.info("Saved 9x9 cell compilations to debug directory.")
            

        LOGGER.info("Digit recognition complete: %d filled cells", sum(value != 0 for row in board for value in row))
        return board, confidences

    def _save_debug_artifacts(
        self,
        debug_dir: Path,
        image: np.ndarray,
        board: List[List[int]],
        confidences: List[List[float]],
        solved_board: List[List[int]],
        matrix: np.ndarray,
    ) -> None:
        """Save intermediate images and metadata for one pipeline run."""
        debug_dir.mkdir(parents=True, exist_ok=True)
        preprocessed = preprocess_for_contours(image)

        # Do not detect the board a second time for the debug image.  The
        # pipeline already used ``matrix`` to create ``warped_board.jpg``.
        # Running find_board_contour again can select a different candidate,
        # especially when the grayscale Canny fallback and the thresholded
        # contour candidates rank similarly.  Recover the exact source corners
        # from the transform that produced the warped board instead.
        warped_corners = np.array(
            [[0, 0], [self.config.board_size - 1, 0],
             [self.config.board_size - 1, self.config.board_size - 1],
             [0, self.config.board_size - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        inverse_matrix = np.linalg.inv(matrix)
        contour = cv2.perspectiveTransform(warped_corners, inverse_matrix)

        contour_image = image.copy()
        cv2.polylines(
            contour_image,
            [np.round(contour).astype(np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.imwrite(str(debug_dir / "01_original.jpg"), image)
        cv2.imwrite(str(debug_dir / "02_contours_preprocessed.png"), preprocessed)
        cv2.imwrite(str(debug_dir / "03_detected_board_contour.jpg"), contour_image)

        metadata = {
            "device": str(self.device),
            "board_size": self.config.board_size,
            "digit_confidence_threshold": self.config.digit_confidence_threshold,
            "board": board,
            "confidences": confidences,
            "solved_board": solved_board,
            "perspective_matrix": matrix.tolist(),
        }
        with (debug_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        LOGGER.info("Saved pipeline debug artifacts to %s", debug_dir)

    def _save_prediction_visualization(
        self,
        warped_board: np.ndarray,
        board: List[List[int]],
        confidences: List[List[float]],
        output_path: Path,
    ) -> None:
        """Save the detected board next to a cell-by-cell prediction grid."""
        board_size = max(450, int(warped_board.shape[0]))
        actual_board = cv2.resize(warped_board, (board_size, board_size))
        if actual_board.ndim == 2:
            actual_board = cv2.cvtColor(actual_board, cv2.COLOR_GRAY2BGR)

        prediction_board = np.full((board_size, board_size, 3), 255, dtype=np.uint8)
        cell_size = board_size // 9
        for row_index in range(9):
            for col_index in range(9):
                x1, y1 = col_index * cell_size, row_index * cell_size
                x2 = (col_index + 1) * cell_size if col_index == 8 else (col_index + 1) * cell_size
                y2 = (row_index + 1) * cell_size if row_index == 8 else (row_index + 1) * cell_size
                value = board[row_index][col_index]
                confidence = confidences[row_index][col_index]
                label = "-" if value == 0 else str(value)
                cv2.putText(
                    prediction_board,
                    label,
                    (x1 + cell_size // 3, y1 + int(cell_size * 0.48)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 180) if value == 0 else (30, 100, 30),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    prediction_board,
                    f"{confidence * 100:.0f}%",
                    (x1 + 4, y1 + int(cell_size * 0.82)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (70, 70, 70),
                    1,
                    lineType=cv2.LINE_AA,
                )

        for index in range(10):
            position = index * cell_size
            thickness = 2 if index % 3 == 0 else 1
            cv2.line(prediction_board, (position, 0), (position, board_size), (0, 0, 0), thickness)
            cv2.line(prediction_board, (0, position), (board_size, position), (0, 0, 0), thickness)

        title_height = 48
        actual_panel = cv2.copyMakeBorder(
            actual_board, title_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        prediction_panel = cv2.copyMakeBorder(
            prediction_board, title_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(actual_panel, "Actual board", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(
            prediction_panel,
            "Predictions (digit / confidence)",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            2,
        )
        visualization = cv2.hconcat([actual_panel, prediction_panel])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), visualization)

    def solve_board(self, board: List[List[int]]) -> Tuple[List[List[int]], bool]:
        solved_board = [row[:] for row in board]
        LOGGER.info("Starting backtracking solver (time limit=%.2f seconds)", self.config.solver_timeout_seconds)
        started = time.monotonic()
        solvable = solve_sudoku(
            solved_board,
            time_limit_seconds=self.config.solver_timeout_seconds,
        )
        LOGGER.info(
            "Backtracking solver finished in %.3f seconds: solvable=%s",
            time.monotonic() - started,
            solvable,
        )
        return solved_board, solvable

    def render_solution(
        self,
        original_image: np.ndarray,
        original_board: List[List[int]],
        solved_board: List[List[int]],
        board_matrix: np.ndarray,
        warped_size: int = 450,
    ) -> np.ndarray:
        solved_canvas = np.zeros((warped_size, warped_size, 3), dtype=np.uint8)
        cell_size = warped_size // 9

        for row_index in range(9):
            for col_index in range(9):
                if original_board[row_index][col_index] != 0:
                    continue
                value = solved_board[row_index][col_index]
                if value == 0:
                    continue
                center_x = col_index * cell_size + cell_size // 3
                center_y = row_index * cell_size + int(cell_size * 0.72)
                cv2.putText(
                    solved_canvas,
                    str(value),
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (159, 159, 146),
                    2,
                    lineType=cv2.LINE_AA,
                )

        inverse_matrix = np.linalg.inv(board_matrix)
        overlay = cv2.warpPerspective(
            solved_canvas,
            inverse_matrix,
            (original_image.shape[1], original_image.shape[0])
        )
        mask = overlay[:, :, 2] > 0

        combined = original_image.copy()
        combined[mask] = overlay[mask]
        # combined = cv2.addWeighted(original_image, 1.0, overlay, 3.0, 0.0)
        return combined

    def solve_image(
        self,
        image_path: str,
        save_overlay_path: str | None = None,
        debug_dir: str | None = None,
        debug: bool = False,
    ) -> Dict[str, object]:
        LOGGER.info("Starting pipeline for image: %s", image_path)
        LOGGER.info("Output overlay: %s | debug=%s", save_overlay_path, debug)
        image = load_image(image_path)
        LOGGER.info("Image loaded successfully: shape=%s", image.shape)
        warped_board, matrix = self.detect_board(image)
        debug_path = Path(debug_dir) if debug and debug_dir is not None else None
        board, confidences = self.recognize_board(warped_board, debug_dir=debug_path)
        LOGGER.info("Solving recognized board")
        solved_board, solvable = self.solve_board(board)
        LOGGER.info("Solver finished: solvable=%s", solvable)

        overlay = self.render_solution(image, board, solved_board, matrix, warped_size=self.config.board_size)

        if save_overlay_path is not None:
            output_path = Path(save_overlay_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), overlay):
                raise IOError(f"Could not write overlay image: {output_path}")
            LOGGER.info("Saved overlay: %s", output_path)

        if debug_path is not None:
            debug_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path / "warped_board.jpg"), warped_board)
            self._save_debug_artifacts(debug_path, image, board, confidences, solved_board, matrix)
            self._save_prediction_visualization(
                warped_board,
                board,
                confidences,
                debug_path / "05_predictions_with_confidence.jpg",
            )
            cv2.imwrite(str(debug_path / "04_solution_overlay.jpg"), overlay)
            LOGGER.info("Saved debug artifacts: %s", debug_path)

        LOGGER.info("Pipeline completed successfully")
        return {
            "board": board,
            "confidences": confidences,
            "solved_board": solved_board,
            "solvable": solvable,
            "overlay": overlay,
        }


def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Sudoku pipeline.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save-overlay", default="outputs/solved_overlay.jpg")
    parser.add_argument("--debug-dir", default="outputs/debug")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pipeline = SudokuPipeline(model_path=args.model)
    result = pipeline.solve_image(
        args.image,
        save_overlay_path=args.save_overlay,
        debug_dir=args.debug_dir,
        debug=args.debug,
    )
    for row in result["solved_board"]:
        print(row)
