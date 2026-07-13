from dataclasses import dataclass


@dataclass(frozen=True)
class SudokuConfig:
    board_size: int = 450
    cell_size: int = 28
    min_cell_foreground_ratio: float = 0.02
    digit_confidence_threshold: float = 0.55
    solver_timeout_seconds: float = 5.0
