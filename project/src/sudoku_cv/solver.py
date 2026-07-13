from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple


Board = List[List[int]]


def find_empty(board: Sequence[Sequence[int]]) -> Optional[Tuple[int, int]]:
    for row_index, row in enumerate(board):
        for col_index, value in enumerate(row):
            if value == 0:
                return row_index, col_index
    return None


def is_valid(board: Sequence[Sequence[int]], number: int, position: Tuple[int, int]) -> bool:
    row_index, col_index = position

    for index in range(9):
        if board[row_index][index] == number and index != col_index:
            return False
        if board[index][col_index] == number and index != row_index:
            return False

    box_row = (row_index // 3) * 3
    box_col = (col_index // 3) * 3
    for local_row in range(box_row, box_row + 3):
        for local_col in range(box_col, box_col + 3):
            if board[local_row][local_col] == number and (local_row, local_col) != position:
                return False
    return True


class SolverTimeoutError(TimeoutError):
    """Raised internally when the Sudoku search exceeds its time budget."""


def solve_sudoku(board: Board, time_limit_seconds: float | None = 5.0) -> bool:
    """Solve ``board`` in place, returning ``False`` when unsolvable or timed out.

    ``time_limit_seconds=None`` disables the limit.  When the limit is reached,
    the board is restored to its state before this function was called.
    """
    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive or None")

    deadline = None if time_limit_seconds is None else time.monotonic() + time_limit_seconds
    original_board = [row[:] for row in board]

    def search() -> bool:
        if deadline is not None and time.monotonic() >= deadline:
            raise SolverTimeoutError

        empty = find_empty(board)
        if empty is None:
            return True

        row_index, col_index = empty
        for number in range(1, 10):
            if is_valid(board, number, (row_index, col_index)):
                board[row_index][col_index] = number
                if search():
                    return True
                board[row_index][col_index] = 0

        return False

    try:
        return search()
    except SolverTimeoutError:
        for row_index, row in enumerate(original_board):
            board[row_index][:] = row
        return False
