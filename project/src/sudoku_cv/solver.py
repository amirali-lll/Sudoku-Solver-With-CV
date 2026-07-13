from __future__ import annotations

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


def solve_sudoku(board: Board) -> bool:
    empty = find_empty(board)
    if empty is None:
        return True

    row_index, col_index = empty
    for number in range(1, 10):
        if is_valid(board, number, (row_index, col_index)):
            board[row_index][col_index] = number
            if solve_sudoku(board):
                return True
            board[row_index][col_index] = 0

    return False
