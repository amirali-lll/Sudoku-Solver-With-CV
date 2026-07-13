import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sudoku_cv.pipeline import SudokuPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Solve a Sudoku image end to end.")
    parser.add_argument("--image", required=True, help="Path to a Sudoku image.")
    parser.add_argument("--model", required=True, help="Path to the trained digit model checkpoint.")
    parser.add_argument("--save-overlay", default="outputs/solved_overlay.jpg", help="Where to save the solved image.")
    parser.add_argument("--debug-dir", default="outputs/debug", help="Where to save intermediate images.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save every pipeline stage, cell crop, digit crop, prediction, and metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pipeline = SudokuPipeline(model_path=args.model)
    result = pipeline.solve_image(
        args.image,
        save_overlay_path=args.save_overlay,
        debug_dir=args.debug_dir,
        debug=args.debug,
    )

    print("Detected board:")
    for row in result["board"]:
        print(row)
    print("Solved board:")
    for row in result["solved_board"]:
        print(row)
    print(f"Solvable: {result['solvable']}")
    print(f"Saved overlay to {Path(args.save_overlay)}")
    if args.debug:
        print(f"Saved debug artifacts to {Path(args.debug_dir)}")


if __name__ == "__main__":
    main()
