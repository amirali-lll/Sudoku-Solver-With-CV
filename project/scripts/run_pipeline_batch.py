#!/usr/bin/env python3
"""
Run scripts/run_pipeline.py for every image in a folder.

Usage:
    python run_pipeline_batch.py --input-dir raw_suduku/images \
        --model outputs/digit_model.pt \
        --overlay-dir outputs/solved \
        --debug-root outputs/debug \
        [--pipeline-script scripts/run_pipeline.py] \
        [--debug] [--dry-run] [--continue-on-error]

For each image "images-6.jpeg" found in --input-dir, it runs:

    python scripts/run_pipeline.py \
        --image raw_suduku/images/images-6.jpeg \
        --model outputs/digit_model.pt \
        --save-overlay outputs/solved/images-6-solved.jpg \
        --debug-dir outputs/debug/images-6-test \
        --debug   (only if --debug is passed)
"""

import argparse
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Batch-run run_pipeline.py over a folder of images.")
    p.add_argument("--input-dir", required=True, help="Folder containing input images.")
    p.add_argument("--model", required=True, help="Path to the digit model (--model arg of run_pipeline.py).")
    p.add_argument("--overlay-dir", required=True, help="Folder to write solved overlay images into.")
    p.add_argument("--debug-root", required=True, help="Root folder to write per-image debug dirs into.")
    p.add_argument(
        "--pipeline-script",
        default="scripts/run_pipeline.py",
        help="Path to run_pipeline.py (default: scripts/run_pipeline.py).",
    )
    p.add_argument("--debug", action="store_true", help="Pass --debug to run_pipeline.py for every image.")
    p.add_argument("--dry-run", action="store_true", help="Print commands instead of running them.")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining images if one fails (default: stop on first failure).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders of --input-dir when looking for images.",
    )
    return p.parse_args()


def find_images(input_dir: Path, recursive: bool):
    pattern_fn = input_dir.rglob if recursive else input_dir.glob
    images = [
        f for f in pattern_fn("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    overlay_dir = Path(args.overlay_dir)
    debug_root = Path(args.debug_root)

    if not input_dir.is_dir():
        sys.exit(f"Error: input dir not found: {input_dir}")

    images = find_images(input_dir, args.recursive)
    if not images:
        sys.exit(f"No images found in {input_dir}")

    overlay_dir.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} image(s) in {input_dir}\n")

    failures = []

    for image_path in images:
        stem = image_path.stem  # e.g. "images-6"
        overlay_path = overlay_dir / f"{stem}-solved.jpg"
        debug_dir = debug_root / f"{stem}-test"

        cmd = [
            sys.executable, args.pipeline_script,
            "--image", str(image_path),
            "--model", args.model,
            "--save-overlay", str(overlay_path),
            "--debug-dir", str(debug_dir),
        ]
        if args.debug:
            cmd.append("--debug")

        print("Running:", " ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  -> FAILED (exit code {result.returncode}): {image_path.name}")
            failures.append(image_path.name)
            if not args.continue_on_error:
                sys.exit(1)
        else:
            print(f"  -> OK: {image_path.name}")

    print("\nDone.")
    if failures:
        print(f"{len(failures)} image(s) failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
