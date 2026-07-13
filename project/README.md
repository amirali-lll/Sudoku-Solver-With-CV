# Sudoku Vision Project

This project builds an end-to-end Sudoku recognition and solving pipeline:

1. detect the Sudoku board in a real image
2. warp it to a top-down square view
3. split the grid into 81 cells
4. recognize digits with a CNN
5. solve the puzzle with backtracking
6. render the solution back onto the original image

## Why this structure

The assignment requires a reproducible project delivered as Python files, not a notebook. The code here is organized as a small package with scripts so each phase can be tested independently.

## Project layout

```text
project/
├── requirements.txt
├── README.md
├── scripts/
│   ├── run_pipeline.py
│   └── train_digits.py
└── src/
    └── sudoku_cv/
        ├── __init__.py
        ├── __main__.py
        ├── config.py
        ├── digits.py
        ├── grid.py
        ├── pipeline.py
        └── solver.py
```

## Setup

```bash
cd /Users/amirali/Programming/University/CV/project
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Training the digit model

The digit recognizer trains on a reproducibly shuffled combination of MNIST, HODA, and synthetic OpenCV-font digits. The synthetic samples add controlled variation in font scale, position, thickness, rotation, and noise. MNIST is downloaded automatically by `torchvision`; HODA must be downloaded/exported manually because HODA distributions use several different file formats.

```bash
python scripts/train_digits.py \
    --hoda-path data/hoda/train.npz \
    --epochs 8 \
    --save-path outputs/digit_model.pt \
    --report-dir outputs/digit_report
```

The HODA file must contain image and label arrays under keys such as `images`/`labels`, `data`/`targets`, or `x`/`y`. Supported extensions are `.npz`, `.npy`, `.mat`, and pickle files. Images from both datasets are converted to 28x28 normalized float tensors with white digits on a black background. The combined array is shuffled before an 80%/10%/10% train/validation/test split.

The report directory contains `metrics.json` with loss, accuracy, precision, recall, F1, and support, PNG confusion matrices for validation and test sets, and `training_curves.png` showing training/validation loss and accuracy per epoch. Use `--synthetic-samples-per-class` to control the number of generated samples per digit.

### Evaluate the model on each dataset separately

To verify that the classifier works across all three sources, run:

```bash
python scripts/evaluate_digits.py \
    --model outputs/digit_model.pt \
    --hoda-path data/hoda/Data_hoda_full.mat \
    --synthetic-samples-per-class 2000 \
    --report-dir outputs/digit_evaluation
```

This uses the MNIST test split, all HODA samples, and generated synthetic samples. It reports separate accuracy/loss values and writes per-source classification reports and confusion matrices to `metrics.json`. It also writes `sample_predictions.png`, where green titles are correct predictions and red titles are errors.

## Running inference

```bash
python scripts/run_pipeline.py --image path/to/sudoku.jpg --model outputs/digit_model.pt --save-overlay outputs/solved.jpg
```

For testing and debugging, enable `--debug`. This creates a directory containing the
original image, contour preprocessing, detected contour, warped board, all 81 cell
crops, all 81 digit crops, the final overlay, and `run_metadata.json` with the
recognized board, confidence values, solved board, device, and perspective matrix.

```bash
python scripts/run_pipeline.py \
    --image path/to/sudoku.jpg \
    --model outputs/digit_model.pt \
    --save-overlay outputs/solved.jpg \
    --debug-dir outputs/debug/my-test \
    --debug
```

## What is already covered

- board detection with contour-based OpenCV preprocessing
- perspective correction
- cell segmentation
- digit classification with a small CNN
- empty-cell detection
- Sudoku backtracking solver
- overlay of the solved digits on the original image
- documentation for each step

## Next recommended step

Run the training script once to produce the first model weights, then test the full pipeline on a few real Sudoku images and iterate on the preprocessing thresholds.
