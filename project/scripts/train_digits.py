import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
import json
import matplotlib.pyplot as plt

from sudoku_cv.digits import (
    CombinedDigitDataset,
    DigitClassifier,
    SyntheticDigitDataset,
    evaluate_digit_model,
    train_digit_model,
)


LOGGER = logging.getLogger("train_digits")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Sudoku digit classifier.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-path", type=str, default="outputs/digit_model.pt")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--synthetic-samples-per-class", type=int, default=2000)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--hoda-path", type=str, required=True, help="Path to HODA .npz/.npy/.mat/.pkl export")
    parser.add_argument("--report-dir", type=str, default="outputs/digit_report")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    LOGGER.info(
        "Using device: %s (MPS available=%s, CUDA available=%s)",
        device,
        torch.backends.mps.is_available(),
        torch.cuda.is_available(),
    )
    LOGGER.info("Downloading/loading MNIST from %s", args.data_dir)
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    LOGGER.info("MNIST samples: %d", len(mnist))
    synthetic = SyntheticDigitDataset(
        samples_per_class=args.synthetic_samples_per_class,
        image_size=28,
        seed=args.seed,
    )
    LOGGER.info("Synthetic samples: %d (%d per digit)", len(synthetic), args.synthetic_samples_per_class)
    LOGGER.info("Loading HODA data from %s", args.hoda_path)
    dataset = CombinedDigitDataset(mnist, args.hoda_path, synthetic_dataset=synthetic, seed=args.seed)
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    LOGGER.info("Combined samples: %d", len(dataset))
    LOGGER.info("Split sizes: train=%d, validation=%d, test=%d", train_size, valid_size, test_size)
    train_set, valid_set, test_set = random_split(
        dataset, [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DigitClassifier(num_classes=10).to(device)
    LOGGER.info("Starting training for %d epochs with batch size %d", args.epochs, args.batch_size)
    history = train_digit_model(
        model,
        train_loader,
        epochs=args.epochs,
        device=device,
        valid_loader=valid_loader,
    )
    validation = evaluate_digit_model(model, valid_loader, device)
    test = evaluate_digit_model(model, test_loader, device)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "history": history}, save_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset_size": len(dataset), "splits": {"train": train_size, "valid": valid_size, "test": test_size}, "history": history, "validation": validation, "test": test}
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    epochs = range(1, args.epochs + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["loss"], marker="o", label="Training")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["accuracy"], marker="o", label="Training")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    axes[1].set_title("Training and validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(report_dir / "training_curves.png", dpi=150)
    plt.close(figure)
    for name, result in (("validation", validation), ("test", test)):
        plt.figure(figsize=(7, 6))
        plt.imshow(result["confusion_matrix"], cmap="Blues")
        plt.title(f"{name.title()} confusion matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.colorbar()
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.tight_layout()
        plt.savefig(report_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close()
    LOGGER.info("Saved model to %s", save_path)
    LOGGER.info("Saved training report, plots, and confusion matrices to %s", report_dir)


if __name__ == "__main__":
    main()
