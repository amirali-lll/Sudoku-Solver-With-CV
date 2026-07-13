"""Evaluate a trained digit model separately on MNIST, HODA, and synthetic digits."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sudoku_cv.digits import (  # noqa: E402
    DigitClassifier,
    SyntheticDigitDataset,
    _standardize_image,
    evaluate_digit_model,
    load_hoda_arrays,
    load_digit_model,
)


class ArrayDigitDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = _standardize_image(self.images[index])
        return torch.from_numpy(image).unsqueeze(0), torch.tensor(int(self.labels[index]))


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def plot_samples(model, datasets_by_name, device, output_path, samples_per_source):
    rows = len(datasets_by_name)
    columns = samples_per_source
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2.5 * rows), squeeze=False)
    generator = np.random.default_rng(42)
    model.eval()
    with torch.no_grad():
        for row, (name, dataset) in enumerate(datasets_by_name.items()):
            count = min(samples_per_source, len(dataset))
            indices = generator.choice(len(dataset), size=count, replace=False)
            images, labels = zip(*(dataset[int(index)] for index in indices))
            logits = model(torch.stack(images).to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            for column in range(columns):
                axis = axes[row, column]
                axis.axis("off")
                if column >= count:
                    continue
                axis.imshow(images[column].squeeze(0), cmap="gray", vmin=0, vmax=1)
                correct = predictions[column] == int(labels[column])
                axis.set_title(
                    f"true={int(labels[column])}\npred={predictions[column]}",
                    color="green" if correct else "red",
                    fontsize=9,
                )
            axes[row, 0].set_ylabel(name, fontsize=11, rotation=90, labelpad=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained digit model on all three digit sources.")
    parser.add_argument("--model", default="outputs/digit_model.pt")
    parser.add_argument("--hoda-path", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic-samples-per-class", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--report-dir", default="outputs/digit_evaluation")
    parser.add_argument("--samples-per-source", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    mnist = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    hoda_images, hoda_labels = load_hoda_arrays(args.hoda_path)
    synthetic = SyntheticDigitDataset(args.synthetic_samples_per_class, seed=42)

    datasets_by_name = {
        "MNIST test": ArrayDigitDataset(mnist.data.numpy(), mnist.targets.numpy()),
        "HODA": ArrayDigitDataset(hoda_images, hoda_labels),
        "Synthetic": synthetic,
    }
    model = load_digit_model(args.model, device)
    metrics = {name: evaluate_digit_model(model, DataLoader(dataset, args.batch_size), device) for name, dataset in datasets_by_name.items()}
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_samples(model, datasets_by_name, device, report_dir / "sample_predictions.png", args.samples_per_source)

    print(f"Device: {device}")
    for name, result in metrics.items():
        print(f"{name}: accuracy={result['accuracy']:.4f}, loss={result['loss']:.4f}")
    print(f"Saved metrics to {report_dir / 'metrics.json'}")
    print(f"Saved prediction samples to {report_dir / 'sample_predictions.png'}")


if __name__ == "__main__":
    main()
