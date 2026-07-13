from __future__ import annotations

from pathlib import Path
import logging
from typing import Dict, List, Tuple
import pickle

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)


class DigitClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _render_digit(digit: int, image_size: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full((image_size, image_size), 255, dtype=np.uint8)
    if digit == 0:
        noise = rng.integers(0, 15, size=(image_size, image_size), dtype=np.uint8)
        canvas = np.clip(canvas - noise, 0, 255).astype(np.uint8)
        return canvas

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = rng.uniform(0.8, 1.3)
    thickness = int(rng.integers(1, 3))
    text = str(digit)
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]

    x = int((image_size - text_size[0]) / 2 + rng.integers(-3, 4))
    y = int((image_size + text_size[1]) / 2 + rng.integers(-3, 4))
    color = int(rng.integers(0, 35))

    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

    angle = float(rng.uniform(-20, 20))
    matrix = cv2.getRotationMatrix2D((image_size / 2, image_size / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, matrix, (image_size, image_size), borderValue=255)

    if rng.random() < 0.7:
        noise = rng.normal(0, 10, size=(image_size, image_size)).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return canvas


class SyntheticDigitDataset(Dataset):
    def __init__(self, samples_per_class: int = 2000, image_size: int = 28, seed: int = 42):
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        self.seed = seed
        self.length = samples_per_class * 10

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + index)
        digit = index // self.samples_per_class
        image = _render_digit(int(digit), self.image_size, rng)
        image = _standardize_image(image, self.image_size)
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(int(digit), dtype=torch.long)
        return image, label


def _standardize_image(image: np.ndarray, image_size: int = 28) -> np.ndarray:
    """Convert a digit image to a white-on-black, [0, 1] 28x28 image."""
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., 0]
    image = image.astype(np.float32)
    if image.size == 0:
        return np.zeros((image_size, image_size), dtype=np.float32)
    image -= image.min()
    maximum = image.max()
    if maximum > 0:
        image /= maximum
    # HODA exports commonly use black ink on a white background, unlike MNIST.
    if float(image.mean()) > 0.5:
        image = 1.0 - image
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _extract_array(payload, names: Tuple[str, ...]):
    if isinstance(payload, dict):
        for name in names:
            if name in payload:
                return payload[name]
    if hasattr(payload, "files"):
        for name in names:
            if name in payload.files:
                return payload[name]
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
    return None


def load_hoda_arrays(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load HODA images and labels from an exported .npz, .mat, or pickle file.

    The loader accepts keys/fields named images/data/x and labels/targets/y.
    It deliberately does not assume one particular HODA distribution format.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        images = _extract_array(payload, ("images", "data", "Data", "x"))
        labels = _extract_array(payload, ("labels", "targets", "y"))
    elif path.suffix.lower() == ".npy":
        payload = np.load(path, allow_pickle=True).item()
        images = _extract_array(payload, ("images", "data", "Data", "x"))
        labels = _extract_array(payload, ("labels", "targets", "y"))
    elif path.suffix.lower() == ".mat":
        try:
            from importlib import import_module
            loadmat = import_module("scipy.io").loadmat
        except ImportError as error:
            raise ImportError("Reading .mat HODA files requires scipy; use .npz or install scipy") from error
        payload = loadmat(path)
        images = _extract_array(payload, ("images", "data", "Data", "x"))
        labels = _extract_array(payload, ("labels", "targets", "y"))
    else:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        images = _extract_array(payload, ("images", "data", "Data", "x"))
        labels = _extract_array(payload, ("labels", "targets", "y"))
    if images is None or labels is None:
        raise ValueError(f"Could not find image/label arrays in {path}")
    images = np.asarray(images)
    # The original HODA MAT file stores each variable-size image in a MATLAB
    # cell array named Data. Keep it object-based because HODA images have
    # different heights and widths; they are resized in __getitem__.
    if images.dtype == object:
        images = np.asarray([np.asarray(image) for image in images.reshape(-1)], dtype=object)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("HODA image and label counts do not match")
    return images, labels


class CombinedDigitDataset(Dataset):
    """A shuffled, normalized combination of MNIST, HODA, and synthetic digits."""

    def __init__(
        self,
        mnist_dataset,
        hoda_path: str | Path,
        synthetic_dataset: SyntheticDigitDataset | None = None,
        seed: int = 42,
    ):
        hoda_images, hoda_labels = load_hoda_arrays(hoda_path)
        if hasattr(mnist_dataset, "data") and hasattr(mnist_dataset, "targets"):
            mnist_images = np.asarray(mnist_dataset.data)
            mnist_labels = np.asarray(mnist_dataset.targets)
        else:
            mnist_samples = [mnist_dataset[index] for index in range(len(mnist_dataset))]
            mnist_images = np.stack([
                sample[0].squeeze().numpy() * 255.0 for sample in mnist_samples
            ])
            mnist_labels = np.asarray([sample[1] for sample in mnist_samples])
        image_arrays = [list(mnist_images), list(hoda_images)]
        label_arrays = [mnist_labels, hoda_labels]
        if synthetic_dataset is not None:
            synthetic_images = np.empty(
                (len(synthetic_dataset), synthetic_dataset.image_size, synthetic_dataset.image_size),
                dtype=np.uint8,
            )
            synthetic_labels = np.empty(len(synthetic_dataset), dtype=np.int64)
            for index in range(len(synthetic_dataset)):
                rng = np.random.default_rng(synthetic_dataset.seed + index)
                digit = index // synthetic_dataset.samples_per_class
                synthetic_images[index] = _render_digit(digit, synthetic_dataset.image_size, rng)
                synthetic_labels[index] = digit
            image_arrays.append(list(synthetic_images))
            label_arrays.append(synthetic_labels)
        images = np.asarray(
            [image for image_array in image_arrays for image in image_array],
            dtype=object,
        )
        labels = np.concatenate(label_arrays, axis=0)
        if np.any((labels < 0) | (labels > 9)):
            raise ValueError("Digit labels must be integers in the range 0..9")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(labels))
        self.images = images[order]
        self.labels = labels[order]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = _standardize_image(self.images[index])
        return torch.from_numpy(image).unsqueeze(0), torch.tensor(int(self.labels[index]), dtype=torch.long)


def train_digit_model(model: nn.Module, data_loader, epochs: int, device: torch.device, valid_loader=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: Dict[str, List[float]] = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        batches = len(data_loader)
        for batch_index, (images, labels) in enumerate(data_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if batch_index == 1 or batch_index == batches or batch_index % max(batches // 4, 1) == 0:
                LOGGER.info(
                    "Epoch %d/%d | batch %d/%d | loss %.4f",
                    epoch + 1,
                    epochs,
                    batch_index,
                    batches,
                    loss.item(),
                )

        epoch_loss = running_loss / max(total, 1)
        epoch_accuracy = correct / max(total, 1)
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_accuracy)
        message = f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}"
        if valid_loader is not None:
            validation = evaluate_digit_model(model, valid_loader, device)
            history["val_loss"].append(float(validation["loss"]))
            history["val_accuracy"].append(float(validation["accuracy"]))
            message += f" - val_loss: {validation['loss']:.4f} - val_acc: {validation['accuracy']:.4f}"
            model.train()
        print(message)

    return history


def evaluate_digit_model(model: nn.Module, data_loader, device: torch.device) -> Dict[str, object]:
    """Return loss, accuracy, confusion matrix, and a per-class report."""
    criterion = nn.CrossEntropyLoss()
    matrix = np.zeros((10, 10), dtype=np.int64)
    total_loss = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
            actual = labels.numpy()
            total_loss += loss.item() * len(labels)
            total += len(labels)
            for truth, prediction in zip(actual, predictions):
                matrix[int(truth), int(prediction)] += 1
    report = {}
    for digit in range(10):
        tp = matrix[digit, digit]
        precision = tp / max(matrix[:, digit].sum(), 1)
        recall = tp / max(matrix[digit, :].sum(), 1)
        report[str(digit)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "support": int(matrix[digit, :].sum()),
        }
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(np.trace(matrix) / max(total, 1)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def load_digit_model(model_path: str, device: torch.device) -> DigitClassifier:
    checkpoint = torch.load(model_path, map_location=device)
    model = DigitClassifier(num_classes=10).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def classify_digit(model: nn.Module, cell_image: np.ndarray, device: torch.device, empty_threshold: float = 0.02) -> Tuple[int, float]:
    gray = cell_image.astype(np.float32)
    foreground_ratio = float(np.count_nonzero(gray > 0)) / gray.size
    if foreground_ratio < empty_threshold:
        return 0, 1.0

    tensor = torch.from_numpy(gray / 255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, prediction = torch.max(probabilities, dim=0)
    return int(prediction.item()), float(confidence.item())
