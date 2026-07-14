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
from PIL import Image, ImageDraw, ImageFont

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
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _render_digit(digit: int, image_size: int, rng: np.random.Generator, font_paths: List[str] = None) -> np.ndarray:
    # Use a larger canvas initially to prevent clipping during rotation
    canvas_size = int(image_size * 2.5)
    img = Image.new('L', (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)

    # 50% chance for Persian/Arabic digit, 50% for English
    is_persian = rng.random() < 0.5
    text = chr(ord('۰') + digit) if is_persian else str(digit)

    font = None
    if font_paths:
        chosen_font_path = rng.choice(font_paths)
        font_size = int(rng.uniform(canvas_size * 0.4, canvas_size * 0.8))
        try:
            font = ImageFont.truetype(chosen_font_path, font_size)
        except IOError:
            pass
            
    if font is None:
        font = ImageFont.load_default()

    color = int(rng.integers(0, 40))

    # Calculate text bounding box to center it
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older versions of Pillow
        text_w, text_h = draw.textsize(text, font=font)

    x = (canvas_size - text_w) // 2 + rng.integers(-4, 5)
    y = (canvas_size - text_h) // 2 + rng.integers(-4, 5)

    draw.text((x, y), text, font=font, fill=color)

    canvas = np.array(img)

    # Affine Transformations (Rotation)
    angle = float(rng.uniform(-15, 15))
    matrix = cv2.getRotationMatrix2D((canvas_size / 2, canvas_size / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, matrix, (canvas_size, canvas_size), borderValue=255)

    # Morphological operations to simulate thick/thin prints
    if rng.random() < 0.6:
        k_size = int(rng.integers(2, 4))
        kernel = np.ones((k_size, k_size), np.uint8)
        if rng.random() < 0.5:
            # Erode white background -> thicker black text
            canvas = cv2.erode(canvas, kernel, iterations=1)
        else:
            # Dilate white background -> thinner black text
            canvas = cv2.dilate(canvas, kernel, iterations=1)

    # Noise injection
    if rng.random() < 0.7:
        noise = rng.normal(0, 15, size=(canvas_size, canvas_size)).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return canvas


class SyntheticDigitDataset(Dataset):
    def __init__(self, samples_per_class: int = 2000, image_size: int = 28, seed: int = 42, font_dir: str = "fonts"):
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        self.seed = seed
        self.length = samples_per_class * 10
        
        # Load font paths
        self.font_paths = []
        font_dir_path = Path(font_dir)
        if font_dir_path.exists():
            self.font_paths.extend([str(p) for p in font_dir_path.glob("**/*.ttf")])
            self.font_paths.extend([str(p) for p in font_dir_path.glob("**/*.TTF")])
            
        if not self.font_paths:
            LOGGER.warning(f"No .ttf fonts found in '{font_dir}'. Synthetic dataset will lack font diversity! Create a 'fonts' folder and add TTF files.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + index)
        digit = index // self.samples_per_class
        image = _render_digit(int(digit), self.image_size, rng, self.font_paths)
        image = _standardize_image(image, self.image_size)
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(int(digit), dtype=torch.long)
        return image, label


def _standardize_image(image: np.ndarray, image_size: int = 28) -> np.ndarray:
    """Convert a digit image to a white-on-black, [0, 1] 28x28 image preserving aspect ratio."""
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

    # Crop out the digit tightly, then pad it to a square to maintain aspect ratio
    mask = (image > 0.1).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(largest_contour)
        cropped = image[y:y + height, x:x + width]
        
        # Calculate padding to make it a perfect square
        max_side = max(width, height)
        pad_y = (max_side - height) // 2
        pad_x = (max_side - width) // 2
        
        # Add a baseline padding so the digit never touches the very edge
        base_pad = max(1, int(round(max_side * 0.15)))
        
        total_top = pad_y + base_pad
        total_bottom = (max_side - height - pad_y) + base_pad
        total_left = pad_x + base_pad
        total_right = (max_side - width - pad_x) + base_pad
        
        image = cv2.copyMakeBorder(
            cropped, total_top, total_bottom, total_left, total_right,
            cv2.BORDER_CONSTANT, value=0.0
        )

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
    if images.dtype == object:
        images = np.asarray([np.asarray(image) for image in images.reshape(-1)], dtype=object)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("HODA image and label counts do not match")
    return images, labels


class CombinedDigitDataset(Dataset):
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
                # Manual loading via __getitem__ equivalent to respect the transforms
                img_tensor, lbl_tensor = synthetic_dataset[index]
                # Revert to uint8 for storage format compatibility in CombinedDataset
                img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
                synthetic_images[index] = img_np
                synthetic_labels[index] = lbl_tensor.item()
                
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
        # We don't standardize again if it's already standardized in Synthetic,
        # but _standardize_image is idempotent, so it's safe.
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
    if gray.size == 0:
        return 0, 1.0
    foreground_ratio = float(np.count_nonzero(gray > 0)) / gray.size
    if foreground_ratio < empty_threshold:
        return 0, 1.0

    standardized = _standardize_image(cell_image)
    standardized_foreground_ratio = float(np.count_nonzero(standardized > 0.1)) / standardized.size
    if standardized_foreground_ratio < empty_threshold:
        return 0, 1.0

    tensor = torch.from_numpy(standardized).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, prediction = torch.max(probabilities, dim=0)
    return int(prediction.item()), float(confidence.item())