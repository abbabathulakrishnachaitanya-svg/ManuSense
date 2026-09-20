"""
app/vision/predictor.py
────────────────────────
Defect classification model, training loop, evaluation, and inference.

Design decisions
────────────────
1.  Architecture choice — ResNet18
    Lightweight enough for a CPU-only hackathon laptop (~11 M params),
    widely supported, and well-understood.  The final fully-connected layer
    is replaced to match our 5-class output.

2.  Grayscale adaptation
    The first convolutional layer (conv1) still expects 3-channel input.
    We do NOT change its weight tensor — instead, preprocessing.py repeats
    the grayscale channel three times so pre-trained ImageNet weights are
    reused without modification.

3.  Transfer learning strategy
    All layers are fine-tuned (not frozen).  With only 12 000 images and
    a 24-hour deadline, full fine-tuning on a small ResNet18 converges
    quickly and typically outperforms frozen-backbone + head-only training
    on visually dissimilar domains (industrial textures vs. ImageNet photos).

4.  Best-model checkpointing
    Only the epoch with the highest validation accuracy is saved.  This
    prevents overfitting artefacts from being used at inference time.

5.  Decision logic
    Three states — ACCEPT / REJECT / REVIEW — are derived from two
    configurable thresholds.  Thresholds are NOT claimed to be optimal;
    they are tunable via .env after reviewing the confusion matrix.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

from app.config import settings

log = logging.getLogger(__name__)

# Label used for "good part" — must match the folder name in the dataset.
NORMAL_CLASS = "normal"


# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Structured inference output for a single image."""

    image_path: str
    predicted_class: str
    confidence: float                          # max softmax probability
    probabilities: dict[str, float]            # {class_name: probability}
    decision: str                              # ACCEPT | REJECT | REVIEW
    is_uncertain: bool = False
    raw_logits: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence":      round(self.confidence, 4),
            "probabilities":   {k: round(v, 4) for k, v in self.probabilities.items()},
            "decision":        self.decision,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(num_classes: int | None = None, pretrained: bool = True) -> nn.Module:
    """
    Build and return a ResNet18 with the final FC layer replaced.

    Args:
        num_classes: Number of output classes.  Defaults to settings.num_classes.
        pretrained:  If True, load ImageNet weights for all layers except the
                     new FC head (which is randomly initialised).

    Returns:
        torch.nn.Module ready for training or inference.
    """
    n = num_classes or settings.num_classes

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the final classifier: in_features=512 for ResNet18
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n)

    return model


# ── Main predictor class ──────────────────────────────────────────────────────

class DefectPredictor:
    """
    Wraps the trained ResNet18 and exposes predict() for single-image inference.

    Usage:
        predictor = DefectPredictor(class_names=["crack","hole","normal","rust","scratch"])
        predictor.load_model()
        result = predictor.predict("path/to/image.png")
        print(result)
    """

    def __init__(self, class_names: list[str] | None = None) -> None:
        # class_names are set here or loaded from the checkpoint.
        self._class_names: list[str] = class_names or []
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load_model(self, weights_path: str | Path | None = None) -> None:
        """
        Load a saved checkpoint into the model.

        The checkpoint dict stores both weights and class_names so the
        predictor is self-contained after loading.

        Args:
            weights_path: Path to the .pt checkpoint.
                          Defaults to settings.model_weights_path.

        Raises:
            FileNotFoundError: If the weights file does not exist.
        """
        path = Path(weights_path) if weights_path else settings.model_weights_path

        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {path.resolve()}.\n"
                "Run the training script first:  python -m app.vision.train"
            )

        checkpoint = torch.load(path, map_location=self._device)

        # Support both bare state-dict saves and our richer checkpoint format
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self._class_names = checkpoint.get("class_names", self._class_names)
            n_classes = len(self._class_names)
            self._model = build_model(num_classes=n_classes, pretrained=False)
            self._model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Bare state dict — class_names must have been provided at init
            n_classes = len(self._class_names) or settings.num_classes
            self._model = build_model(num_classes=n_classes, pretrained=False)
            self._model.load_state_dict(checkpoint)

        self._model.to(self._device)
        self._model.eval()
        log.info("Model loaded from %s (device=%s)", path, self._device)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, image_path: str | Path) -> PredictionResult:
        """
        Classify a single image and return a structured result with a decision.

        Args:
            image_path: Path to the inspection image.

        Returns:
            :class:`PredictionResult`

        Raises:
            RuntimeError:        If the model has not been loaded.
            FileNotFoundError:   If the image does not exist.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first.")

        # ── Pre-process ──────────────────────────────────────────────────────
        from app.vision.preprocessing import preprocess_single_image
        tensor = preprocess_single_image(image_path).to(self._device)

        # ── Forward pass ─────────────────────────────────────────────────────
        with torch.no_grad():
            logits = self._model(tensor)                # (1, num_classes)
            probs  = torch.softmax(logits, dim=1)[0]    # (num_classes,)

        probs_list = probs.cpu().tolist()
        confidence, pred_idx = probs.max(dim=0)
        confidence  = float(confidence)
        pred_idx    = int(pred_idx)

        predicted_class = self._class_names[pred_idx]
        probabilities   = {
            cls: float(p)
            for cls, p in zip(self._class_names, probs_list)
        }

        decision = self._make_decision(predicted_class, confidence)

        return PredictionResult(
            image_path      = str(image_path),
            predicted_class = predicted_class,
            confidence      = confidence,
            probabilities   = probabilities,
            decision        = decision,
            is_uncertain    = confidence < settings.uncertainty_threshold,
            raw_logits      = logits.cpu().squeeze().tolist(),
        )

    # ── Decision logic ───────────────────────────────────────────────────────

    @staticmethod
    def _make_decision(predicted_class: str, confidence: float) -> str:
        """
        Map (predicted_class, confidence) to one of three advisory states.

        Thresholds come from settings (configurable via .env):
            confidence_threshold  — high-confidence boundary
            uncertainty_threshold — low-confidence boundary

        States:
            ACCEPT  — model is confident the part is normal
            REJECT  — model is confident a defect is present
            REVIEW  — model is uncertain; a human should inspect

        These are advisory only — no machine is controlled.
        """
        high = settings.confidence_threshold
        low  = settings.uncertainty_threshold

        if confidence < low:
            # Below the uncertainty floor — we don't trust any prediction
            return "REVIEW"

        if predicted_class == NORMAL_CLASS:
            if confidence >= high:
                return "ACCEPT"
            else:
                # Normal prediction but not confident enough — safer to review
                return "REVIEW"
        else:
            # Any defect class prediction triggers a reject or review
            if confidence >= high:
                return "REJECT"
            else:
                return "REVIEW"


# ── Training utilities ────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        (average_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on a data loader.

    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """
    Collect all ground-truth labels and model predictions from a loader.

    Returns:
        (all_labels, all_predictions) as flat Python lists.
    """
    model.eval()
    all_labels  = []
    all_preds   = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())

    return all_labels, all_preds


def compute_metrics(
    labels: list[int],
    preds: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """
    Compute accuracy, per-class precision/recall/F1, and confusion matrix.

    Uses scikit-learn so we get macro/weighted averages for free.

    Args:
        labels:       Ground-truth class indices.
        preds:        Predicted class indices.
        class_names:  Ordered list of class name strings.

    Returns:
        Dict with keys: accuracy, precision, recall, f1,
        confusion_matrix, classification_report.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    accuracy   = accuracy_score(labels, preds)
    precision  = precision_score(labels, preds, average="weighted", zero_division=0)
    recall     = recall_score(labels, preds, average="weighted", zero_division=0)
    f1         = f1_score(labels, preds, average="weighted", zero_division=0)
    cm         = confusion_matrix(labels, preds)
    report     = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    return {
        "accuracy":               accuracy,
        "precision":              precision,
        "recall":                 recall,
        "f1":                     f1,
        "confusion_matrix":       cm.tolist(),
        "classification_report":  report,
    }
