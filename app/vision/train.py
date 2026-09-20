"""
app/vision/train.py
────────────────────
Standalone training script for the ManuSense defect classifier.

Run from the project root:
    python -m app.vision.train

What it does
────────────
1.  Builds train / validation DataLoaders from train/
2.  Instantiates a pretrained ResNet18 with 5-class output head
3.  Trains for NUM_EPOCHS, printing loss and accuracy each epoch
4.  Saves the best checkpoint (highest val accuracy) to models/best_model.pt
5.  Evaluates the best model on the validation set and prints metrics
6.  Saves the confusion matrix plot to data/processed/confusion_matrix.png

All hyperparameters come from .env / app/config.py — nothing is hardcoded here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Make sure the project root is on the path when run as __main__ ────────────
# This lets  python -m app.vision.train  work from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.vision.predictor import (
    build_model,
    collect_predictions,
    compute_metrics,
    evaluate,
    train_one_epoch,
)
from app.vision.preprocessing import build_dataloaders

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def save_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Render and save the confusion matrix as a PNG."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            np.array(cm),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — Validation Set")
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        log.info("Confusion matrix saved → %s", output_path)
    except Exception as exc:
        log.warning("Could not save confusion matrix plot: %s", exc)


def train() -> None:
    """Full training loop with best-model checkpointing."""

    # ── Seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(settings.random_seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    log.info("Loading dataset from: %s", settings.train_dir.resolve())
    train_loader, val_loader, class_names = build_dataloaders()
    log.info(
        "Classes (%d): %s", len(class_names), class_names
    )
    log.info(
        "Train batches: %d  |  Val batches: %d",
        len(train_loader), len(val_loader),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = build_model(num_classes=len(class_names), pretrained=True)
    model.to(device)

    # ── Optimiser & loss ──────────────────────────────────────────────────────
    # CrossEntropyLoss is standard for multi-class classification.
    # Classes are balanced (2400 each) so no class-weighting is needed.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    # Reduce LR by 0.1 when val loss plateaus for 3 epochs — helps squeeze
    # out extra accuracy without manual LR scheduling.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_epoch    = 0
    weights_path  = settings.model_weights_path
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Starting training for %d epochs", settings.num_epochs)
    log.info("─" * 70)

    for epoch in range(1, settings.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        log.info(
            "Epoch %02d/%02d  |  "
            "train_loss=%.4f  train_acc=%.4f  |  "
            "val_loss=%.4f  val_acc=%.4f%s",
            epoch, settings.num_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            "  ← best" if val_acc > best_val_acc else "",
        )

        # Save checkpoint when val accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy":     val_acc,
                    "class_names":      class_names,
                },
                weights_path,
            )

    log.info("─" * 70)
    log.info(
        "Training complete. Best val accuracy: %.4f at epoch %d",
        best_val_acc, best_epoch,
    )
    log.info("Best model saved → %s", weights_path.resolve())

    # ── Final evaluation ──────────────────────────────────────────────────────
    log.info("Running final evaluation on validation set …")

    # Reload the best checkpoint for evaluation
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    all_labels, all_preds = collect_predictions(model, val_loader, device)
    metrics = compute_metrics(all_labels, all_preds, class_names)

    log.info("─" * 70)
    log.info("Validation metrics (best checkpoint):")
    log.info("  Accuracy  : %.4f", metrics["accuracy"])
    log.info("  Precision : %.4f (weighted)", metrics["precision"])
    log.info("  Recall    : %.4f (weighted)", metrics["recall"])
    log.info("  F1        : %.4f (weighted)", metrics["f1"])
    log.info("─" * 70)
    log.info("Classification report:\n%s", metrics["classification_report"])

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm_path = Path(settings.processed_dir) / "confusion_matrix.png"
    save_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path)


if __name__ == "__main__":
    train()
