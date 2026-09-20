"""
app/vision/evaluate.py
───────────────────────
Full training + held-out test evaluation for ManuSense Vision MVP.

Differences from train.py
──────────────────────────
train.py  — 80/20 train/val split, saves best checkpoint, reports val metrics.
evaluate.py — stratified 70/15/15 train/val/test split; trains to completion;
              reloads best checkpoint; evaluates on the HELD-OUT test set;
              runs sample predictions + Grad-CAM test; saves all artefacts.

Run from the project root:
    python -m app.vision.evaluate

Nothing in the original dataset is modified.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── Ensure project root is on sys.path ───────────────────────────────────────
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
from app.vision.preprocessing import get_transforms, _TransformSubset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Output directories ────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = settings.random_seed
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── 1. Stratified three-way split ─────────────────────────────────────────────

def build_stratified_splits(root: Path):
    """
    Build stratified train / val / test splits.

    Strategy
    --------
    - Load ImageFolder with no transform (raw PIL images).
    - Extract per-sample class labels for stratification.
    - Use sklearn train_test_split twice:
        first  → 70 % train  |  30 % temp
        second → 15 % val   |  15 % test  (50/50 of the 30 %)
    - Wrap each Subset in _TransformSubset with the correct pipeline.

    Returns
    -------
    train_ds, val_ds, test_ds, class_names
    """
    full = datasets.ImageFolder(root=str(root), transform=None)
    class_names = full.classes
    all_indices = list(range(len(full)))
    all_labels  = [full.targets[i] for i in all_indices]

    # 70 % train, 30 % temp  — stratified
    idx_train, idx_temp = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=SEED,
        stratify=all_labels,
    )

    # Split temp 50/50 → 15 % val, 15 % test
    labels_temp = [full.targets[i] for i in idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=labels_temp,
    )

    train_ds = _TransformSubset(Subset(full, idx_train), get_transforms(train=True))
    val_ds   = _TransformSubset(Subset(full, idx_val),   get_transforms(train=False))
    test_ds  = _TransformSubset(Subset(full, idx_test),  get_transforms(train=False))

    return train_ds, val_ds, test_ds, class_names


# ── 2. Confusion matrix plot ──────────────────────────────────────────────────

def save_confusion_matrix(cm, class_names, path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            np.array(cm), annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info("Saved → %s", path)
    except Exception as exc:
        log.warning("Could not save confusion matrix: %s", exc)


# ── 3. Sample predictions ─────────────────────────────────────────────────────

def run_sample_predictions(
    class_names: list[str],
    weights_path: Path,
    n_per_class: int = 2,
) -> list[dict]:
    """
    Run predict() on n_per_class real images from every class folder.
    Returns a list of result dicts.
    """
    from app.vision.predictor import DefectPredictor

    predictor = DefectPredictor()
    predictor.load_model(weights_path)

    results = []
    root = PROJECT_ROOT / settings.train_dir

    for cls in class_names:
        folder = root / cls
        images = sorted(folder.glob("*.png"))[:n_per_class]
        for img_path in images:
            try:
                r = predictor.predict(img_path)
                results.append({
                    "file":            img_path.name,
                    "true_class":      cls,
                    "predicted_class": r.predicted_class,
                    "confidence":      round(r.confidence, 4),
                    "decision":        r.decision,
                    "probabilities":   {k: round(v, 4) for k, v in r.probabilities.items()},
                    "correct":         r.predicted_class == cls,
                })
            except Exception as exc:
                results.append({"file": img_path.name, "true_class": cls, "error": str(exc)})

    return results


# ── 4. Grad-CAM test ──────────────────────────────────────────────────────────

def run_gradcam_test(class_names: list[str], weights_path: Path) -> dict:
    """
    Run Grad-CAM on one image from each class; save overlays to reports/.
    Returns a summary dict.
    """
    from app.vision.explainability import explain_prediction
    from app.vision.preprocessing import preprocess_single_image

    # Load model
    device = torch.device("cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    root = PROJECT_ROOT / settings.train_dir
    summary = {}

    for cls in class_names:
        folder = root / cls
        img_path = sorted(folder.glob("*.png"))[0]
        try:
            tensor = preprocess_single_image(img_path)
            out = explain_prediction(
                model=model,
                tensor=tensor,
                output_dir=REPORTS_DIR / "gradcam",
                filename=f"gradcam_{cls}.png",
            )
            hmap = out["heatmap"]
            summary[cls] = {
                "status":        "OK",
                "heatmap_shape": list(hmap.shape),
                "heatmap_min":   round(float(hmap.min()), 4),
                "heatmap_max":   round(float(hmap.max()), 4),
                "overlay_saved": out.get("overlay_path", "not saved"),
            }
        except Exception as exc:
            summary[cls] = {"status": "FAILED", "error": str(exc)}

    return summary


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("ManuSense Vision MVP — Training + Evaluation")
    log.info("=" * 70)

    # ── Dataset ──────────────────────────────────────────────────────────────
    root = PROJECT_ROOT / settings.train_dir
    log.info("Dataset root: %s", root.resolve())

    train_ds, val_ds, test_ds, class_names = build_stratified_splits(root)
    log.info("Classes: %s", class_names)
    log.info("Train: %d  |  Val: %d  |  Test: %d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=settings.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=settings.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=settings.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = build_model(num_classes=len(class_names), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    weights_path = PROJECT_ROOT / settings.model_weights_path
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch   = 0
    history      = []

    log.info("Starting training — %d epochs", settings.num_epochs)
    log.info("─" * 70)
    t0 = time.time()

    for epoch in range(1, settings.num_epochs + 1):
        t_ep = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t_ep

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy":     val_acc,
                "class_names":      class_names,
            }, weights_path)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss":   round(val_loss,   4), "val_acc":   round(val_acc,   4),
        })

        log.info(
            "Epoch %02d/%02d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | %.0fs%s",
            epoch, settings.num_epochs,
            train_loss, train_acc, val_loss, val_acc, elapsed,
            " ← best" if is_best else "",
        )

    total_time = time.time() - t0
    log.info("─" * 70)
    log.info("Training complete in %.1f s (%.1f min)", total_time, total_time / 60)
    log.info("Best val accuracy: %.4f at epoch %d", best_val_acc, best_epoch)

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info("─" * 70)
    log.info("Evaluating best checkpoint on HELD-OUT TEST SET …")

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    all_labels, all_preds = collect_predictions(model, test_loader, device)
    test_metrics = compute_metrics(all_labels, all_preds, class_names)

    log.info("─" * 70)
    log.info("TEST SET RESULTS:")
    log.info("  Loss      : %.4f", test_loss)
    log.info("  Accuracy  : %.4f", test_metrics["accuracy"])
    log.info("  Precision : %.4f (weighted)", test_metrics["precision"])
    log.info("  Recall    : %.4f (weighted)", test_metrics["recall"])
    log.info("  F1        : %.4f (weighted)", test_metrics["f1"])
    log.info("─" * 70)
    log.info("Per-class report:\n%s", test_metrics["classification_report"])
    log.info("Confusion matrix:\n%s", np.array(test_metrics["confusion_matrix"]))

    # ── Save artefacts ────────────────────────────────────────────────────────
    # Training history JSON
    history_path = REPORTS_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    log.info("Training history saved → %s", history_path)

    # Test metrics JSON
    metrics_out = {
        "test_loss":      round(test_loss, 4),
        "accuracy":       round(test_metrics["accuracy"],  4),
        "precision":      round(test_metrics["precision"], 4),
        "recall":         round(test_metrics["recall"],    4),
        "f1":             round(test_metrics["f1"],        4),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
        "train_count":    len(train_ds),
        "val_count":      len(val_ds),
        "test_count":     len(test_ds),
        "best_epoch":     best_epoch,
        "best_val_acc":   round(best_val_acc, 4),
        "training_time_seconds": round(total_time, 1),
        "class_names":    class_names,
    }
    metrics_path = REPORTS_DIR / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    log.info("Test metrics saved → %s", metrics_path)

    # Confusion matrix PNG
    save_confusion_matrix(
        test_metrics["confusion_matrix"], class_names,
        REPORTS_DIR / "confusion_matrix_test.png",
        "Confusion Matrix — Test Set",
    )

    # ── Sample predictions ────────────────────────────────────────────────────
    log.info("─" * 70)
    log.info("Running sample predictions (2 images per class) …")
    sample_preds = run_sample_predictions(class_names, weights_path, n_per_class=2)

    log.info("Sample prediction results:")
    for p in sample_preds:
        if "error" in p:
            log.error("  %s [%s] → ERROR: %s", p["file"], p["true_class"], p["error"])
        else:
            correct_marker = "✓" if p["correct"] else "✗"
            log.info(
                "  [%s] %s | true=%-8s pred=%-8s conf=%.4f decision=%s",
                correct_marker, p["file"], p["true_class"],
                p["predicted_class"], p["confidence"], p["decision"],
            )

    preds_path = REPORTS_DIR / "sample_predictions.json"
    preds_path.write_text(json.dumps(sample_preds, indent=2))
    log.info("Sample predictions saved → %s", preds_path)

    # ── Grad-CAM test ─────────────────────────────────────────────────────────
    log.info("─" * 70)
    log.info("Running Grad-CAM test (1 image per class) …")
    gradcam_summary = run_gradcam_test(class_names, weights_path)
    for cls, result in gradcam_summary.items():
        if result["status"] == "OK":
            log.info(
                "  Grad-CAM [%s]: heatmap=%s  min=%.4f  max=%.4f  saved=%s",
                cls, result["heatmap_shape"],
                result["heatmap_min"], result["heatmap_max"],
                result["overlay_saved"],
            )
        else:
            log.warning("  Grad-CAM [%s]: FAILED — %s", cls, result.get("error"))

    gradcam_path = REPORTS_DIR / "gradcam_summary.json"
    gradcam_path.write_text(json.dumps(gradcam_summary, indent=2))
    log.info("Grad-CAM summary saved → %s", gradcam_path)

    log.info("=" * 70)
    log.info("All done. Reports in: %s", REPORTS_DIR.resolve())
    log.info("=" * 70)


if __name__ == "__main__":
    main()
