"""
app/vision/run_eval_only.py
────────────────────────────
Evaluation-only script.  Does NOT retrain.
Loads the existing best_model.pt, runs the held-out test set,
sample predictions for all 5 classes, and Grad-CAM test.

Run from project root:
    python -m app.vision.run_eval_only
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.vision.predictor import build_model, collect_predictions, compute_metrics, evaluate
from app.vision.preprocessing import get_transforms, _TransformSubset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEED         = settings.random_seed
REPORTS_DIR  = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = PROJECT_ROOT / settings.model_weights_path


# ── Reproduce the EXACT same stratified split used during training ────────────

def get_test_loader():
    root = PROJECT_ROOT / settings.train_dir
    full = datasets.ImageFolder(root=str(root), transform=None)
    class_names  = full.classes
    all_indices  = list(range(len(full)))
    all_labels   = [full.targets[i] for i in all_indices]

    # Same split as evaluate.py — seed identical
    idx_train, idx_temp = train_test_split(
        all_indices, test_size=0.30, random_state=SEED, stratify=all_labels
    )
    labels_temp = [full.targets[i] for i in idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=SEED, stratify=labels_temp
    )

    test_ds = _TransformSubset(Subset(full, idx_test), get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    return test_loader, class_names, len(idx_train), len(idx_val), len(idx_test)


# ── Save confusion matrix PNG ─────────────────────────────────────────────────

def save_cm_png(cm, class_names, path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — Test Set (Epoch 1 checkpoint)")
        plt.tight_layout()
        fig.savefig(path, dpi=150); plt.close(fig)
        log.info("Confusion matrix PNG → %s", path)
    except Exception as e:
        log.warning("Could not save CM PNG: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Checkpoint ───────────────────────────────────────────────────────────
    if not WEIGHTS_PATH.exists():
        log.error("Checkpoint not found: %s", WEIGHTS_PATH)
        sys.exit(1)

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    ck_epoch   = checkpoint["epoch"]
    ck_val_acc = checkpoint["val_accuracy"]
    class_names = checkpoint["class_names"]

    log.info("Checkpoint : %s", WEIGHTS_PATH.resolve())
    log.info("Epoch      : %d", ck_epoch)
    log.info("Val acc    : %.4f", ck_val_acc)
    log.info("Classes    : %s", class_names)

    # ── Build model ───────────────────────────────────────────────────────────
    device = torch.device("cpu")
    model  = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ── Test loader ───────────────────────────────────────────────────────────
    test_loader, _, n_train, n_val, n_test = get_test_loader()
    log.info("Split — Train: %d  Val: %d  Test: %d", n_train, n_val, n_test)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc_raw = evaluate(model, test_loader, criterion, device)
    all_labels, all_preds   = collect_predictions(model, test_loader, device)

    correct   = sum(l == p for l, p in zip(all_labels, all_preds))
    incorrect = len(all_labels) - correct

    # sklearn metrics — both weighted and macro
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score, precision_score, recall_score,
    )

    accuracy           = accuracy_score(all_labels, all_preds)
    precision_weighted = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall_weighted    = recall_score   (all_labels, all_preds, average="weighted", zero_division=0)
    f1_weighted        = f1_score       (all_labels, all_preds, average="weighted", zero_division=0)
    precision_macro    = precision_score(all_labels, all_preds, average="macro",    zero_division=0)
    recall_macro       = recall_score   (all_labels, all_preds, average="macro",    zero_division=0)
    f1_macro           = f1_score       (all_labels, all_preds, average="macro",    zero_division=0)
    cm                 = confusion_matrix(all_labels, all_preds).tolist()
    report             = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )

    log.info("=" * 70)
    log.info("TEST SET RESULTS  (checkpoint epoch=%d, val_acc=%.4f)", ck_epoch, ck_val_acc)
    log.info("=" * 70)
    log.info("  Total test images : %d", n_test)
    log.info("  Correct           : %d", correct)
    log.info("  Incorrect         : %d", incorrect)
    log.info("  Test loss         : %.4f", test_loss)
    log.info("─" * 70)
    log.info("  Accuracy          : %.4f", accuracy)
    log.info("  Precision weighted: %.4f", precision_weighted)
    log.info("  Recall    weighted: %.4f", recall_weighted)
    log.info("  F1        weighted: %.4f", f1_weighted)
    log.info("  Precision macro   : %.4f", precision_macro)
    log.info("  Recall    macro   : %.4f", recall_macro)
    log.info("  F1        macro   : %.4f", f1_macro)
    log.info("─" * 70)
    log.info("Per-class report:\n%s", report)
    log.info("Confusion matrix (rows=true, cols=pred):\n%s", np.array(cm))
    log.info("  Classes order: %s", class_names)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_out = {
        "checkpoint":          str(WEIGHTS_PATH),
        "checkpoint_epoch":    ck_epoch,
        "checkpoint_val_acc":  round(ck_val_acc, 4),
        "class_names":         class_names,
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "test_loss":           round(test_loss, 4),
        "correct":             correct,
        "incorrect":           incorrect,
        "accuracy":            round(accuracy,           4),
        "precision_weighted":  round(precision_weighted, 4),
        "recall_weighted":     round(recall_weighted,    4),
        "f1_weighted":         round(f1_weighted,        4),
        "precision_macro":     round(precision_macro,    4),
        "recall_macro":        round(recall_macro,       4),
        "f1_macro":            round(f1_macro,           4),
        "confusion_matrix":    cm,
        "classification_report": report,
    }
    mp = REPORTS_DIR / "test_metrics.json"
    mp.write_text(json.dumps(metrics_out, indent=2))
    log.info("Metrics saved → %s", mp)

    # ── Confusion matrix PNG ──────────────────────────────────────────────────
    save_cm_png(cm, class_names, REPORTS_DIR / "confusion_matrix_test.png")

    # ── Sample predictions — one real image per class ─────────────────────────
    log.info("=" * 70)
    log.info("SAMPLE PREDICTIONS (1 image per class from train/ folder)")
    log.info("=" * 70)

    from app.vision.predictor import DefectPredictor
    predictor = DefectPredictor()
    predictor.load_model(WEIGHTS_PATH)

    root = PROJECT_ROOT / settings.train_dir
    sample_results = []

    for cls in class_names:
        img_path = sorted((root / cls).glob("*.png"))[0]
        r = predictor.predict(img_path)
        entry = {
            "file":            img_path.name,
            "true_class":      cls,
            "predicted_class": r.predicted_class,
            "confidence":      round(r.confidence, 4),
            "decision":        r.decision,
            "correct":         r.predicted_class == cls,
            "probabilities":   {k: round(v, 4) for k, v in r.probabilities.items()},
        }
        sample_results.append(entry)

        log.info("  File       : %s", img_path.name)
        log.info("  True class : %s", cls)
        log.info("  Predicted  : %s", r.predicted_class)
        log.info("  Confidence : %.4f", r.confidence)
        log.info("  Decision   : %s", r.decision)
        log.info("  Probs      : %s", {k: round(v,4) for k,v in r.probabilities.items()})
        log.info("  Correct    : %s", r.predicted_class == cls)
        log.info("  " + "─" * 50)

    sp = REPORTS_DIR / "sample_predictions.json"
    sp.write_text(json.dumps(sample_results, indent=2))
    log.info("Sample predictions saved → %s", sp)

    # ── Grad-CAM test — one image per class ───────────────────────────────────
    log.info("=" * 70)
    log.info("GRAD-CAM TEST (1 image per class)")
    log.info("=" * 70)

    from app.vision.explainability import explain_prediction
    from app.vision.preprocessing import preprocess_single_image

    gradcam_dir = REPORTS_DIR / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)
    gradcam_summary = {}

    for cls in class_names:
        img_path = sorted((root / cls).glob("*.png"))[0]
        try:
            tensor = preprocess_single_image(img_path)
            out    = explain_prediction(
                model=model,
                tensor=tensor,
                output_dir=gradcam_dir,
                filename=f"gradcam_{cls}.png",
            )
            hmap = out["heatmap"]
            gradcam_summary[cls] = {
                "status":        "OK",
                "image":         img_path.name,
                "heatmap_shape": list(hmap.shape),
                "heatmap_min":   round(float(hmap.min()), 4),
                "heatmap_max":   round(float(hmap.max()), 4),
                "overlay_saved": out.get("overlay_path", "not saved"),
            }
            log.info(
                "  [%s] heatmap=%s min=%.4f max=%.4f  overlay→%s",
                cls, hmap.shape, hmap.min(), hmap.max(),
                out.get("overlay_path", "not saved"),
            )
        except Exception as exc:
            gradcam_summary[cls] = {"status": "FAILED", "error": str(exc)}
            log.warning("  [%s] Grad-CAM FAILED: %s", cls, exc)

    gp = REPORTS_DIR / "gradcam_summary.json"
    gp.write_text(json.dumps(gradcam_summary, indent=2))
    log.info("Grad-CAM summary saved → %s", gp)

    # ── Final file listing ────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("FILES WRITTEN:")
    for f in sorted(REPORTS_DIR.rglob("*")):
        if f.is_file():
            log.info("  %s  (%.1f KB)", f.relative_to(PROJECT_ROOT), f.stat().st_size/1024)
    log.info("=" * 70)
    log.info("Evaluation complete. No retraining was performed.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
