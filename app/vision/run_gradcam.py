"""
app/vision/run_gradcam.py
─────────────────────────
Standalone Grad-CAM test. Loads the existing best_model.pt,
runs Grad-CAM on one image per class, saves overlay PNGs to
reports/gradcam/, and updates reports/gradcam_summary.json.

Does NOT retrain. Does NOT modify the dataset.

Run from project root:
    python -m app.vision.run_gradcam
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.vision.predictor import build_model
from app.vision.preprocessing import preprocess_single_image
from app.vision.explainability import explain_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CKPT_PATH   = PROJECT_ROOT / settings.model_weights_path
REPORTS_DIR = PROJECT_ROOT / "reports"
GRADCAM_DIR = REPORTS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not CKPT_PATH.exists():
        log.error("Checkpoint not found: %s", CKPT_PATH)
        sys.exit(1)

    checkpoint  = torch.load(CKPT_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    ck_epoch    = checkpoint["epoch"]
    ck_val_acc  = checkpoint["val_accuracy"]

    log.info("Checkpoint : %s", CKPT_PATH.resolve())
    log.info("Epoch      : %d  |  Val acc: %.4f", ck_epoch, ck_val_acc)
    log.info("Classes    : %s", class_names)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Run Grad-CAM for one image per class ──────────────────────────────────
    train_root = PROJECT_ROOT / settings.train_dir
    summary    = {}

    log.info("=" * 60)
    log.info("Grad-CAM test — 1 image per class")
    log.info("=" * 60)

    for cls in class_names:
        img_path = sorted((train_root / cls).glob("*.png"))[0]
        log.info("Class: %-10s  Image: %s", cls, img_path.name)

        try:
            tensor = preprocess_single_image(img_path)
            result = explain_prediction(
                model      = model,
                tensor     = tensor,
                output_dir = GRADCAM_DIR,
                filename   = f"gradcam_{cls}.png",
            )
            hmap = result["heatmap"]
            saved_path = result.get("overlay_path", "not saved")

            log.info(
                "  heatmap shape=%-12s  min=%.4f  max=%.4f",
                str(hmap.shape), float(hmap.min()), float(hmap.max()),
            )
            log.info("  overlay saved → %s", saved_path)

            summary[cls] = {
                "status":        "OK",
                "checkpoint":    str(CKPT_PATH),
                "epoch":         ck_epoch,
                "image":         img_path.name,
                "heatmap_shape": list(hmap.shape),
                "heatmap_min":   round(float(hmap.min()), 4),
                "heatmap_max":   round(float(hmap.max()), 4),
                "overlay_saved": saved_path,
                "note": (
                    "Grad-CAM is a model explanation showing which image regions "
                    "influenced the prediction. It is NOT ground-truth defect "
                    "localization — this dataset has no bounding-box annotations."
                ),
            }

        except Exception as exc:
            log.warning("  FAILED: %s", exc)
            summary[cls] = {"status": "FAILED", "error": str(exc)}

    # ── Save summary ──────────────────────────────────────────────────────────
    out_path = REPORTS_DIR / "gradcam_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("=" * 60)
    log.info("Grad-CAM summary saved → %s", out_path)

    # ── List overlay files ────────────────────────────────────────────────────
    overlays = sorted(GRADCAM_DIR.glob("*.png"))
    log.info("Overlay files written (%d):", len(overlays))
    for f in overlays:
        log.info("  %s  (%.1f KB)", f.name, f.stat().st_size / 1024)

    log.info("=" * 60)
    log.info("Grad-CAM test complete. No retraining was performed.")


if __name__ == "__main__":
    main()
