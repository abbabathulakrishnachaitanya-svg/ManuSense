"""
app/vision/explainability.py
─────────────────────────────
Grad-CAM heatmap generation for the ManuSense defect classifier.

Design decisions
────────────────
1.  Library choice: pytorch-grad-cam (grad-cam on PyPI)
    Actively maintained, architecture-agnostic, and works with ResNet18
    out of the box via the layer4 target layer.

2.  Target layer
    For ResNet18 the last convolutional block (model.layer4[-1]) produces
    the highest-resolution feature maps before global average pooling.
    This is the standard choice for Grad-CAM on ResNets.

3.  Grayscale overlay
    The input image arrives as a 3-channel tensor (grayscale repeated).
    The overlay is rendered in colour (jet colormap) so the heatmap is
    clearly distinguishable from the greyscale background.

4.  Separation from predictor.py
    Explainability is kept here so predictor.py stays import-friendly in
    environments where grad-cam is not yet installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def generate_gradcam(
    model: Any,
    tensor: Any,                  # (1, 3, H, W) float tensor on any device
    target_layer: Any | None = None,
) -> np.ndarray:
    """
    Produce a Grad-CAM heatmap for a single pre-processed image tensor.

    Args:
        model:        Loaded ResNet18 (torch.nn.Module) in eval mode.
        tensor:       Pre-processed image tensor, shape (1, 3, H, W).
        target_layer: Conv layer to hook.  Defaults to model.layer4[-1]
                      (standard for ResNet18).

    Returns:
        Heatmap as a numpy array of shape (H, W) with values in [0, 1].

    Raises:
        ImportError: If the grad-cam package is not installed.
        RuntimeError: If the model architecture is incompatible.
    """
    # The installed package (grad-cam==1.0.0) exposes pytorch_grad_cam.gradcam.CAM.
    # Its API differs from the modern grad-cam>=1.3 package:
    #   CAM(model, target_layer)  then  cam(input_tensor, method='gradcam')
    # We adapt to this version here.
    try:
        from pytorch_grad_cam.gradcam import CAM
    except ImportError as exc:
        raise ImportError(
            "grad-cam package not found.  Install it with:\n"
            "  pip install grad-cam"
        ) from exc

    layer = target_layer if target_layer is not None else model.layer4[-1]

    cam = CAM(model=model, target_layer=layer, use_cuda=False)
    # Returns numpy array of shape (H, W) with values in [0, 1]
    grayscale_cam = cam(input_tensor=tensor, method="gradcam")

    return grayscale_cam


def overlay_heatmap(
    original_tensor: Any,   # (1, 3, H, W) normalised tensor
    heatmap: np.ndarray,    # (H, W) float in [0, 1]
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend the original image and the Grad-CAM heatmap.

    The tensor is denormalised (ImageNet stats reversed) back to [0,1]
    before blending so the background is visible.

    Args:
        original_tensor: Pre-processed image tensor (1, 3, H, W).
        heatmap:         Grad-CAM output from :func:`generate_gradcam`.
        alpha:           Heatmap opacity (0 = invisible, 1 = heatmap only).

    Returns:
        Blended image as uint8 numpy array of shape (H, W, 3).
    """
    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise ImportError(
            "grad-cam package not found.  Install it with:\n"
            "  pip install grad-cam"
        ) from exc

    # Denormalise: reverse ImageNet normalisation
    _MEAN = np.array([0.485, 0.456, 0.406])
    _STD  = np.array([0.229, 0.224, 0.225])

    img = original_tensor.squeeze(0).cpu().numpy()   # (3, H, W)
    img = np.transpose(img, (1, 2, 0))               # (H, W, 3)
    img = img * _STD + _MEAN                         # denormalise
    img = np.clip(img, 0, 1).astype(np.float32)

    # v1.0.0 show_cam_on_image(img, mask) — no use_rgb or image_weight args
    overlay = show_cam_on_image(img, heatmap)
    return overlay                                    # (H, W, 3) uint8


def explain_prediction(
    model: Any,
    tensor: Any,
    output_dir: str | Path | None = None,
    filename: str = "gradcam_overlay.png",
) -> dict[str, Any]:
    """
    High-level helper: generate heatmap, create overlay, and optionally save.

    Args:
        model:      Loaded ResNet18 in eval mode.
        tensor:     Pre-processed image tensor (1, 3, H, W).
        output_dir: If provided, save the overlay PNG here.
        filename:   Output file name.

    Returns:
        Dict with keys:
            ``heatmap``      — numpy (H, W) float array
            ``overlay``      — numpy (H, W, 3) uint8 array
            ``overlay_path`` — saved path string (only if output_dir given)
    """
    heatmap = generate_gradcam(model, tensor)
    overlay = overlay_heatmap(tensor, heatmap)

    result: dict[str, Any] = {"heatmap": heatmap, "overlay": overlay}

    if output_dir is not None:
        try:
            from PIL import Image as PILImage

            out_path = Path(output_dir) / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(overlay).save(out_path)
            result["overlay_path"] = str(out_path)
            log.info("Grad-CAM overlay saved → %s", out_path)
        except Exception as exc:
            log.warning("Could not save Grad-CAM overlay: %s", exc)

    return result
