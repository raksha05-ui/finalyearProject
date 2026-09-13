"""
PET Scan Analysis — standalone module.

This module is intentionally self-contained and does NOT import or reuse
anything from the existing breast-ultrasound CNN pipeline in main.py
(Classifier, load_bundled_model, _preprocess_for_model, predict, etc.).

Why a separate pipeline instead of reusing the existing model:
  The existing `model_cnn.pt` / `Classifier` was trained specifically on
  breast ULTRASOUND images (grayscale B-mode speckle patterns) for a
  benign-vs-malignant classification. PET scans are a completely different
  imaging modality (metabolic/functional imaging, typically shown as
  colorized SUV heatmaps), so the existing model's learned features do not
  transfer and it must NOT be used to score PET images. Using it anyway
  would silently produce a confident-looking but meaningless result.

How to plug in a real trained PET model later:
  1. Place your trained PET model weights file next to main.py, e.g.:
         pet_model.pt
  2. Implement the body of `load_pet_model()` below (mirrors
     `load_bundled_model()` in main.py) to construct your PET model
     architecture and load the state dict from that file.
  3. Implement `preprocess_pet_image()` with whatever resizing/normalization
     your PET model was trained with.
  4. Implement the "real model" branch inside `predict_pet_scan()` to run
     your model on the preprocessed tensor and turn the output into
     (label, confidence, prob_dict).
  That's it — the Gradio UI wiring in main.py does not need to change.
"""

import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
except Exception:  # pragma: no cover - torch is already a project dependency
    torch = None
    T = None

# ---------------------------------------------------------------------------
# 1) MODEL LOADING (PET-specific)
# ---------------------------------------------------------------------------

PET_MODEL_PATH = "pet_model.pt"  # <-- place a trained PET model here later
_PET_MODEL = None
_PET_MODEL_LOAD_ATTEMPTED = False


def load_pet_model(path: str = PET_MODEL_PATH) -> Optional["torch.nn.Module"]:
    """Load a PET-specific trained model, if one has been provided.

    >>> THIS IS THE FUNCTION TO EDIT WHEN YOU HAVE A TRAINED PET MODEL. <<<

    Currently returns None (no PET model shipped yet), so the app falls
    back to a clearly-labeled placeholder analysis instead of ever running
    an unvalidated prediction.

    Example of what to put here once you have a trained model + a matching
    architecture class (define the class in this file, not in main.py):

        model = PetClassifier()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    """
    if not os.path.exists(path):
        return None
    try:
        # TODO: replace with your real PET model architecture + loading code.
        # model = PetClassifier()
        # state = torch.load(path, map_location="cpu")
        # model.load_state_dict(state)
        # model.eval()
        # return model
        return None
    except Exception:
        return None


def get_pet_model() -> Optional["torch.nn.Module"]:
    """Cached accessor, mirroring get_model() in main.py but fully separate."""
    global _PET_MODEL, _PET_MODEL_LOAD_ATTEMPTED
    if not _PET_MODEL_LOAD_ATTEMPTED:
        _PET_MODEL_LOAD_ATTEMPTED = True
        _PET_MODEL = load_pet_model()
    return _PET_MODEL


def pet_model_status_text() -> str:
    if os.path.exists(PET_MODEL_PATH) and get_pet_model() is not None:
        return "Using trained PET scan model (pet_model.pt)."
    return (
        "No trained PET scan model connected yet — showing a placeholder "
        "analysis only. Add pet_model.pt (and finish load_pet_model() in "
        "pet_scan_module.py) to enable real PET predictions."
    )


# ---------------------------------------------------------------------------
# 2) PREPROCESSING (PET-specific)
# ---------------------------------------------------------------------------

def preprocess_pet_image(pil_img: Image.Image):
    """Preprocess a PET scan image for the PET model.

    Kept separate from _preprocess_for_model() in main.py on purpose, since
    PET images (often colorized SUV/heatmap overlays) will likely need
    different resizing/normalization than the ultrasound pipeline once a
    real PET model is connected.
    """
    if T is None:
        return None
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


# ---------------------------------------------------------------------------
# 3) PREDICTION (PET-specific)
# ---------------------------------------------------------------------------

def predict_pet_scan(image: np.ndarray):
    """Run PET scan analysis.

    Returns a dict with:
      - label: short headline string
      - confidence: float 0-1 (or None)
      - detail: longer explanation string
      - report_path: path to a downloadable .txt report
      - is_placeholder: True if no real PET model is connected yet
    """
    if image is None:
        return {
            "label": "No image uploaded",
            "confidence": None,
            "detail": "Please upload a PET scan image first.",
            "report_path": None,
            "is_placeholder": True,
        }

    pil = Image.fromarray(np.uint8(image)).convert("RGB")
    model = get_pet_model()

    if model is not None and torch is not None:
        # --- Real PET model path (fill in once a trained model exists) ---
        input_tensor = preprocess_pet_image(pil)
        with torch.no_grad():
            # TODO: replace with your real PET model's forward pass + output
            # handling, e.g.:
            # output = model(input_tensor)
            # probs = torch.softmax(output, dim=1).squeeze(0)
            # ...
            pass
        label = "PET analysis unavailable"
        confidence = None
        detail = "Model output handling not yet implemented."
        is_placeholder = True
    else:
        # --- No trained PET model connected: transparent placeholder ---
        label = "PET model not yet connected"
        confidence = None
        detail = (
            "This module is ready to display and analyze PET scans, but no "
            "PET-specific trained model has been connected yet. No "
            "prediction is being made on this image. See pet_scan_module.py "
            "for exactly where to plug in a trained PET model."
        )
        is_placeholder = True

    report_lines = [
        "PET Scan Analysis Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Result: {label}",
        f"Confidence: {confidence if confidence is not None else 'N/A'}",
        "",
        detail,
        "",
        "AI-generated result for educational/research purposes only. "
        "This is not a medical diagnosis.",
    ]
    fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="pet_report_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "label": label,
        "confidence": confidence,
        "detail": detail,
        "report_path": report_path,
        "is_placeholder": is_placeholder,
    }


# ---------------------------------------------------------------------------
# 4) RESULT DISPLAY (PET-specific HTML, styled to match the existing app)
# ---------------------------------------------------------------------------

def build_pet_result_html(result: dict) -> str:
    label = result["label"]
    confidence = result["confidence"]
    detail = result["detail"]

    confidence_html = ""
    if confidence is not None:
        confidence_html = (
            f"<div class='verdict-sub'>Confidence: {confidence * 100:.1f}%</div>"
        )

    badge_color = "#334155" if result.get("is_placeholder") else "#2563eb"

    return f"""
    <div class="verdict-card" style="background:{badge_color};">
      <div class="verdict-title">PET Scan Result: {label}</div>
      {confidence_html}
      <div class="verdict-sub" style="margin-top:6px;">{detail}</div>
      <div class="verdict-sub" style="margin-top:10px;font-size:0.8rem;opacity:0.85;">
        AI-generated result for educational/research purposes only.
        This is not a medical diagnosis.
      </div>
    </div>
    """


def analyze_pet_scan(image):
    """Top-level function to wire directly into the Gradio button click."""
    if image is None:
        html = (
            "<div class='verdict-card' style='background:#334155;'>"
            "<div class='verdict-title'>No image uploaded</div>"
            "<div class='verdict-sub'>Please upload a PET scan image first.</div></div>"
        )
        return html, None
    result = predict_pet_scan(image)
    html = build_pet_result_html(result)
    return html, result["report_path"]
