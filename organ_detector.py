"""
Simple organ/modality detector used to gate analyses.

This is a lightweight inference-only module. To enable robust detection,
place a trained `organ_detector.pt` (a small PyTorch classifier) next to
`main.py`. If the model is missing, `predict_organ` returns ('unknown', 0.0)
and calling code should fall back to heuristics.

Model expected classes (index -> label):
  0 -> ultrasound
  1 -> mammogram
  2 -> mri
  3 -> other

"""
from typing import Optional, Tuple
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

MODEL_PATH = "organ_detector.pt"
_MODEL = None


class OrganDetector(torch.nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # very small conv net for inference-time loading if user trains one
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _load_model(path: str = MODEL_PATH) -> Optional[torch.nn.Module]:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.exists(path):
        return None
    try:
        m = OrganDetector()
        state = torch.load(path, map_location="cpu")
        m.load_state_dict(state)
        m.eval()
        _MODEL = m
        return _MODEL
    except Exception:
        return None


def _preprocess(img: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(img.convert("RGB")).unsqueeze(0)


def predict_organ(pil_img: Image.Image) -> Tuple[str, float]:
    """Return (label, confidence).

    If no model is available, returns ('unknown', 0.0).
    """
    model = _load_model()
    if model is None:
        return "unknown", 0.0
    try:
        t = _preprocess(pil_img)
        with torch.no_grad():
            out = model(t)
            probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        labels = ["ultrasound", "mammogram", "mri", "other"]
        return labels[idx], float(probs[idx])
    except Exception:
        return "unknown", 0.0


def is_breast_label(label: str) -> bool:
    return label in ("ultrasound", "mammogram", "mri")
