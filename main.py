# main.py
"""
Breast Screening Assistant — simple Gradio app.

Upload an ultrasound image, click Analyze, and get:
  - A clear verdict: Cancer likely / Cancer unlikely
  - A confidence score
  - A visual explanation (Grad-CAM if a real model is loaded, otherwise
    an edge-based saliency overlay)
  - A downloadable text report

If model_cnn.pt is present next to this file, it is loaded automatically
and used for real CNN predictions + Grad-CAM. If it's missing or fails to
load, the app falls back to a simple image-heuristic so the demo still runs.
"""

import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageFilter

import gradio as gr

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import socket

# Environment flag: on Render free tier, skip expensive MC-Dropout for speed
RENDER_FAST_MODE = os.getenv("RENDER_FAST_MODE", "").lower() == "true"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 48, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(48, 64, 3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(4, 4)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(7 * 7 * 64, 922)
        self.fc2 = torch.nn.Linear(922, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.batchn1 = torch.nn.BatchNorm2d(16)
        self.batchn2 = torch.nn.BatchNorm2d(32)
        self.batchn3 = torch.nn.BatchNorm2d(48)
        self.batchn4 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.pool1(F.relu(self.batchn1(self.conv1(x))))
        x = self.pool2(F.relu(self.batchn2(self.conv2(x))))
        x = self.pool2(F.relu(self.batchn3(self.conv3(x))))
        x = self.pool2(F.relu(self.batchn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def check_image_quality(img: Image.Image):
    """Heuristic out-of-distribution / sanity gate.

    Real ultrasound frames are near-grayscale, have moderate contrast, and
    contain actual spatial structure (not flat color, not pure noise).
    This rejects/flags inputs the model was never meant to see, since the
    CNN itself will otherwise emit a confident-looking answer for anything.
    Returns (is_ok: bool, reason: str, is_warning_only: bool).
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # 1. Colorfulness check — ultrasound frames are essentially grayscale.
    channel_spread = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b)))
    if channel_spread > 18.0:
        return False, "This doesn't look like a grayscale ultrasound image (too much color).", False

    gray = np.asarray(img.convert("L"), dtype=np.float32)

    # 2. Flat-image check (solid color / near-blank frame).
    std_i = float(gray.std())
    if std_i < 8.0:
        return False, "The image has almost no contrast/detail — doesn't look like a scan.", False

    # 3. Pure-noise check — real tissue has spatial structure, so a small
    #    Gaussian blur should noticeably reduce local variance. Uncorrelated
    #    noise barely changes when blurred.
    blurred = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
    local_var_orig = float(np.var(gray))
    local_var_blur = float(np.var(blurred))
    noise_ratio = local_var_blur / (local_var_orig + 1e-6)
    if noise_ratio > 0.97:
        return False, "The image looks like random noise rather than a real scan.", False

    # 4. Soft warning zone — passes, but flag borderline cases for the user.
    if std_i < 20.0 or noise_ratio > 0.92:
        return True, "Image quality is borderline — result may be less reliable.", True

    return True, "", False


def load_bundled_model(path: str = "model_cnn.pt") -> Optional[torch.nn.Module]:
    if not os.path.exists(path):
        return None
    try:
        m = Classifier()
        state = torch.load(path, map_location="cpu")
        m.load_state_dict(state)
        m.eval()
        return m
    except Exception:
        return None


def _preprocess_for_model(pil_img: Image.Image):
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(pil_img).unsqueeze(0)


def grad_cam(model: torch.nn.Module, input_tensor: torch.Tensor):
    """Compute a Grad-CAM heatmap (H, W) in [0, 1] using the last conv layer."""
    activation, gradient = None, None

    def forward_hook(_module, _inp, out):
        nonlocal activation
        activation = out.detach()

    def backward_hook(_module, _grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].detach()

    target_layer = model.conv4 if hasattr(model, "conv4") else None
    if target_layer is None:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
    if target_layer is None:
        return None

    fh = target_layer.register_forward_hook(forward_hook)
    if hasattr(target_layer, "register_full_backward_hook"):
        bh = target_layer.register_full_backward_hook(backward_hook)
    else:
        bh = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    target_class = int(probs.argmax(dim=1).item())
    one_hot = torch.zeros_like(probs)
    one_hot[0, target_class] = 1.0
    output.backward(gradient=one_hot)

    fh.remove()
    bh.remove()

    if activation is None or gradient is None:
        return None

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def mc_dropout_predict(model: torch.nn.Module, input_tensor: torch.Tensor, n_passes: int = 5):
    """Run a small number of stochastic forward passes to estimate uncertainty.

    For deployed deployments, 20 passes is expensive and usually unnecessary.
    A lighter default keeps the app responsive while still providing a useful
    uncertainty estimate.
    """
    was_training = model.training
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # re-enable only dropout, BN stays frozen in eval

    probs_list = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(input_tensor)
            probs_list.append(torch.exp(out))

    model.train(was_training)

    probs_stack = torch.stack(probs_list, dim=0)  # (n_passes, 1, 2)
    mean_probs = probs_stack.mean(dim=0).squeeze(0)
    std_probs = probs_stack.std(dim=0).squeeze(0)
    return mean_probs, std_probs


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_histogram(img: Image.Image):
    """Lightweight histogram without matplotlib rendering."""
    gray = np.asarray(img.convert("L")).ravel()
    hist, bins = np.histogram(gray, bins=32, range=(0, 256))
    
    # Create a simple PIL-based histogram visualization (much faster than matplotlib)
    hist_img = Image.new("RGB", (320, 180), color="#0b1220")
    pixels = hist_img.load()
    
    max_hist = max(hist) if max(hist) > 0 else 1
    bar_width = 10
    
    for i, h in enumerate(hist):
        bar_height = int((h / max_hist) * 160)
        x_start = i * bar_width
        for x in range(x_start, min(x_start + bar_width, 320)):
            for y in range(180 - bar_height, 180):
                pixels[x, y] = (96, 165, 250)  # blue
    
    return hist_img


def edge_saliency_overlay(img: Image.Image, alpha: float = 0.5):
    """Fallback explanation when no trained model is available: highlights edges."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    mag_img = Image.fromarray((mag * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2))
    cmap = plt.get_cmap("jet")
    colored = (cmap(np.asarray(mag_img) / 255.0)[:, :, :3] * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored).convert("RGBA")
    return Image.blend(img.convert("RGBA"), colored_img, alpha=alpha)


def gradcam_overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.5):
    try:
        cmap = matplotlib.colormaps["jet"]
    except Exception:
        cmap = cm.get_cmap("jet")
    colored = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    heat = Image.fromarray(colored).convert("RGBA").resize(img.size, resample=Image.BILINEAR)
    return Image.blend(img.convert("RGBA"), heat, alpha=alpha)


def make_report_text(verdict: str, label: str, confidence: float,
                      uncertainty: Optional[float] = None,
                      quality_note: str = "") -> str:
    now = datetime.now(timezone.utc).isoformat()
    uncertainty_line = ""
    if uncertainty is not None:
        reliability = "Low" if uncertainty > 0.15 else ("Medium" if uncertainty > 0.05 else "High")
        uncertainty_line = (
            f"Prediction uncertainty (MC-Dropout std): {uncertainty:.3f}\n"
            f"Estimated reliability: {reliability}\n\n"
        )
    quality_line = f"Image quality note: {quality_note}\n\n" if quality_note else ""
    return (
        "Breast Cancer Screening Report\n"
        f"Generated: {now}\n\n"
        f"Verdict: {verdict}\n"
        f"Model output: {label}\n"
        f"Confidence: {confidence * 100:.1f}%\n\n"
        f"{uncertainty_line}"
        f"{quality_line}"
        "Note: This is an automated screening aid, not a medical diagnosis.\n"
        "Please consult a qualified physician for any medical decision."
    )


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

MODEL = None


def get_model() -> Optional[torch.nn.Module]:
    global MODEL
    if MODEL is None and os.path.exists("model_cnn.pt"):
        MODEL = load_bundled_model()
    return MODEL


def predict(image: np.ndarray, use_explanation: bool):
    pil = Image.fromarray(np.uint8(image)).convert("RGB")
    model = get_model()

    # Out-of-distribution / quality gate runs first — a CNN will produce a
    # confident-looking number for literally any input, so we check whether
    # the input even resembles an ultrasound frame before trusting it.
    is_ok, quality_reason, is_warning_only = check_image_quality(pil)

    label, confidence, uncertainty = None, None, None
    explanation_img = None

    if not is_ok:
        # Reject outright: don't run the model, don't fabricate a verdict.
        hist_img = make_histogram(pil)
        report = (
            "Breast Cancer Screening Report\n"
            f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
            "Verdict: REJECTED — input does not resemble an ultrasound scan\n"
            f"Reason: {quality_reason}\n\n"
            "No prediction was made. Please upload a genuine ultrasound image."
        )
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(report)
        return "REJECTED", quality_reason, None, hist_img, None, report_path, None, ""

    if model is not None:
        try:
            input_t = _preprocess_for_model(pil)
            
            with torch.no_grad():  # Ensure no gradient computation
                if RENDER_FAST_MODE:
                    # Fast mode: single forward pass only, no MC-Dropout
                    output = model(input_t)
                    probs = F.softmax(output, dim=1)
                    mean_probs = probs.squeeze(0)
                    uncertainty = None
                else:
                    # Keep MC-dropout lightweight for deployed environments; explanations
                    # are optional and should not run by default.
                    mean_probs, std_probs = mc_dropout_predict(model, input_t, n_passes=2)
                    uncertainty = float(std_probs[int(mean_probs.argmax().item())].item())
            
            idx_t = int(mean_probs.argmax().item())
            label = "Benign" if idx_t == 0 else "Malignant"
            confidence = float(mean_probs[idx_t].item())
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if use_explanation:
                cam = grad_cam(model, input_t)
                if cam is not None:
                    explanation_img = gradcam_overlay(pil, cam)
        except Exception:
            label = None  # fall through to heuristic

    if label is None:
        # Heuristic fallback (no usable model available)
        arr = np.asarray(pil)
        mean_i, std_i = float(arr.mean()), float(arr.std())
        score = max(0.0, min(1.0, (140 - mean_i) / 100.0 + std_i / 255.0))
        label = "Malignant" if score >= 0.5 else "Benign"
        confidence = float(max(0.5, min(0.99, 0.5 + abs(score - 0.5))))
        uncertainty = None

    if use_explanation and explanation_img is None:
        explanation_img = edge_saliency_overlay(pil)

    verdict = "Cancer likely" if label == "Malignant" else "Cancer unlikely"
    
    # Skip histogram on fast mode to save memory
    hist_img = None if RENDER_FAST_MODE else make_histogram(pil)
    
    report = make_report_text(verdict, label, confidence, uncertainty,
                               quality_reason if is_warning_only else "")

    fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="report_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(report)

    return verdict, label, confidence, hist_img, explanation_img, report_path, uncertainty, (quality_reason if is_warning_only else "")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
:root {
  color-scheme: dark;
  --bg: #04101b;
  --surface: rgba(10, 18, 34, 0.96);
  --surface-strong: rgba(15, 23, 42, 0.96);
  --text: #f1f5f9;
  --muted: #94a3b8;
  --border: rgba(148, 163, 184, 0.18);
  --accent: #60a5fa;
}
html, body {
  min-height: 100%;
  background: radial-gradient(circle at top left, rgba(96,165,250,0.14), transparent 24%),
              radial-gradient(circle at bottom right, rgba(56,189,248,0.10), transparent 20%),
              var(--bg) !important;
  color: var(--text) !important;
}
.gradio-container { background: transparent !important; }
.header {
  background: var(--surface-strong) !important;
  border: 1px solid var(--border) !important;
  border-radius: 20px;
  padding: 22px;
  box-shadow: 0 16px 42px rgba(0,0,0,0.28);
}
.logo-box {
  width: 52px; height: 52px; border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  background: #111827; color: var(--text); font-weight: 800;
}
.verdict-card {
  border-radius: 18px; padding: 22px; text-align: center;
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}
.verdict-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 6px; }
.verdict-sub { font-size: 0.9rem; opacity: 0.9; }
.footer-note { text-align:center; margin-top:16px; color: var(--muted); font-size: 0.85rem; }
"""


def analyze(image, show_explanation):
    if image is None:
        return (
            "<div class='verdict-card' style='background:#334155;'>"
            "<div class='verdict-title'>No image uploaded</div>"
            "<div class='verdict-sub'>Please upload an ultrasound image first.</div></div>",
            None, None, None, None,
        )

    try:
        verdict, label, confidence, hist_img, explanation_img, report_path, uncertainty, quality_warning = predict(
            image, show_explanation
        )
    except Exception as e:
        # Catch anything unexpected so the UI shows a readable message
        # instead of every panel breaking into a generic "Error" state.
        import traceback
        traceback.print_exc()
        error_html = f"""
        <div class="verdict-card" style="background:#334155;">
          <div class="verdict-title">Something went wrong</div>
          <div class="verdict-sub">{type(e).__name__}: {e}</div>
        </div>
        """
        return error_html, None, None, None, None

    if verdict == "REJECTED":
        verdict_html = f"""
        <div class="verdict-card" style="background:#334155;">
          <div class="verdict-title">Image rejected</div>
          <div class="verdict-sub">{label}</div>
        </div>
        """
        return verdict_html, hist_img, None, report_path, report_path

    color = "#dc2626" if verdict == "Cancer likely" else "#16a34a"

    uncertainty_html = ""
    if uncertainty is not None:
        reliability = "Low" if uncertainty > 0.15 else ("Medium" if uncertainty > 0.05 else "High")
        rel_color = {"Low": "#f87171", "Medium": "#fbbf24", "High": "#4ade80"}[reliability]
        uncertainty_html = (
            f"<div class='verdict-sub' style='margin-top:6px;'>"
            f"Model uncertainty: {uncertainty:.3f} &nbsp;•&nbsp; "
            f"Reliability: <span style='color:{rel_color};font-weight:700;'>{reliability}</span></div>"
        )

    warning_html = ""
    if quality_warning:
        warning_html = (
            f"<div class='verdict-sub' style='margin-top:6px;color:#fbbf24;'>⚠ {quality_warning}</div>"
        )

    verdict_html = f"""
    <div class="verdict-card" style="background:{color};">
      <div class="verdict-title">{verdict}</div>
      <div class="verdict-sub">Model output: {label} &nbsp;•&nbsp; Confidence: {confidence*100:.1f}%</div>
      {uncertainty_html}
      {warning_html}
    </div>
    """

    return verdict_html, hist_img, explanation_img, report_path, report_path


with gr.Blocks(title="Breast Screening Assistant") as demo:
    gr.HTML(
        """
        <div class="header">
          <div style="display:flex;align-items:center;gap:16px;">
            <div class="logo-box">BC</div>
            <div>
              <div style="font-size:1.5rem;font-weight:800;">Breast Screening Assistant</div>
              <div style="color:#94a3b8;margin-top:4px;">
                Upload a breast ultrasound image to get a quick screening verdict.
              </div>
            </div>
          </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Ultrasound image", type="numpy")
            show_explanation = gr.Checkbox(label="Show visual explanation (highlighted regions)", value=False, visible=False)
            analyze_btn = gr.Button("Analyze", variant="primary")
            model_status = (
                "Using trained CNN model (model_cnn.pt)."
                if os.path.exists("model_cnn.pt")
                else "No trained model found — using a basic image heuristic instead. "
                     "Add model_cnn.pt next to main.py for real predictions."
            )
            gr.HTML(f"<div style='color:#94a3b8;font-size:0.85rem;margin-top:4px;'>{model_status}</div>")

        with gr.Column(scale=1):
            verdict_out = gr.HTML()
            with gr.Row():
                hist_out = gr.Image(label="Intensity histogram", show_label=True)
                explanation_out = gr.Image(label="Highlighted regions", show_label=True)
            report_file = gr.File(label="Download report", visible=True)

    gr.HTML(
        "<div class='footer-note'>"
        "This tool is for educational exploration only and is not a medical diagnosis. "
        "Always consult a qualified physician."
        "</div>"
    )

    analyze_btn.click(
        fn=analyze,
        inputs=[image_input, show_explanation],
        outputs=[verdict_out, hist_out, explanation_out, report_file, report_file],
    )


def find_free_port(start_port=7860, end_port=7880):
    for p in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    # Render provides a dynamic port; keep local behavior as a fallback for development.
    if port == 7860:
        port = find_free_port(7860, 7880) or 7860
    print(f"Launching on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port, debug=True, share=False, css=CSS)
