
# main.py

import io
import os
import tempfile
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageFilter
import PIL.ImageDraw as ImageDraw

import gradio as gr

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import socket


def _make_hist_image(img: Image.Image):
    # Create a simple histogram image using NumPy and PIL (avoid ImageDraw issues).
    gray = np.asarray(img.convert("L")).ravel()
    counts, bin_edges = np.histogram(gray, bins=32, range=(0, 255))

    # Image size (w, h)
    w, h = 400, 220
    pad_left, pad_right, pad_top, pad_bottom = 40, 12, 18, 28
    plot_w = w - pad_left - pad_right
    plot_h = h - pad_top - pad_bottom

    # Normalize counts to plot height
    if counts.max() > 0:
        norm = counts / counts.max()
    else:
        norm = counts * 0.0

    # Prepare RGB background
    bg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    bg_rgb[:, :] = np.array([16, 24, 36], dtype=np.uint8)

    # Draw axes (x and y) as light lines
    axis_color = np.array([180, 190, 200], dtype=np.uint8)
    y_axis_x = pad_left
    x_axis_y = h - pad_bottom
    if 0 <= y_axis_x < w:
        bg_rgb[pad_top:x_axis_y, y_axis_x:y_axis_x+2] = axis_color
    if 0 <= x_axis_y < h:
        bg_rgb[x_axis_y:x_axis_y+2, pad_left:w-pad_right] = axis_color

    # Draw bars
    bar_w = max(1, plot_w // len(counts))
    bar_color = np.array([76, 114, 176], dtype=np.uint8)
    for i, v in enumerate(norm):
        bx = pad_left + i * bar_w
        by = pad_top + int((1.0 - v) * plot_h)
        x0 = int(bx)
        x1 = int(min(w - pad_right, bx + bar_w - 1))
        y0 = int(by)
        y1 = x_axis_y - 1
        if x0 < x1 and y0 < y1:
            bg_rgb[y0:y1, x0:x1] = bar_color

    # Convert to PIL Image and attempt to add simple title using default font
    im = Image.fromarray(bg_rgb)
    try:
        from PIL import ImageFont
        from PIL import ImageDraw as _ID
        draw = _ID.Draw(im)
        font = ImageFont.load_default()
        draw.text((pad_left, 4), "Pixel intensity distribution", fill=(226, 232, 240), font=font)
    except Exception:
        pass

    return im


def _make_saliency_overlay(img: Image.Image, alpha: float = 0.5):
    gray = np.asarray(img.convert("L"), dtype=np.float32)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)

    if mag.max() > 0:
        mag = (mag - mag.min()) / (mag.max() - mag.min())
    else:
        mag = mag * 0.0

    mag_img = Image.fromarray((mag * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2))
    cmap = plt.get_cmap("jet")
    mag_arr = np.asarray(mag_img) / 255.0
    colored = (cmap(mag_arr)[:, :, :3] * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored)

    orig = img.convert("RGBA")
    colored_img = colored_img.convert("RGBA")
    blended = Image.blend(orig, colored_img, alpha=alpha)
    return blended


def _make_report_text(result: str, confidence: float):
    now = datetime.utcnow().isoformat() + "Z"
    return (
        f"Breast Cancer Quick Assessment Report\nGenerated: {now}\n\n"
        f"Result: {result}\nConfidence: {confidence*100:.1f}%\n\n"
        "Notes: This is a heuristic demo. Not a medical diagnosis."
    )


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


def load_user_model(path: str = "model_cnn.pt") -> Optional[torch.nn.Module]:
    if not os.path.exists(path):
        return None
    try:
        m = Classifier()
        state = torch.load(path, map_location="cpu")
        m.load_state_dict(state)
        m.eval()
        return m
    except Exception:
        try:
            r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            r.fc = torch.nn.Linear(r.fc.in_features, 2)
            state = torch.load(path, map_location="cpu")
            r.load_state_dict(state)
            r.eval()
            return r
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


def grad_cam(model: torch.nn.Module, input_tensor: torch.Tensor, target_layer=None, target_class=None):
    """Compute Grad-CAM heatmap and return a normalized 2D numpy array (H,W) in [0,1]."""
    device = next(model.parameters()).device
    model.to(device)
    input_tensor = input_tensor.to(device)

    activation = None
    gradient = None

    def forward_hook(module, inp, out):
        nonlocal activation
        activation = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].detach()

    if target_layer is None:
        candidates = ["layer4", "conv4", "conv3", "features"]
        target_layer = None
        for name, module in model.named_modules():
            if any(c in name for c in candidates):
                target_layer = module
                break
        if target_layer is None:
            for module in reversed(list(model.modules())):
                if isinstance(module, (torch.nn.Conv2d,)):
                    target_layer = module
                    break

    if target_layer is None:
        return None

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    if isinstance(output, tuple):
        output = output[0]
    probs = F.softmax(output, dim=1)
    if target_class is None:
        target_class = int(probs.argmax(dim=1).item())

    one_hot = torch.zeros_like(probs)
    one_hot[0, target_class] = 1.0
    output.backward(gradient=one_hot)

    fh.remove()
    bh.remove()

    if activation is None or gradient is None:
        return None

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam


def predict(image: np.ndarray, model: Optional[torch.nn.Module] = None):
    try:
        if image is None:
            return "", 0.0, None, None, None

        pil = Image.fromarray(np.uint8(image)).convert("RGB")
        hist_img = _make_hist_image(pil)
        saliency_overlay = _make_saliency_overlay(pil)

        if model is not None:
            try:
                input_t = _preprocess_for_model(pil)
                with torch.no_grad():
                    output = model(input_t)
                if isinstance(output, tuple):
                    output = output[0]
                probs = F.softmax(output, dim=1)
                confidence, idx = probs.max(dim=1)
                label = "Likely benign" if int(idx.item()) == 0 else "Possible malignant"
                confidence = float(confidence.item())
            except Exception:
                model = None

        if model is None:
            arr = np.asarray(pil)
            mean_intensity = float(arr.mean())
            std_intensity = float(arr.std())

            score = max(0.0, min(1.0, (140 - mean_intensity) / 100.0 + std_intensity / 255.0))
            if score < 0.35:
                label = "Likely benign"
            elif score < 0.65:
                label = "Unclear — recommend follow-up"
            else:
                label = "Possible malignant — seek medical advice"

            confidence = 1.0 - abs(0.5 - score) * 2.0
            confidence = float(max(0.0, min(1.0, confidence)))

        report = _make_report_text(label, confidence)
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(report)

        return label, confidence, hist_img, path, saliency_overlay
    except Exception:
        import traceback
        traceback.print_exc()
        # return safe fallbacks
        try:
            from PIL import Image as PILImage
            fallback = PILImage.new('RGBA', (400, 220), (30, 30, 30, 255))
        except Exception:
            fallback = None
        return 'Error', 0.0, fallback, None, fallback


EXAMPLES = []


# try to load a user-provided model from disk (model_cnn.pt)
user_model = load_user_model()

css = """
:root {
  color-scheme: dark;
  --bg: #07101d;
  --surface: rgba(10, 18, 34, 0.96);
  --surface-strong: rgba(15, 23, 42, 0.96);
  --surface-muted: rgba(30, 41, 59, 0.95);
  --text: #e2e8f0;
  --muted: #94a3b8;
  --border: rgba(148, 163, 184, 0.18);
  --accent: #60a5fa;
  --accent-soft: rgba(96, 165, 250, 0.16);
}
html, body {
  min-height: 100%;
  margin: 0;
  padding: 0;
  background: radial-gradient(circle at top left, rgba(59, 130, 246, 0.16), transparent 24%),
              radial-gradient(circle at bottom right, rgba(56, 189, 248, 0.12), transparent 20%),
              var(--bg) !important;
  color: var(--text) !important;
}
.gradio-container,
.gradio-container > div,
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-row,
.gradio-container .gr-column,
.gradio-container .gr-form,
.gradio-container .gr-interface,
.gradio-container .gr-tabs,
.gradio-container .gr-panel,
.gradio-container .gr-image,
.gradio-container .gr-file,
.gradio-container .gr-slider,
.gradio-container .gr-dropdown,
.gradio-container .gr-number,
.gradio-container .gr-checkbox {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
.gradio-container {
  background: transparent !important;
}
.gradio-container .gr-box,
.gradio-container .gr-row,
.gradio-container .gr-column {
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18) !important;
}
.header {
  background: var(--surface-strong) !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 16px 42px rgba(0, 0, 0, 0.24);
  backdrop-filter: blur(22px);
}
.logo-box {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #111827;
  color: var(--text);
  font-weight: 800;
}
.hero-title,
.hero-sub,
.small.muted,
.gr-markdown,
.gr-text,
.gr-label,
.gr-block-title,
.gr-form,
.gr-number,
.gr-dropdown,
.gr-slider,
.gr-checkbox {
  color: var(--text) !important;
}
.small.muted,
.gr-markdown .muted {
  color: var(--muted) !important;
}
.gr-button,
.gr-button:hover,
.gr-button:focus {
  background: var(--accent) !important;
  color: white !important;
  border: none !important;
}
.gr-button {
  box-shadow: 0 12px 28px rgba(59, 130, 246, 0.18) !important;
}
"""

with gr.Blocks(title="Breast Screening Assistant") as demo:
    # Header / Hero
    with gr.Row(elem_id="top-row"):
        with gr.Column(scale=3):
            gr.HTML(
                """
                <style>
                :root {
                  color-scheme: dark;
                  --bg: #04101b;
                  --surface: rgba(8, 14, 24, 0.96);
                  --surface-strong: rgba(12, 18, 32, 0.96);
                  --surface-muted: rgba(17, 25, 43, 0.98);
                  --text: #f8fafc;
                  --muted: #94a3b8;
                  --border: rgba(148, 163, 184, 0.22);
                  --accent: #60a5fa;
                }
                html, body {
                  min-height: 100%;
                  margin: 0;
                  padding: 0;
                  background: radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 24%),
                              radial-gradient(circle at bottom right, rgba(56, 189, 248, 0.12), transparent 20%),
                              var(--bg) !important;
                  color: var(--text) !important;
                }
                .gradio-container,
                .gradio-container > div,
                .gradio-container .gr-block,
                .gradio-container .gr-box,
                .gradio-container .gr-row,
                .gradio-container .gr-column,
                .gradio-container .gr-form,
                .gradio-container .gr-interface,
                .gradio-container .gr-tabs,
                .gradio-container .gr-panel,
                .gradio-container .gr-block-title,
                .gradio-container .gr-markdown,
                .gradio-container .gr-text,
                .gradio-container .gr-label,
                .gradio-container .gr-button,
                .gradio-container .gr-number,
                .gradio-container .gr-dropdown,
                .gradio-container .gr-slider,
                .gradio-container .gr-file,
                .gradio-container .gr-image,
                .gradio-container .gr-checkbox {
                  color: var(--text) !important;
                }
                .gradio-container,
                .gradio-container > div,
                .gradio-container .gr-block,
                .gradio-container .gr-box,
                .gradio-container .gr-row,
                .gradio-container .gr-column,
                .gradio-container .gr-form,
                .gradio-container .gr-interface,
                .gradio-container .gr-tabs,
                .gradio-container .gr-panel {
                  background: var(--surface) !important;
                  border: 1px solid var(--border) !important;
                }
                .gradio-container .gr-input,
                .gradio-container .gr-button,
                .gradio-container .gr-dropdown,
                .gradio-container .gr-slider,
                .gradio-container .gr-number,
                .gradio-container .gr-file,
                .gradio-container .gr-image,
                .gradio-container .gr-checkbox {
                  background: var(--surface-muted) !important;
                  color: var(--text) !important;
                  border-color: rgba(148, 163, 184, 0.14) !important;
                }
                .gradio-container .gr-button,
                .gradio-container .gr-button:hover,
                .gradio-container .gr-button:focus {
                  background: var(--accent) !important;
                  color: white !important;
                  border: none !important;
                }
                .header {
                  background: var(--surface-strong) !important;
                  border: 1px solid var(--border) !important;
                  box-shadow: 0 16px 44px rgba(0, 0, 0, 0.28);
                  backdrop-filter: blur(20px);
                }
                .logo-box {
                  background: rgba(12, 18, 32, 0.96);
                  color: var(--text);
                }
                .hero-sub,
                .small.muted,
                .gr-markdown .muted {
                  color: var(--muted) !important;
                }
                </style>
                <div class="header" style="padding: 22px; border-radius: 24px;">
                    <div style="display:flex;align-items:center;gap:16px;">
                        <div class="logo-box">BC</div>
                        <div>
                            <div class="hero-title" style="font-size:1.55rem; font-weight:800; letter-spacing:-0.02em;">Breast Screening Assistant</div>
                            <div class="hero-sub" style="margin-top:4px;">Upload an ultrasound image and receive a clear screening summary and visual explanations.</div>
                        </div>
                    </div>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown("**Quick tips**\n- Use high-quality ultrasound crops.\n- For model explanations enable Grad-CAM and provide `model_cnn.pt`.")

    

    # Main content with sidebar: sidebar | main | results
    with gr.Row():
        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### Session")
            last_analysis_out = gr.Markdown("**No analyses yet**")
            recent_out = gr.Markdown("No recent analyses")
            clear_history = gr.Button("Clear history")
            save_result = gr.Button("Save result")
            save_csv = gr.File(label="Saved CSV")
            gr.Markdown("---")
            gr.Markdown("### Model")
            model_info = gr.HTML("<div class='small muted'>No model loaded. Uses bundled `model_cnn.pt` if available.</div>")
            refresh_model = gr.Button("Refresh model info")
            load_model_btn = gr.Button("Load bundled model")

        # Main area: input and controls
        with gr.Column(scale=2):
            gr.Markdown("### Input")
            image_input = gr.Image(label="Upload ultrasound image", type="numpy")
            show_quick_card = gr.Checkbox(label="Show quick assessment card", value=False)
            with gr.Row():
                with gr.Column(scale=2):
                    btn = gr.Button("Analyze", variant="primary", elem_classes="btn-primary")
                    notebook_analyze_btn = gr.Button("Notebook Analyze")
                    notebook_hist_btn = gr.Button("Notebook Histogram")
                    notebook_saliency_btn = gr.Button("Notebook Saliency")
                with gr.Column(scale=1):
                    model_file = gr.File(label="Optional model file (.pt)", file_types=[".pt"], type="filepath")
                    use_gradcam = gr.Checkbox(label="Use Grad-CAM explanation", value=True)
                    cmap_select = gr.Dropdown(label="Colormap", choices=["jet", "viridis", "plasma", "magma"], value="jet")
                    threshold = gr.Slider(label="Threshold (hide low activations)", minimum=0.0, maximum=1.0, step=0.01, value=0.15)
                    alpha = gr.Slider(label="Overlay alpha", minimum=0.0, maximum=1.0, step=0.05, value=0.5)
            gr.HTML("<div class='small muted'>Upload model_cnn.pt to enable real CNN prediction and Grad-CAM explanation.</div>")

        # Right column: results
        with gr.Column(scale=1):
            gr.Markdown("### Assessment")
            with gr.Row():
                with gr.Column(scale=1):
                    label_out = gr.Label(label="Assessment")
                with gr.Column(scale=1):
                    conf_out = gr.Number(label="Confidence (0–1)")
                with gr.Column(scale=1):
                    conf_badge = gr.HTML()
                with gr.Column(scale=1):
                    diag_out = gr.Textbox(label="Cancer likelihood", interactive=False)
            gr.Markdown("---")
            gr.Markdown("#### Intensity histogram")
            hist_out = gr.Image(label="Intensity histogram")
            gr.Markdown("#### Explanation")
            saliency_out = gr.Image(label="Explanation (saliency overlay)")
            saliency_file = gr.File(label="Download saliency image")
            gr.Markdown("---")
            download_report = gr.File(label="Download report")
            quick_card = gr.HTML()

    # Footer
    gr.HTML("<div style='text-align:center;margin-top:12px;color:#94a3b8;'>This demo is for exploration only — not a medical diagnosis.</div>")

    def analyze_and_prepare(image, model_file, use_gradcam_flag=True, cmap_name="jet", thresh=0.15, alpha_val=0.5, show_quick=True):
        """Analyze an uploaded image and prepare outputs for the Gradio UI.

        Errors are caught and logged; a safe fallback is returned on failure.
        """
        # Default safe fallbacks
        fallback_img = None
        try:
            from PIL import Image as PILImage
            fallback_img = PILImage.new('RGBA', (400, 220), (30, 30, 30, 255))
        except Exception:
            fallback_img = None

        try:
            active_model = user_model
            if model_file is not None:
                model_path = None
                if isinstance(model_file, dict):
                    model_path = model_file.get("tmp_path") or model_file.get("name")
                elif isinstance(model_file, str):
                    model_path = model_file
                elif hasattr(model_file, "name"):
                    model_path = model_file.name

                if model_path:
                    loaded_model = load_user_model(model_path)
                    if loaded_model is not None:
                        active_model = loaded_model

            label, confidence, hist_img, report_path, saliency_img = predict(image, active_model)

        except Exception:
            import traceback
            traceback.print_exc()
            label = 'Error'
            confidence = 0.0
            hist_img = fallback_img
            report_path = None
            saliency_img = fallback_img

        # compute human-friendly diagnosis
        try:
            lbl = (label or "").lower()
            c = float(confidence)
            if "malignant" in lbl:
                if c >= 0.6:
                    diagnosis = "Likely cancer: YES"
                elif c >= 0.4:
                    diagnosis = "Possible cancer — follow-up recommended"
                else:
                    diagnosis = "Unclear — follow-up recommended"
            elif "benign" in lbl:
                if c >= 0.6:
                    diagnosis = "Likely cancer: NO"
                elif c >= 0.4:
                    diagnosis = "Probably not cancer — consider follow-up"
                else:
                    diagnosis = "Unclear — follow-up recommended"
        except Exception:
            diagnosis = "Unclear"

        # confidence badge HTML
        try:
            c = float(confidence)
            if c >= 0.6:
                color = '#16a34a'  # green
                text = 'LOW'
            elif c >= 0.4:
                color = '#f59e0b'  # amber
                text = 'MED'
            else:
                color = '#ef4444'  # red
                text = 'HIGH'
            conf_badge_html = f"<div style='display:inline-block;padding:6px 10px;border-radius:12px;background:{color};color:#fff;font-weight:700'>{text}</div>"
        except Exception:
            conf_badge_html = ""

        # append diagnosis to report file if present
        try:
            if report_path and os.path.exists(report_path):
                with open(report_path, 'a', encoding='utf-8') as fh:
                    fh.write('\nDiagnosis: ' + (diagnosis or '') + '\n')
        except Exception:
            pass
        # if the user requested Grad-CAM and a model is available, compute it
        if use_gradcam_flag and active_model is not None and image is not None:
            try:
                pil_img = Image.fromarray(np.uint8(image)).convert("RGB")
                input_t = _preprocess_for_model(pil_img)
                cam = grad_cam(active_model, input_t)
                if cam is not None:
                    cmap = cm.get_cmap(cmap_name)
                    colored = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
                    heat = Image.fromarray(colored).convert("RGBA")
                    if thresh is not None and thresh > 0.0:
                        mask = (cam >= thresh).astype(np.uint8) * 255
                        alpha_channel = Image.fromarray(mask).convert("L")
                        heat.putalpha(alpha_channel)
                    else:
                        heat.putalpha(int(255 * alpha_val))

                    heat = heat.resize(pil_img.size, resample=Image.BILINEAR)
                    orig = pil_img.convert("RGBA")
                    if thresh is not None and thresh > 0.0:
                        saliency_img = Image.alpha_composite(orig, heat)
                        saliency_img = Image.blend(orig, saliency_img, alpha=alpha_val)
                    else:
                        saliency_img = Image.blend(orig, heat, alpha=alpha_val)
            except Exception:
                pass
        # `gr.Label` expects either a string or a mapping {label: score}.
        label_mapping = {label: float(confidence)}

        # Derive a simple diagnosis string for the UI
        try:
            lbl = (label or "").lower()
            if "malignant" in lbl:
                if confidence >= 0.6:
                    diagnosis = "Likely cancer: YES"
                elif confidence >= 0.4:
                    diagnosis = "Possible cancer — follow-up recommended"
                else:
                    diagnosis = "Unclear — follow-up recommended"
            elif "benign" in lbl:
                if confidence >= 0.6:
                    diagnosis = "Likely cancer: NO"
                elif confidence >= 0.4:
                    diagnosis = "Probably not cancer — consider follow-up"
                else:
                    diagnosis = "Unclear — follow-up recommended"
            else:
                diagnosis = "Unclear — follow-up recommended"
        except Exception:
            diagnosis = "Unclear"

        # update recent history (store in simple JSON)
        try:
            rec_file = os.path.join(os.path.dirname(__file__), 'recent.json')
            recs = []
            if os.path.exists(rec_file):
                with open(rec_file, 'r', encoding='utf-8') as fh:
                    recs = json.load(fh)
            entry = { 'time': datetime.utcnow().isoformat() + 'Z', 'label': label, 'confidence': float(confidence) }
            recs.insert(0, entry)
            recs = recs[:8]
            with open(rec_file, 'w', encoding='utf-8') as fh:
                json.dump(recs, fh)
            last_text = f"**Last:** {entry['time']} — {entry['label']} ({entry['confidence']*100:.1f}%)"
            recent_md = "\n\n".join([f"- {r['time']}: **{r['label']}** ({r['confidence']*100:.1f}%)" for r in recs])
        except Exception:
            last_text = "**No analyses yet**"
            recent_md = "No recent analyses"

        # prepare quick assessment HTML (optional)
        quick_html = ""
        if show_quick:
            quick_html = f"""
            <div style='background:#ffffff; color:#0f172a; border-radius:14px; padding:16px; box-shadow:0 10px 30px rgba(15,23,42,0.08);'>
              <h3 style='color:#0f172a;'>Quick assessment</h3>
              <p style='color:#0f172a;'><strong>Signal score:</strong> {int(confidence*100)}/100</p>
              <p style='color:#0f172a;'><strong>Result:</strong> {label}</p>
            </div>
            """

        # save saliency image for download if available
        saliency_path = None
        try:
            if saliency_img is not None:
                fd_s, saliency_path = tempfile.mkstemp(suffix='.png', prefix='saliency_')
                os.close(fd_s)
                saliency_img.convert('RGB').save(saliency_path, format='PNG')
        except Exception:
            saliency_path = None

        return label_mapping, float(confidence), conf_badge_html, diagnosis, hist_img, saliency_img, report_path, last_text, recent_md, quick_html, saliency_path

    btn.click(fn=analyze_and_prepare, inputs=[image_input, model_file, use_gradcam, cmap_select, threshold, alpha, show_quick_card], outputs=[label_out, conf_out, conf_badge, diag_out, hist_out, saliency_out, download_report, last_analysis_out, recent_out, quick_card, saliency_file])

    # --- Notebook helper functions (from the enhanced notebook) ---
    def _notebook_analyze_image(path_or_array):
        # returns dict with label and score
        if isinstance(path_or_array, str):
            img = Image.open(path_or_array).convert('RGB')
        else:
            img = Image.fromarray(np.uint8(path_or_array)).convert('RGB')
        arr = np.asarray(img)
        mean = float(arr.mean())
        std = float(arr.std())
        score = max(0.0, min(1.0, (140 - mean) / 100.0 + std / 255.0))
        if score < 0.35:
            label = 'Likely benign'
        elif score < 0.65:
            label = 'Unclear — recommend follow-up'
        else:
            label = 'Possible malignant'
        return {'label': label, 'score': float(score), 'mean': mean, 'std': std, 'pil': img}

    def _notebook_saliency_overlay(path_or_array, alpha=0.5):
        if isinstance(path_or_array, str):
            img = Image.open(path_or_array).convert('RGB')
        else:
            img = Image.fromarray(np.uint8(path_or_array)).convert('RGB')
        gray = np.asarray(img.convert('L'), dtype=np.float32)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx ** 2 + gy ** 2)
        if mag.max() > 0:
            mag = (mag - mag.min()) / (mag.max() - mag.min())
        else:
            mag = mag * 0.0
        mag_img = Image.fromarray((mag * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2))
        cmap = plt.get_cmap('jet')
        mag_arr = np.asarray(mag_img) / 255.0
        colored = (cmap(mag_arr)[:, :, :3] * 255).astype(np.uint8)
        colored_img = Image.fromarray(colored).convert('RGBA')
        blended = Image.blend(img.convert('RGBA'), colored_img, alpha=alpha)
        return blended

    def notebook_analyze_callback(image):
        if image is None:
            return { '': 0 }, 0.0, '', 'No image', None, None, None, "**No analyses yet**", "No recent analyses", "", None
        info = _notebook_analyze_image(image)
        label = info['label']
        confidence = float(info['score'])
        pil_img = info['pil']
        # create hist and saliency using existing helpers
        hist_img = _make_hist_image(pil_img)
        saliency_img = _notebook_saliency_overlay(pil_img, alpha=0.5)
        # build mapping and diagnosis
        label_mapping = {label: confidence}
        try:
            if confidence >= 0.6:
                conf_badge_html = "<div style='display:inline-block;padding:6px 10px;border-radius:12px;background:#16a34a;color:#fff;font-weight:700'>LOW</div>"
            elif confidence >= 0.4:
                conf_badge_html = "<div style='display:inline-block;padding:6px 10px;border-radius:12px;background:#f59e0b;color:#fff;font-weight:700'>MED</div>"
            else:
                conf_badge_html = "<div style='display:inline-block;padding:6px 10px;border-radius:12px;background:#ef4444;color:#fff;font-weight:700'>HIGH</div>"
        except Exception:
            conf_badge_html = ''
        # diagnosis text
        if 'malignant' in label.lower():
            diagnosis = 'Possible cancer — follow-up recommended' if confidence < 0.6 else 'Likely cancer: YES'
        elif 'benign' in label.lower():
            diagnosis = 'Likely cancer: NO' if confidence >= 0.6 else 'Probably not cancer — consider follow-up'
        else:
            diagnosis = 'Unclear — follow-up recommended'
        # write report
        try:
            report = _make_report_text(label, confidence)
            fd, path = tempfile.mkstemp(suffix='.txt', prefix='report_')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(report + '\nDiagnosis: ' + diagnosis + '\n')
        except Exception:
            path = None
        # update recent
        try:
            rec_file = os.path.join(os.path.dirname(__file__), 'recent.json')
            recs = []
            if os.path.exists(rec_file):
                with open(rec_file, 'r', encoding='utf-8') as fh:
                    recs = json.load(fh)
            entry = { 'time': datetime.utcnow().isoformat() + 'Z', 'label': label, 'confidence': float(confidence) }
            recs.insert(0, entry)
            recs = recs[:8]
            with open(rec_file, 'w', encoding='utf-8') as fh:
                json.dump(recs, fh)
            last_text = f"**Last:** {entry['time']} — {entry['label']} ({entry['confidence']*100:.1f}%)"
            recent_md = "\n\n".join([f"- {r['time']}: **{r['label']}** ({r['confidence']*100:.1f}%)" for r in recs])
        except Exception:
            last_text = "**No analyses yet**"
            recent_md = "No recent analyses"

        # save saliency image for download
        saliency_path = None
        try:
            if saliency_img is not None:
                fd_s, saliency_path = tempfile.mkstemp(suffix='.png', prefix='saliency_')
                os.close(fd_s)
                saliency_img.convert('RGB').save(saliency_path, format='PNG')
        except Exception:
            saliency_path = None

        return label_mapping, float(confidence), conf_badge_html, diagnosis, hist_img, saliency_img, path, last_text, recent_md, "", saliency_path

    def notebook_hist_callback(image):
        if image is None:
            return None
        pil = Image.fromarray(np.uint8(image)).convert('RGB')
        return _make_hist_image(pil)

    def notebook_saliency_callback(image):
        if image is None:
            return None
        pil = Image.fromarray(np.uint8(image)).convert('RGB')
        return _notebook_saliency_overlay(pil, alpha=0.5)

    notebook_analyze_btn.click(fn=notebook_analyze_callback, inputs=[image_input], outputs=[label_out, conf_out, conf_badge, diag_out, hist_out, saliency_out, download_report, last_analysis_out, recent_out, quick_card, saliency_file])
    notebook_hist_btn.click(fn=notebook_hist_callback, inputs=[image_input], outputs=[hist_out])
    notebook_saliency_btn.click(fn=notebook_saliency_callback, inputs=[image_input], outputs=[saliency_out])

    def clear_history_fn():
        try:
            rec_file = os.path.join(os.path.dirname(__file__), 'recent.json')
            if os.path.exists(rec_file):
                os.remove(rec_file)
        except Exception:
            pass
        return "**No analyses yet**", "No recent analyses"

    clear_history.click(fn=clear_history_fn, inputs=None, outputs=[last_analysis_out, recent_out])

    def _format_model_info(path):
        if not path or not os.path.exists(path):
            return "<div class='small muted'>No model file found at <code>model_cnn.pt</code>.</div>"
        try:
            st = os.stat(path)
            size_mb = st.st_size / (1024 * 1024)
            mtime = datetime.utcfromtimestamp(st.st_mtime).isoformat() + 'Z'
            return f"<div><strong>Model:</strong> {Path(path).name}<br><strong>Size:</strong> {size_mb:.2f} MB<br><strong>Modified:</strong> {mtime}</div>"
        except Exception:
            return "<div class='small muted'>Unable to read model metadata.</div>"

    def refresh_model_info_fn(model_file):
        # prefer uploaded model path if provided
        model_path = None
        if model_file is not None:
            if isinstance(model_file, dict):
                model_path = model_file.get("tmp_path") or model_file.get("name")
            elif isinstance(model_file, str):
                model_path = model_file
            elif hasattr(model_file, "name"):
                model_path = model_file.name

        if model_path and os.path.exists(model_path):
            return _format_model_info(model_path)
        # fallback to bundled model
        bundled = os.path.join(os.path.dirname(__file__), "model_cnn.pt")
        return _format_model_info(bundled)

    def load_bundled_model_fn():
        global user_model
        bundled = os.path.join(os.path.dirname(__file__), "model_cnn.pt")
        loaded = load_user_model(bundled)
        if loaded is not None:
            user_model = loaded
            return _format_model_info(bundled)
        return "<div class='small muted'>Bundled `model_cnn.pt` not found or could not be loaded.</div>"

    refresh_model.click(fn=refresh_model_info_fn, inputs=[model_file], outputs=[model_info])
    load_model_btn.click(fn=load_bundled_model_fn, inputs=None, outputs=[model_info])

    # Batch prediction and model info
    gr.Markdown("### Batch predictions")
    batch_files = gr.Files(label="Upload multiple images (optional)")
    run_batch = gr.Button("Run batch predictions")
    batch_csv = gr.File(label="Download batch CSV")

    def batch_predict(files, model_file, use_gradcam_flag=True, cmap_name="jet", thresh=0.15, alpha_val=0.5):
        if not files:
            return None
        rows = []
        # try to load provided model or use user_model
        active_model = user_model
        if model_file is not None:
            model_path = None
            if isinstance(model_file, dict):
                model_path = model_file.get("tmp_path") or model_file.get("name")
            elif isinstance(model_file, str):
                model_path = model_file
            elif hasattr(model_file, "name"):
                model_path = model_file.name

            if model_path:
                loaded_model = load_user_model(model_path)
                if loaded_model is not None:
                    active_model = loaded_model

        for f in files:
            try:
                img = Image.open(f.name if hasattr(f, 'name') else f).convert('RGB')
                label, confidence, hist_img, report_path, saliency = predict(np.array(img), active_model)
                rows.append({"filename": Path(f.name).name if hasattr(f, 'name') else str(f), "label": label, "confidence": confidence})
            except Exception:
                rows.append({"filename": getattr(f, 'name', str(f)), "label": "error", "confidence": 0.0})

        if not rows:
            return None

        fd, path = tempfile.mkstemp(suffix=".csv", prefix="batch_preds_")
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["filename", "label", "confidence"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        return path

    run_batch.click(fn=batch_predict, inputs=[batch_files, model_file, use_gradcam, cmap_select, threshold, alpha], outputs=[batch_csv])

    def export_history_fn():
        rec_file = os.path.join(os.path.dirname(__file__), 'recent.json')
        if not os.path.exists(rec_file):
            return None
        try:
            with open(rec_file, 'r', encoding='utf-8') as fh:
                recs = json.load(fh)
        except Exception:
            recs = []
        if not recs:
            return None
        fd, path = tempfile.mkstemp(suffix='.csv', prefix='history_')
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['time','label','confidence'])
            writer.writeheader()
            for r in recs:
                writer.writerow(r)
        return path

    export_history = gr.Button('Download history CSV')
    export_out = gr.File(label='History CSV')
    export_history.click(fn=export_history_fn, inputs=None, outputs=[export_out])

    def save_result_fn(label_map, confidence, diagnosis):
        # label_map is a mapping {label: score}
        try:
            label = next(iter(label_map.keys())) if label_map else (diagnosis or 'unknown')
        except Exception:
            label = diagnosis or 'unknown'
        try:
            rec_file = os.path.join(os.path.dirname(__file__), 'recent.json')
            recs = []
            if os.path.exists(rec_file):
                with open(rec_file, 'r', encoding='utf-8') as fh:
                    recs = json.load(fh)
        except Exception:
            recs = []

        entry = { 'time': datetime.utcnow().isoformat() + 'Z', 'label': label, 'confidence': float(confidence), 'diagnosis': diagnosis }
        recs.insert(0, entry)
        recs = recs[:64]
        try:
            with open(rec_file, 'w', encoding='utf-8') as fh:
                json.dump(recs, fh)
        except Exception:
            pass

        # write CSV
        try:
            fd, path = tempfile.mkstemp(suffix='.csv', prefix='saved_results_')
            with os.fdopen(fd, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['time','label','confidence','diagnosis'])
                writer.writeheader()
                for r in recs:
                    writer.writerow({'time': r.get('time'), 'label': r.get('label'), 'confidence': r.get('confidence'), 'diagnosis': r.get('diagnosis', '')})
        except Exception:
            path = None

        # return updated UI snippets and file
        last_text = f"**Last:** {entry['time']} — {entry['label']} ({entry['confidence']*100:.1f}%)"
        recent_md = "\n\n".join([f"- {r['time']}: **{r['label']}** ({r['confidence']*100:.1f}%)" for r in recs])
        return last_text, recent_md, path

    save_result.click(fn=save_result_fn, inputs=[label_out, conf_out, diag_out], outputs=[last_analysis_out, recent_out, save_csv])


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
    port = find_free_port(7860, 7880) or 7860
    print(f"Launching on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port, debug=True)

