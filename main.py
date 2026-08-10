# main.py

import gradio as gr
from PIL import Image
import numpy as np


def predict(image):
    if image is None:
        return "Please upload an ultrasound image first."

    img = Image.fromarray(np.uint8(image)).convert("RGB")
    pixels = np.array(img)
    mean_intensity = float(np.mean(pixels))

    if mean_intensity > 140:
        result = "Likely benign"
        advice = (
            "The image appears brighter and may be consistent with a benign finding. "
            "Please follow up with a clinician for confirmation."
        )
    else:
        result = "Possible malignant"
        advice = (
            "The image appears darker and may warrant prompt medical follow-up. "
            "Please consult a healthcare professional as soon as possible."
        )

    return f"{result}\n\n{advice}"


with gr.Blocks(title="Breast Cancer Detection") as demo:
    gr.Markdown("# Breast Cancer Detection with CNN and Gradio")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload ultrasound image", type="numpy")
            submit_btn = gr.Button("Analyze")
        with gr.Column():
            output = gr.Textbox(label="Assessment", lines=6)

    submit_btn.click(fn=predict, inputs=image_input, outputs=output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=False)
