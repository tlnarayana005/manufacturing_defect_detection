from __future__ import annotations

import pathlib

import streamlit as st
import torch
from PIL import Image

from src.data.dataset import get_class_labels, load_image_folder
from src.models.gradcam import GradCAM
from src.models.model_factory import load_checkpoint
from src.predict import build_inference_transform
from src.utils import get_device, load_config


@st.cache_resource
def load_model(checkpoint_path: pathlib.Path, model_name: str, num_classes: int, device: str):
    model, metadata = load_checkpoint(str(checkpoint_path), model_name, num_classes=num_classes, device=device)
    return model, metadata


def get_target_layer(model_name: str, model: torch.nn.Module):
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "convnext_tiny":
        return model.stages[-1][2]
    raise ValueError(f"Unsupported model: {model_name}")


def run_app() -> None:
    st.set_page_config(page_title="Manufacturing Defect Detection", layout="wide")
    st.title("AI-Powered Manufacturing Defect Detection System")
    st.markdown(
        "Upload a manufacturing product image to receive defect prediction, category, confidence, and Grad-CAM visualization."
    )

    config = load_config(pathlib.Path("configs.yaml"))
    model_name = config["training"].get("model_name", "resnet50")
    image_size = int(config["dataset"]["image_size"])
    checkpoint_path = pathlib.Path("models") / f"{model_name}_best.pth"
    device = get_device(config["training"].get("device", "cuda")).type

    if not checkpoint_path.exists():
        st.warning("No trained model checkpoint found. Train a model first using src/train.py.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("metadata", {}).get("class_names")

    if class_names is None:
        processed_root = pathlib.Path(config["dataset"]["processed_root"])
        test_dataset = load_image_folder(processed_root, "test", image_size)
        class_names, _ = get_class_labels(test_dataset)

    model, metadata = load_model(checkpoint_path, model_name, num_classes=len(class_names), device=device)

    uploaded_file = st.file_uploader("Upload a manufacturing product image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = build_inference_transform(image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        scores = torch.softmax(output, dim=1)[0]
        best_idx = int(scores.argmax())
        predicted = class_names[best_idx]
        confidence = float(scores[best_idx])

    defect_status = "Defect" if predicted.lower() != "good" else "No Defect"
    st.metric(label="Defect Status", value=defect_status)
    st.metric(label="Predicted Category", value=predicted)
    st.metric(label="Confidence", value=f"{confidence:.2%}")

    target_layer = get_target_layer(model_name, model)
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam(input_tensor, class_idx=best_idx)
    visualization = gradcam.overlay_heatmap(image, heatmap)

    st.subheader("Grad-CAM Visual Explanation")
    st.image(visualization, use_column_width=True)


if __name__ == "__main__":
    run_app()
