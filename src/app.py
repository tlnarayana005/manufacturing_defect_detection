from __future__ import annotations

import io
import pathlib

import streamlit as st
import torch
from PIL import Image

from src.batch_predict import batch_predict, export_batch_results_to_csv
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
    if model_name in {"efficientnet_b0", "efficientnet_b3"}:
        return model.features[-1]
    if model_name == "convnext_tiny":
        return model.stages[-1][2]
    raise ValueError(f"Unsupported model: {model_name}")


def run_app() -> None:
    st.set_page_config(page_title="Manufacturing Defect Detection", layout="wide")
    st.title("Manufacturing Defect Detection")
    st.markdown(
        "Upload manufacturing product images to receive defect predictions with Grad-CAM visualizations."
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

    model, _ = load_model(checkpoint_path, model_name, num_classes=len(class_names), device=device)

    # Create tabs for single and batch prediction
    tab1, tab2 = st.tabs(["Single Image Prediction", "Batch Prediction"])

    with tab1:
        st.subheader("Single Image Prediction")
        uploaded_file = st.file_uploader("Upload a manufacturing product image", type=["png", "jpg", "jpeg"], key="single")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

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
            st.image(visualization, use_container_width=True)

    with tab2:
        st.subheader("Batch Image Prediction")
        st.markdown("Upload multiple manufacturing images for batch processing. Results will be exported to CSV.")

        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="batch",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold for Defect Detection",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Predictions below this threshold will be flagged",
            )
        with col2:
            st.write("")
            st.write("")
            process_batch = st.button("Process Batch", type="primary", key="process_batch")

        if uploaded_files and process_batch:
            st.info(f"Processing {len(uploaded_files)} images...")

            # Create temporary paths for uploaded files
            image_paths = []
            for uploaded_file in uploaded_files:
                image_paths.append(pathlib.Path(uploaded_file.name))

            # Run batch prediction
            results = batch_predict(
                model,
                image_paths,
                class_names,
                image_size,
                device=device,
                confidence_threshold=confidence_threshold,
            )

            # Display results summary
            st.success(f"Batch processing complete! Processed {len(results)} images.")

            # Summary statistics
            successful = sum(1 for r in results if r["status"] == "success")
            failed = sum(1 for r in results if r["status"] == "failed")
            defects_detected = sum(1 for r in results if r.get("defect_detected") is True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Processed", len(results))
            col2.metric("Successful", successful)
            col3.metric("Failed", failed)
            col4.metric("Defects Detected", defects_detected)

            # Display results table
            st.subheader("Detailed Results")
            results_display = [
                {
                    "Image": r["image_path"].split("/")[-1],
                    "Label": r["label"] or "Error",
                    "Confidence": f"{r['confidence']:.2%}" if r["confidence"] is not None else "N/A",
                    "Defect": "Yes" if r["defect_detected"] else "No" if r["defect_detected"] is not None else "Error",
                    "Status": r["status"],
                }
                for r in results
            ]
            st.dataframe(results_display, use_container_width=True)

            # Export to CSV
            st.subheader("Export Results")
            csv_buffer = io.StringIO()
            try:
                # Create CSV content manually
                import csv

                fieldnames = ["image_path", "label", "confidence", "defect_detected", "status", "error"]
                writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow(
                        {
                            "image_path": result["image_path"],
                            "label": result["label"],
                            "confidence": result["confidence"],
                            "defect_detected": result["defect_detected"],
                            "status": result["status"],
                            "error": result["error"],
                        }
                    )

                csv_content = csv_buffer.getvalue()
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_content,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error preparing CSV export: {e}")


if __name__ == "__main__":
    run_app()
