"""Batch prediction module for processing multiple images."""

from __future__ import annotations

import csv
import pathlib
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from src.models.model_factory import load_checkpoint
from src.predict import build_inference_transform


def batch_predict(
    model: torch.nn.Module,
    image_paths: list[pathlib.Path],
    class_names: list[str],
    image_size: int,
    device: str = "cpu",
    confidence_threshold: float = 0.0,
) -> list[dict[str, object]]:
    """
    Perform batch prediction on multiple images.

    Args:
        model: Trained model for inference
        image_paths: List of paths to images
        class_names: List of class labels
        image_size: Image size for preprocessing
        device: Device to run inference on (cpu or cuda)
        confidence_threshold: Minimum confidence to report a defect

    Returns:
        List of prediction results with image path, label, confidence, and probabilities
    """
    transform = build_inference_transform(image_size)
    results = []

    model.eval()
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                image = Image.open(image_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                score, index = torch.max(probabilities, dim=0)

                result = {
                    "image_path": str(image_path),
                    "label": class_names[index.item()],
                    "confidence": float(score.item()),
                    "probabilities": probabilities.cpu().numpy().tolist(),
                    "defect_detected": float(score.item()) >= confidence_threshold,
                    "status": "success",
                    "error": None,
                }
            except Exception as e:
                result = {
                    "image_path": str(image_path),
                    "label": None,
                    "confidence": None,
                    "probabilities": None,
                    "defect_detected": None,
                    "status": "failed",
                    "error": str(e),
                }

            results.append(result)

    return results


def export_batch_results_to_csv(
    results: list[dict[str, object]],
    output_path: pathlib.Path,
) -> None:
    """
    Export batch prediction results to CSV file.

    Args:
        results: List of prediction results from batch_predict
        output_path: Path to save the CSV file
    """
    if not results:
        raise ValueError("No results to export")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract probability column names from first successful result
    prob_keys = []
    for result in results:
        if result["status"] == "success" and result["probabilities"] is not None:
            prob_keys = [f"prob_{i}" for i in range(len(result["probabilities"]))]
            break

    fieldnames = [
        "image_path",
        "label",
        "confidence",
        "defect_detected",
        "status",
        "error",
    ] + prob_keys

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "image_path": result["image_path"],
                "label": result["label"],
                "confidence": result["confidence"],
                "defect_detected": result["defect_detected"],
                "status": result["status"],
                "error": result["error"],
            }

            # Add probabilities if available
            if result["probabilities"] is not None:
                for i, prob in enumerate(result["probabilities"]):
                    row[f"prob_{i}"] = prob

            writer.writerow(row)
