from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from yaml import safe_load

from src.data.dataset import load_image_folder, get_class_labels
from src.models.metrics import compute_metrics
from src.models.model_factory import load_checkpoint


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return safe_load(handle)


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def run_evaluation(config: dict, checkpoint: Path, model_name: str) -> None:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    paths_cfg = config["paths"]
    device = torch.device(training_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    image_size = int(dataset_cfg["image_size"])
    processed_root = Path(dataset_cfg["processed_root"])

    test_dataset = load_image_folder(processed_root, "test", image_size)
    class_names, _ = get_class_labels(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=training_cfg["batch_size"], shuffle=False, num_workers=min(8, os.cpu_count() or 0))

    model, _ = load_checkpoint(str(checkpoint), model_name, num_classes=len(class_names), device=str(device))
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.tolist())
            y_prob.extend(probabilities.tolist())

    metrics = compute_metrics(y_true, y_pred, class_names)
    if len(class_names) > 2:
        y_true_binary = label_binarize(y_true, classes=list(range(len(class_names))))
        metrics["roc_auc"] = float(roc_auc_score(y_true_binary, np.array(y_prob), average="macro", multi_class="ovr"))
    elif len(class_names) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, np.array(y_prob)[:, 1]))
    else:
        metrics["roc_auc"] = float("nan")

    outputs_dir = Path(paths_cfg["output_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "evaluation_report.txt"
    cm_path = outputs_dir / "confusion_matrix.png"

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("Evaluation Summary\n")
        report_file.write("===================\n")
        report_file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        report_file.write(f"Precision: {metrics['precision']:.4f}\n")
        report_file.write(f"Recall: {metrics['recall']:.4f}\n")
        report_file.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        report_file.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
        report_file.write("Classification Report:\n")
        report_file.write(metrics["classification_report"])

    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path)
    print(f"Saved evaluation report to {report_path}")
    print(f"Saved confusion matrix to {cm_path}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained defect detection model.")
    parser.add_argument("--config", type=Path, default=Path("configs.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="resnet50")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_evaluation(config, args.checkpoint, args.model_name)
