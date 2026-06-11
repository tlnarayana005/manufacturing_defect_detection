# Model Evaluation Report

## Overview

This report summarizes model performance for the AI-Powered Manufacturing Defect Detection System.

## Evaluation Metrics

- Accuracy: measured on the held-out test split
- Precision: weighted average across defect classes
- Recall: weighted average across defect classes
- F1-score: harmonic mean of precision and recall
- Confusion matrix: visual comparison of true vs predicted categories

## Results

> Update this section with actual run-time values after training and evaluation.

### Example Results

- ResNet50 validation accuracy: 0.92
- EfficientNet-B0 validation accuracy: 0.90
- Best test F1-score: 0.91

## Notes

- Use `src/train.py` to retrain model architectures and generate saved checkpoints.
- Use `src/evaluate.py --checkpoint models/resnet50_best.pth --model_name resnet50` to generate evaluation artifacts.
- The `outputs/confusion_matrix.png` file will contain a visual representation of multi-class performance.
