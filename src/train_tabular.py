from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DROP_COLUMNS = {"defect_id", "defect_date"}


def load_tabular_data(csv_path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found. Available columns: {list(data.columns)}")

    if "defect_date" in data.columns:
        defect_date = pd.to_datetime(data["defect_date"], errors="coerce")
        data["defect_month"] = defect_date.dt.month.fillna(0).astype(int)
        data["defect_dayofweek"] = defect_date.dt.dayofweek.fillna(0).astype(int)

    features = data.drop(columns=[target])
    drop_columns = [column for column in DEFAULT_DROP_COLUMNS if column in features.columns]
    features = features.drop(columns=drop_columns)
    labels = data[target]

    return features, labels


def build_pipeline(features: pd.DataFrame, seed: int) -> Pipeline:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_tabular(
    csv_path: Path,
    target: str,
    model_path: Path,
    report_path: Path,
    test_size: float,
    seed: int,
) -> None:
    features, labels = load_tabular_data(csv_path, target)
    stratify = labels if labels.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    pipeline = build_pipeline(x_train, seed)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    labels_sorted = sorted(labels.unique().tolist())
    metrics = {
        "target": target,
        "rows": int(len(features)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": features.columns.tolist(),
        "classes": labels_sorted,
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels_sorted).tolist(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "target": target,
            "features": features.columns.tolist(),
            "classes": labels_sorted,
        },
        model_path,
    )
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved model to {model_path}")
    print(f"Saved report to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular defect classifier from a CSV file.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", type=str, default="defect_type")
    parser.add_argument("--model-path", type=Path, default=Path("models/tabular_defect_model.joblib"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/tabular_training_report.json"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tabular(
        csv_path=args.csv,
        target=args.target,
        model_path=args.model_path,
        report_path=args.report_path,
        test_size=args.test_size,
        seed=args.seed,
    )
