import argparse
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _scan_images(image_dir: Path, label: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for image_path in image_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            rows.append({"image_path": str(image_path), "label": label})
    return rows


def collect_images(raw_root: Path, category: str) -> pd.DataFrame:
    category_root = raw_root / category
    if not category_root.exists():
        raise FileNotFoundError(f"Category folder not found: {category_root}")

    rows: List[Dict[str, str]] = []
    train_root = category_root / "train"
    test_root = category_root / "test"

    if train_root.exists() and test_root.exists():
        # MVTec AD-style layout for supervised defect classification
        if (train_root / "good").exists():
            rows.extend(_scan_images(train_root / "good", "good"))
        if (test_root / "good").exists():
            rows.extend(_scan_images(test_root / "good", "good"))
        for defect_folder in test_root.iterdir():
            if defect_folder.is_dir() and defect_folder.name.lower() != "good":
                rows.extend(_scan_images(defect_folder, defect_folder.name))
    else:
        for subset in category_root.glob("**/*"):
            if subset.is_file() and subset.suffix.lower() in VALID_EXTENSIONS:
                rows.append({"image_path": str(subset), "label": subset.parent.name})

    if not rows:
        raise ValueError(f"No images found under {category_root}")

    df = pd.DataFrame(rows)
    if df["label"].isna().any():
        raise ValueError("Detected unlabeled images during dataset collection.")

    return df


def _safe_split(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    if len(df) < 2 or df["label"].nunique() < 2:
        return df.sample(frac=1, random_state=kwargs.get("random_state", 42)).reset_index(drop=True)
    if df["label"].value_counts().min() < 2:
        return df.sample(frac=1, random_state=kwargs.get("random_state", 42)).reset_index(drop=True)
    return train_test_split(df, **kwargs)[0]


def create_split_frames(
    df: pd.DataFrame,
    val_split: float,
    test_split: float,
    seed: int,
    clean: bool = False,
) -> Dict[str, pd.DataFrame]:
    if not 0.0 < val_split < 1.0 or not 0.0 < test_split < 1.0:
        raise ValueError("val_split and test_split must be between 0 and 1.")

    stratify = df["label"] if df["label"].nunique() > 1 and df["label"].value_counts().min() > 1 else None
    train_val, test = train_test_split(
        df,
        test_size=test_split,
        stratify=stratify,
        random_state=seed,
        shuffle=True,
    )
    val_ratio = val_split / (1.0 - test_split)
    stratify = train_val["label"] if train_val["label"].nunique() > 1 and train_val["label"].value_counts().min() > 1 else None
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=stratify,
        random_state=seed,
        shuffle=True,
    )

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def prepare_image(source_path: Path, destination_path: Path, image_size: int) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.LANCZOS)
        image.save(destination_path)


def copy_split_data(split_frames: Dict[str, pd.DataFrame], processed_root: Path, image_size: int) -> None:
    for split_name, frame in split_frames.items():
        for _, row in frame.iterrows():
            label = row["label"]
            src = Path(row["image_path"])
            dst = processed_root / split_name / label / src.name
            prepare_image(src, dst, image_size)


def prepare_mvtec_dataset(
    raw_root: Path,
    processed_root: Path,
    category: str,
    image_size: int,
    val_split: float,
    test_split: float,
    seed: int,
    clean: bool = False,
) -> None:
    logging.info("Collecting raw images...")
    df = collect_images(raw_root, category)
    logging.info("Class distribution: %s", dict(Counter(df["label"])))

    if clean and processed_root.exists():
        logging.info("Removing existing processed dataset at %s", processed_root)
        shutil.rmtree(processed_root)

    logging.info("Preparing train/val/test splits...")
    split_frames = create_split_frames(df, val_split, test_split, seed)
    logging.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(split_frames["train"]),
        len(split_frames["val"]),
        len(split_frames["test"]),
    )

    logging.info("Copying processed images to %s", processed_root)
    copy_split_data(split_frames, processed_root, image_size)
    logging.info("Dataset preprocessing complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed defect dataset from MVTec AD raw images.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--category", type=str, default="metal_nut")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.12)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove previous processed outputs before building the dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_mvtec_dataset(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        category=args.category,
        image_size=args.image_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        clean=args.clean,
    )
