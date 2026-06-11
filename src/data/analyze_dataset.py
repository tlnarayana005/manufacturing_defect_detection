from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def collect_image_paths(raw_root: Path, category: str) -> list[Path]:
    category_root = raw_root / category
    if not category_root.exists():
        raise FileNotFoundError(f"Category folder not found: {category_root}")

    paths: list[Path] = []
    for image_path in category_root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            paths.append(image_path)
    return sorted(paths)


def compute_class_distribution(raw_root: Path, category: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for image_path in collect_image_paths(raw_root, category):
        counts[image_path.parent.name] += 1
    return dict(counts)


def compute_image_hash(image_path: Path) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        return hashlib.sha256(image.tobytes()).hexdigest()


def detect_duplicates(raw_root: Path, category: str) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    label_conflicts: dict[str, list[Path]] = defaultdict(list)

    for image_path in collect_image_paths(raw_root, category):
        image_hash = compute_image_hash(image_path)
        hash_to_paths[image_hash].append(image_path)

    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    for paths in duplicates.values():
        labels = {p.parent.name for p in paths}
        if len(labels) > 1:
            label_conflicts[image_hash] = paths

    return duplicates, label_conflicts


def run_analysis(raw_root: Path, category: str) -> None:
    distribution = compute_class_distribution(raw_root, category)
    print("Class distribution:")
    for label, count in sorted(distribution.items(), key=lambda item: item[1], reverse=True):
        print(f"  {label}: {count}")

    duplicates, label_conflicts = detect_duplicates(raw_root, category)
    print(f"\nTotal duplicate image groups: {len(duplicates)}")
    print(f"Potential noisy label groups: {len(label_conflicts)}")

    if label_conflicts:
        print("\nPotential noisy labels detected for the following duplicate image groups:")
        for image_hash, paths in label_conflicts.items():
            print(f"  Hash: {image_hash}")
            for path in paths:
                print(f"    {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw defect dataset class distribution and duplicate images.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--category", type=str, default="metal_nut")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.raw_root, args.category)
