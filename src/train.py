from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import build_dataloaders, get_class_labels
from src.models.metrics import compute_metrics
from src.models.model_factory import create_model, save_checkpoint, freeze_backbone, unfreeze_backbone
from src.utils import get_device, load_config, set_seed


class EarlyStopping:
    def __init__(self, patience: int = 4, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_class_weights(dataset: nn.Module) -> torch.Tensor:
    target_counts = Counter(dataset.targets)
    weights = [0.0] * len(target_counts)
    for class_index, count in target_counts.items():
        weights[class_index] = 1.0 / float(count)
    return torch.tensor(weights, dtype=torch.float32)


def train(config: dict, model_name: str | None = None) -> None:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    paths_cfg = config["paths"]

    model_name = model_name or training_cfg.get("model_name", "resnet50")
    processed_root = Path(dataset_cfg["processed_root"])
    image_size = int(dataset_cfg["image_size"])
    batch_size = int(training_cfg["batch_size"])
    epochs = int(training_cfg["epochs"])
    learning_rate = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg["weight_decay"])
    seed = int(dataset_cfg.get("random_seed", 42))
    device = get_device(training_cfg.get("device", "cuda"))
    balance_classes = bool(training_cfg.get("balance_classes", True))
    mixed_precision = bool(training_cfg.get("mixed_precision", True)) and device.type == "cuda"
    freeze_epochs = int(training_cfg.get("freeze_backbone_epochs", 0))
    unfreeze_epoch = freeze_epochs + 1
    early_stopping = EarlyStopping(patience=int(training_cfg.get("early_stopping_patience", 4)))
    num_workers = min(8, os.cpu_count() or 0)

    set_seed(seed)

    train_loader, val_loader, _, train_dataset, _, _ = build_dataloaders(
        processed_root,
        image_size,
        batch_size,
        num_workers=num_workers,
        balance_classes=balance_classes,
    )
    class_names, _ = get_class_labels(train_dataset)
    num_classes = len(class_names)

    model = create_model(model_name, num_classes=num_classes, pretrained=bool(training_cfg.get("pretrained", True)))
    if freeze_epochs > 0:
        freeze_backbone(model)
    model = model.to(device)

    class_weights = None
    if bool(training_cfg.get("use_class_weights", False)):
        class_weights = build_class_weights(train_dataset).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7, verbose=True)
    scaler = GradScaler(enabled=mixed_precision)

    checkpoint_dir = Path(paths_cfg["model_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
    final_checkpoint_path = checkpoint_dir / f"{model_name}_final.pth"

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch and freeze_epochs > 0:
            unfreeze_backbone(model)
            optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay)

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=mixed_precision):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if float(training_cfg.get("gradient_clip_norm", 0.0)) > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=mixed_precision):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        val_metrics = compute_metrics(all_labels, all_preds, class_names)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| accuracy={val_metrics['accuracy']:.4f} | f1={val_metrics['f1_score']:.4f} "
            f"| elapsed={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                str(best_checkpoint_path),
                metadata={
                    "epoch": epoch,
                    "class_names": class_names,
                    "model_name": model_name,
                    "config": config,
                },
                optimizer_state=optimizer.state_dict(),
                scaler_state=scaler.state_dict() if mixed_precision else None,
            )
            print(f"Saved best model to {best_checkpoint_path}")

        if early_stopping.step(val_loss):
            print(f"Early stopping after {epoch} epochs due to no improvement.")
            break

    save_checkpoint(
        model,
        str(final_checkpoint_path),
        metadata={
            "epoch": epoch,
            "class_names": class_names,
            "model_name": model_name,
            "config": config,
        },
        optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict() if mixed_precision else None,
    )
    print(f"Training complete. Final model saved to {final_checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train defect detection models.")
    parser.add_argument("--config", type=Path, default=Path("configs.yaml"))
    parser.add_argument("--model_name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, model_name=args.model_name)
