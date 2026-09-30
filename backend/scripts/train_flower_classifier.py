"""Train a ResNet-18 classifier on the TF Flowers dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
DATA_DIR = ROOT / "backend" / "data" / "datasets"
DATASET_DIR = DATA_DIR / "flower_photos"
ARCHIVE_PATH = DATA_DIR / "flower_photos.tgz"
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
MODEL_DIR = ROOT / "backend" / "data" / "models" / "classifier"
METRICS_PATH = ROOT / "backend" / "data" / "metrics" / "classifier_metrics.json"
SPLIT_PATH = MODEL_DIR / "split.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    return parser.parse_args()


def ensure_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_DIR.is_dir():
        if not ARCHIVE_PATH.is_file():
            print(f"Downloading TF Flowers from {DATASET_URL}")
            urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
            root = DATA_DIR.resolve()
            for member in archive.getmembers():
                target = (DATA_DIR / member.name).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"Unsafe path in dataset archive: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"Links are not allowed in dataset archive: {member.name}")
            archive.extractall(DATA_DIR)
    if not any(DATASET_DIR.glob("*/")):
        raise RuntimeError(f"No class folders found under {DATASET_DIR}")


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_count: int) -> dict:
    model.eval()
    confusion = torch.zeros((class_count, class_count), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            for actual, predicted in zip(labels, predictions):
                confusion[actual, predicted] += 1

    true_positives = confusion.diag().float()
    precision = true_positives / confusion.sum(dim=0).clamp_min(1)
    recall = true_positives / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    total = int(confusion.sum())
    return {
        "accuracy": float(true_positives.sum() / max(total, 1)),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(confusion[index].sum()),
            }
            for index in range(class_count)
        ],
        "confusion_matrix": confusion.tolist(),
        "sample_count": total,
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("epochs and batch-size must be positive; workers cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_dataset()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    base_dataset = datasets.ImageFolder(DATASET_DIR)
    classes = base_dataset.classes
    if len(classes) != 5:
        raise ValueError(f"Expected the five TF Flowers classes, found: {classes}")

    from app.classifier.resnet import build_resnet18
    from app.classifier.training import stratified_split

    train_indices, validation_indices, test_indices = stratified_split(base_dataset.samples, seed=args.seed)
    if args.max_train_per_class:
        rng = random.Random(args.seed)
        selected: list[int] = []
        for class_index in range(len(classes)):
            class_indices = [index for index in train_indices if base_dataset.targets[index] == class_index]
            rng.shuffle(class_indices)
            selected.extend(class_indices[: args.max_train_per_class])
        train_indices = selected

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "classes": classes,
                "train": [base_dataset.samples[i][0] for i in train_indices],
                "validation": [base_dataset.samples[i][0] for i in validation_indices],
                "test": [base_dataset.samples[i][0] for i in test_indices],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_transform, eval_transform = make_transforms()
    train_dataset = Subset(datasets.ImageFolder(DATASET_DIR, transform=train_transform), train_indices)
    validation_dataset = Subset(datasets.ImageFolder(DATASET_DIR, transform=eval_transform), validation_indices)
    test_dataset = Subset(datasets.ImageFolder(DATASET_DIR, transform=eval_transform), test_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=args.epochs * len(train_loader)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_validation_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            sample_count += len(labels)

        validation = evaluate(model, validation_loader, device, len(classes))
        train_accuracy = correct / max(sample_count, 1)
        average_loss = total_loss / max(sample_count, 1)
        print(
            f"Epoch {epoch}/{args.epochs}: loss={average_loss:.4f}, "
            f"train_accuracy={train_accuracy:.4f}, "
            f"validation_accuracy={validation['accuracy']:.4f}, "
            f"validation_macro_f1={validation['macro_f1']:.4f}"
        )
        if validation["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation["accuracy"]
            torch.save(model.state_dict(), MODEL_DIR / "model.pt")
            (MODEL_DIR / "classes.json").write_text(
                json.dumps(classes, indent=2), encoding="utf-8"
            )

    model.load_state_dict(
        torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, device, len(classes))
    test_metrics.update(
        {
            "model": "resnet18-imagenet1k-v1",
            "dataset": "TF Flowers",
            "classes": classes,
            "seed": args.seed,
            "epochs": args.epochs,
            "best_validation_accuracy": best_validation_accuracy,
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(validation_dataset),
        }
    )
    METRICS_PATH.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"Test accuracy={test_metrics['accuracy']:.4f}; macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"Saved weights to {MODEL_DIR / 'model.pt'} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
