"""Evaluate one explicitly supplied CIFAR-10 checkpoint without training."""

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

from cifar10_cnn import CIFAR10CNN
from cifar10_data import get_data_loaders


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a checkpoint mapping saved by train.py")
    if checkpoint.get("dataset") != "cifar10":
        raise ValueError(f"Expected a cifar10 checkpoint, got {checkpoint.get('dataset')}")
    if not all(key in checkpoint for key in ("model_state_dict", "epoch", "seed")):
        raise ValueError("Checkpoint is missing model state, epoch or seed")
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("Checkpoint model state must be a mapping")
    return checkpoint


def evaluate_split(dataloader, model, device):
    """Accumulate per-example cross-entropy and correct predictions."""
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses = F.cross_entropy(logits, labels, reduction="none")
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Non-finite evaluation loss")
            loss_sum += losses.double().sum().item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    if not count:
        raise ValueError("Evaluation loader is empty")
    return {
        "loss": loss_sum / count,
        "accuracy_percent": 100 * correct / count,
        "examples": count,
    }


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    downloads = parser.add_mutually_exclusive_group()
    downloads.add_argument("--download", action="store_true", dest="download")
    downloads.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(download=False)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.output and args.output.exists():
        parser.error("output file exists; choose a new file")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected_device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = CIFAR10CNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    _, validation, test = get_data_loaders(
        root=str(args.data_root), batch_size=args.batch_size,
        seed=checkpoint["seed"], download=args.download,
    )
    report = {
        "dataset": "cifar10",
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "seed": checkpoint["seed"],
        "batch_size": args.batch_size,
        "device": str(device),
        "validation": evaluate_split(validation, model, device),
        "test": evaluate_split(test, model, device),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
