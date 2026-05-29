"""Train RGB or optical-flow TSN on miniUCF.

This file intentionally uses constants instead of many command-line arguments.
For this part of the exercise, choose the experiment by changing the constants
near the top of the file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miniUCF import FLOW, RGB, MiniUCFFlowDataset, MiniUCFRGBDataset  # noqa: E402
from TSN.model import FLOW_STACK_SIZE, NUM_SEGMENTS, flow_tsn, rgb_tsn  # noqa: E402


# Change these constants for the current experiment.
MODALITY = FLOW  # RGB or FLOW
USE_IMAGENET_INIT = False

EPOCHS = 1
BATCH_SIZE = 2
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Debug setting: set to 0 to train on the whole training split.
MAX_TRAIN_BATCHES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = PROJECT_ROOT / "TSN" / "checkpoints"


def rgb_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def flow_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.226]),
        ]
    )


def build_train_dataset():
    if MODALITY == RGB:
        return MiniUCFRGBDataset(
            split="train",
            num_segments=NUM_SEGMENTS,
            transform=rgb_transforms(),
        )

    if MODALITY == FLOW:
        return MiniUCFFlowDataset(
            split="train",
            num_segments=NUM_SEGMENTS,
            transform=flow_transforms(),
            flow_stack_size=FLOW_STACK_SIZE,
        )

    raise ValueError(f"Unknown MODALITY: {MODALITY}")


def build_model() -> nn.Module:
    if MODALITY == RGB:
        return rgb_tsn(pretrained=USE_IMAGENET_INIT)
    if MODALITY == FLOW:
        return flow_tsn(pretrained=USE_IMAGENET_INIT)
    raise ValueError(f"Unknown MODALITY: {MODALITY}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (clips, labels) in enumerate(loader):
        if MAX_TRAIN_BATCHES and batch_index >= MAX_TRAIN_BATCHES:
            break

        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def checkpoint_path() -> Path:
    init_name = "imagenet" if USE_IMAGENET_INIT else "random"
    return CHECKPOINT_DIR / f"{MODALITY}_{init_name}.pt"


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "modality": MODALITY,
            "use_imagenet_init": USE_IMAGENET_INIT,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def main() -> None:
    dataset = build_train_dataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(model, loader, criterion, optimizer)
        print(f"Epoch {epoch:03d}: train loss {train_loss:.4f}, train acc {train_accuracy:.3f}")
        save_checkpoint(model, optimizer, epoch)


if __name__ == "__main__":
    main()
