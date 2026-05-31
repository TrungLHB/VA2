"""Train RGB or optical-flow TSN on miniUCF."""

from __future__ import annotations

from datetime import datetime
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

TASK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_ROOT.parents[0]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import FLOW, FUSION, RGB, MiniUCFFlowDataset, MiniUCFRGBDataset, MiniUCFFusionDataset
from TSN.model import FLOW_STACK_SIZE, LateFusionTSN, NUM_SEGMENTS, flow_tsn, rgb_tsn


MODALITY = os.environ.get("MODALITY", RGB)  # RGB, FLOW, or FUSION
USE_IMAGENET_INIT = bool(int(os.environ.get("USE_IMAGENET_INIT", "0")))

EPOCHS = int(os.environ.get("EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(PROJECT_ROOT / "TSN" / "checkpoints")
TENSORBOARD_DIR = Path(PROJECT_ROOT / "TSN" / "runs")
RGB_FUSION_CHECKPOINT = Path(os.environ.get("RGB_CHECKPOINT", CHECKPOINT_DIR / "rgb_imagenet.pt"))
FLOW_FUSION_CHECKPOINT = Path(os.environ.get("FLOW_CHECKPOINT", CHECKPOINT_DIR / "flow_imagenet.pt"))


def build_dataset(split: str):
    if MODALITY == RGB:
        return MiniUCFRGBDataset(
            split=split,
            num_segments=NUM_SEGMENTS,
        )

    if MODALITY == FLOW:
        return MiniUCFFlowDataset(
            split=split,
            num_segments=NUM_SEGMENTS,
            flow_stack_size=FLOW_STACK_SIZE,
        )

    if MODALITY == FUSION:
        return MiniUCFFusionDataset(
            split=split,
            num_segments=NUM_SEGMENTS,
            flow_stack_size=FLOW_STACK_SIZE,
        )

    raise ValueError(f"Unknown MODALITY: {MODALITY}")


def build_train_dataset():
    return build_dataset(split="train")


def build_validation_dataset():
    return build_dataset(split="validation")


def build_model() -> nn.Module:
    if MODALITY == RGB:
        return rgb_tsn(pretrained=USE_IMAGENET_INIT)
    if MODALITY == FLOW:
        return flow_tsn(pretrained=USE_IMAGENET_INIT)
    if MODALITY == FUSION:
        return build_fusion_model()
    raise ValueError(f"Unknown MODALITY: {MODALITY}")


def build_fusion_model() -> LateFusionTSN:
    rgb_model = rgb_tsn(pretrained=False).to(DEVICE)
    flow_model = flow_tsn(pretrained=False).to(DEVICE)
    load_checkpoint_from_path(rgb_model, RGB_FUSION_CHECKPOINT)
    load_checkpoint_from_path(flow_model, FLOW_FUSION_CHECKPOINT)
    return LateFusionTSN(rgb_model, flow_model).to(DEVICE)


def make_data_loader(dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (clips, labels) in enumerate(loader):
        if isinstance(clips, (tuple, list)):
            clips = tuple(clip.to(DEVICE) for clip in clips)
        else:
            clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(*clips) if isinstance(clips, tuple) else model(clips)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def checkpoint_path() -> Path:
    init_name = "imagenet" if USE_IMAGENET_INIT else "random"
    return CHECKPOINT_DIR / f"{MODALITY}_{init_name}.pt"


def experiment_name() -> str:
    init_name = "imagenet" if USE_IMAGENET_INIT else "random"
    default_name = f"{MODALITY}_{init_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return os.environ.get("RUN_NAME", default_name)


def create_tensorboard_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("Install tensorboard, or set USE_TENSORBOARD=0 to disable logging.") from exc

    log_dir = TENSORBOARD_DIR / experiment_name()
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(
        "config",
        "\n".join(
            [
                f"modality: {MODALITY}",
                f"use_imagenet_init: {USE_IMAGENET_INIT}",
                f"epochs: {EPOCHS}",
                f"batch_size: {BATCH_SIZE}",
                f"learning_rate: {LEARNING_RATE}",
                f"weight_decay: {WEIGHT_DECAY}",
                f"checkpoint: {checkpoint_path()}",
            ]
        ),
        global_step=0,
    )
    print(f"TensorBoard logs: {log_dir}")
    return writer


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


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    path = checkpoint_path()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch = int(checkpoint.get("epoch", 0))
    print(f"Loaded checkpoint from epoch {epoch}: {path}")
    return epoch


def load_checkpoint_from_path(model: nn.Module, path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    epoch = int(checkpoint.get("epoch", 0))
    print(f"Loaded checkpoint from epoch {epoch}: {path}")
    return epoch


def main() -> None:
    if MODALITY == FUSION:
        fusion_model = build_fusion_model()
        validation_dataset = build_validation_dataset()
        validation_loader = make_data_loader(validation_dataset, shuffle=False)
        fusion_loss, fusion_accuracy = evaluate(fusion_model, validation_loader, nn.NLLLoss())
        print(f"Late fusion validation: loss {fusion_loss:.4f}, acc {fusion_accuracy:.3f}")
        return

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    writer = create_tensorboard_writer()

    try:
        train_dataset = build_train_dataset()
        train_loader = make_data_loader(train_dataset, shuffle=True)

        validation_dataset = build_validation_dataset()
        validation_loader = make_data_loader(validation_dataset, shuffle=False)

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
            print(f"Epoch {epoch:03d}: train loss {train_loss:.4f}, train acc {train_accuracy:.3f}")
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)

            validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion)
            print(
                f"Epoch {epoch:03d}: validation loss {validation_loss:.4f}, "
                f"validation acc {validation_accuracy:.3f}"
            )
            if writer is not None:
                writer.add_scalar("Loss/validation", validation_loss, epoch)
                writer.add_scalar("Accuracy/validation", validation_accuracy, epoch)

            save_checkpoint(model, optimizer, epoch)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
