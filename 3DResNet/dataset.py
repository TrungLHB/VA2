"""RGB clip dataset for the 3D ResNet task."""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as f

from miniUCF.dataset import (
    RGB_FRAME_TEMPLATE,
    RGB_FRAMES_ROOT,
    VideoRecord,
    extract_rgb_frames,
    read_class_mapping,
    read_split_file,
    split_file_for,
)


IMAGE_SIZE = 112
RESIZE_SIZE = 128
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


class TrainClipTransform:
    """Apply the same random spatial crop and flip to every frame in a clip."""

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        resized = [f.resize(frame, RESIZE_SIZE) for frame in frames]
        crop_params = transforms.RandomCrop.get_params(resized[0], output_size=(IMAGE_SIZE, IMAGE_SIZE))
        flip = random.random() < 0.5

        tensors = []
        for frame in resized:
            frame = f.crop(frame, *crop_params)
            if flip:
                frame = f.hflip(frame)
            tensor = f.to_tensor(frame)
            tensor = f.normalize(tensor, mean=RGB_MEAN, std=RGB_STD)
            tensors.append(tensor)

        return torch.stack(tensors, dim=1)


class EvalClipTransform:
    """Apply deterministic resize and center crop to every frame in a clip."""

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for frame in frames:
            frame = f.resize(frame, RESIZE_SIZE)
            frame = f.center_crop(frame, (IMAGE_SIZE, IMAGE_SIZE))
            tensor = f.to_tensor(frame)
            tensor = f.normalize(tensor, mean=RGB_MEAN, std=RGB_STD)
            tensors.append(tensor)

        return torch.stack(tensors, dim=1)


class MiniUCFRGBClipDataset(Dataset):
    """Load contiguous RGB clips for 3D CNN training and multi-view testing."""

    def __init__(
        self,
        split: str,
        clip_length: int,
        transform,
        num_test_views: int = 1,
        extract_missing: bool = True,
    ) -> None:
        if clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if num_test_views <= 0:
            raise ValueError("num_test_views must be positive")

        self.split = split
        self.clip_length = clip_length
        self.transform = transform
        self.num_test_views = num_test_views
        self.training = split == "train"

        class_to_idx, _ = read_class_mapping()
        self.records = read_split_file(split_file_for(split), class_to_idx)

        if extract_missing:
            self._extract_missing_frames()

        self.frame_counts = {record.identifier: self._frame_count(record) for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record = self.records[index]
        frame_count = self.frame_counts[record.identifier]

        if self.training:
            indices = self._train_indices(frame_count)
            clip = self._load_clip(record, indices)
            return self.transform(clip), record.label

        views = []
        for start_index in self._test_start_indices(frame_count):
            indices = self._clip_indices_from_start(start_index, frame_count)
            views.append(self.transform(self._load_clip(record, indices)))

        return torch.stack(views, dim=0), record.label

    def _extract_missing_frames(self) -> None:
        missing_records = [
            record
            for record in self.records
            if not any((RGB_FRAMES_ROOT / record.identifier).glob("*.jpg"))
        ]
        if missing_records:
            print(f"Extracting RGB frames for {len(missing_records)} missing {self.split} videos...")
            extract_rgb_frames(split_files=[split_file_for(self.split)], overwrite=False)

    @staticmethod
    def _frame_count(record: VideoRecord) -> int:
        frame_dir = RGB_FRAMES_ROOT / record.identifier
        frame_count = len(list(frame_dir.glob("*.jpg")))
        if frame_count == 0:
            raise FileNotFoundError(f"No extracted RGB frames found in {frame_dir}")
        return frame_count

    def _train_indices(self, frame_count: int) -> List[int]:
        if frame_count >= self.clip_length:
            max_start = frame_count - self.clip_length + 1
            start_index = random.randint(1, max_start)
            return self._clip_indices_from_start(start_index, frame_count)

        return self._spread_indices(frame_count)

    def _test_start_indices(self, frame_count: int) -> List[int]:
        if frame_count < self.clip_length:
            return [1 for _ in range(self.num_test_views)]

        max_start = frame_count - self.clip_length + 1
        if self.num_test_views == 1:
            return [(max_start + 1) // 2]

        return [
            int(round(view_index * (max_start - 1) / (self.num_test_views - 1))) + 1
            for view_index in range(self.num_test_views)
        ]

    def _clip_indices_from_start(self, start_index: int, frame_count: int) -> List[int]:
        return [min(start_index + offset, frame_count) for offset in range(self.clip_length)]

    def _spread_indices(self, frame_count: int) -> List[int]:
        if self.clip_length == 1:
            return [1]
        return [
            int(round(index * (frame_count - 1) / (self.clip_length - 1))) + 1
            for index in range(self.clip_length)
        ]

    @staticmethod
    def _load_clip(record: VideoRecord, indices: List[int]) -> List[Image.Image]:
        frame_dir = RGB_FRAMES_ROOT / record.identifier
        frames = []
        for frame_index in indices:
            frame_path = frame_dir / RGB_FRAME_TEMPLATE.format(frame_index)
            if not frame_path.exists():
                raise FileNotFoundError(f"Missing RGB frame: {frame_path}")
            frames.append(Image.open(frame_path).convert("RGB"))
        return frames


__all__ = [
    "EvalClipTransform",
    "IMAGE_SIZE",
    "MiniUCFRGBClipDataset",
    "RESIZE_SIZE",
    "RGB_MEAN",
    "RGB_STD",
    "TrainClipTransform",
]
