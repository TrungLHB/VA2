"""RGB clip dataset for the 3D ResNet task."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as f


IMAGE_SIZE = 112
RESIZE_SIZE = 128
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CLASSES_FILE = DATA_ROOT / "classes.txt"
TRAIN_SPLIT_FILE = DATA_ROOT / "train.txt"
VALIDATION_SPLIT_FILE = DATA_ROOT / "validation.txt"
VIDEO_ROOT = DATA_ROOT / "mini_UCF"
RGB_FRAMES_ROOT = DATA_ROOT / "mini_UCF_frames"

RGB_FRAME_TEMPLATE = "img_{:05d}.jpg"


@dataclass(frozen=True)
class VideoRecord:
    """Metadata for one video listed in train.txt or validation.txt."""

    identifier: str
    class_name: str
    video_name: str
    label: int


def read_class_mapping(classes_file: Path = CLASSES_FILE) -> Tuple[Dict[str, int], Dict[int, str]]:
    class_to_idx: Dict[str, int] = {}
    idx_to_class: Dict[int, str] = {}

    with classes_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            class_id, class_name = line.split(maxsplit=1)
            class_id = int(class_id)
            class_to_idx[class_name] = class_id
            idx_to_class[class_id] = class_name

    return class_to_idx, idx_to_class


def read_split_file(split_file: Path, class_to_idx: Dict[str, int]) -> List[VideoRecord]:
    records: List[VideoRecord] = []

    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            identifier = line.strip()
            if not identifier:
                continue
            class_name, video_name = identifier.split("/", maxsplit=1)
            records.append(
                VideoRecord(
                    identifier=identifier,
                    class_name=class_name,
                    video_name=video_name,
                    label=class_to_idx[class_name],
                )
            )

    return records


def split_file_for(split: str) -> Path:
    if split == "train":
        return TRAIN_SPLIT_FILE
    if split in {"validation", "val"}:
        return VALIDATION_SPLIT_FILE
    raise ValueError("split must be 'train' or 'validation'")


def extract_rgb_frames(
    split_files: Optional[Sequence[Path]] = None,
    overwrite: bool = False,
) -> None:
    """Extract RGB frames from AVI files into data/mini_UCF_frames."""

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install opencv-python to extract RGB frames from AVI files.") from exc

    if split_files is None:
        video_paths = sorted(VIDEO_ROOT.glob("*/*.avi"))
    else:
        identifiers: List[str] = []
        for split_file in split_files:
            with split_file.open("r", encoding="utf-8") as handle:
                identifiers.extend(line.strip() for line in handle if line.strip())
        video_paths = [VIDEO_ROOT / f"{identifier}.avi" for identifier in sorted(set(identifiers))]

    for video_path in video_paths:
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file: {video_path}")

        output_dir = RGB_FRAMES_ROOT / video_path.relative_to(VIDEO_ROOT).with_suffix("")
        if output_dir.exists() and not overwrite and any(output_dir.glob("*.jpg")):
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        frame_index = 1
        success, frame_bgr = capture.read()
        while success:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_path = output_dir / RGB_FRAME_TEMPLATE.format(frame_index)
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            frame_index += 1
            success, frame_bgr = capture.read()

        capture.release()


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
