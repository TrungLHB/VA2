"""Dataset utilities for the miniUCF action-recognition subset.

The dataset can read either extracted RGB frames or the provided optical-flow
JPEGs. RGB frames are intentionally stored outside the AVI tree so extraction is
done once and subsequent training runs only load images from disk.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


RGB = "rgb"
FLOW = "flow"
TCHW = "TCHW"
CTHW = "CTHW"

DATA_ROOT = Path("data")
CLASSES_FILE = DATA_ROOT / "classes.txt"
TRAIN_SPLIT_FILE = DATA_ROOT / "train.txt"
VALIDATION_SPLIT_FILE = DATA_ROOT / "validation.txt"
VIDEO_ROOT = DATA_ROOT / "mini_UCF"
RGB_FRAMES_ROOT = DATA_ROOT / "mini_UCF_frames"
FLOW_ROOT = DATA_ROOT / "mini_UCF_flow"
DEFAULT_IMAGE_TEMPLATE = "img_{:05d}.jpg"


@dataclass(frozen=True)
class VideoRecord:
    """Metadata for one video listed in train.txt or validation.txt."""

    identifier: str
    class_name: str
    video_name: str
    label: int


def read_class_mapping(classes_file: str | Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read ``classes.txt`` into both name-to-id and id-to-name mappings."""

    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    with Path(classes_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            class_id, class_name = line.split(maxsplit=1)
            class_id_int = int(class_id)
            name_to_id[class_name] = class_id_int
            id_to_name[class_id_int] = class_name
    return name_to_id, id_to_name


def read_split_file(split_file: str | Path, class_to_idx: Dict[str, int]) -> List[VideoRecord]:
    """Read a miniUCF split file into ``VideoRecord`` objects."""

    records: List[VideoRecord] = []
    with Path(split_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            identifier = line.strip()
            if not identifier:
                continue
            class_name, video_name = identifier.split("/", maxsplit=1)
            if class_name not in class_to_idx:
                raise ValueError(f"Unknown class {class_name!r} in split file {split_file}")
            records.append(
                VideoRecord(
                    identifier=identifier,
                    class_name=class_name,
                    video_name=video_name,
                    label=class_to_idx[class_name],
                )
            )
    return records


def extract_rgb_frames(
    split_files: Optional[Sequence[str | Path]] = None,
    overwrite: bool = False,
) -> None:
    """Extract RGB frames from AVI files using OpenCV.

    Args:
        split_files: Optional split files limiting extraction to listed videos.
            If omitted, every ``*.avi`` below ``data/mini_UCF`` is extracted.
        overwrite: Re-extract videos that already have at least one frame.

    Raises:
        ImportError: If OpenCV is not installed.
        RuntimeError: If a video cannot be opened.
    """

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install opencv-python to extract RGB frames from AVI files.") from exc

    if split_files is None:
        video_paths = sorted(VIDEO_ROOT.glob("*/*.avi"))
    else:
        identifiers: List[str] = []
        for split_file in split_files:
            with Path(split_file).open("r", encoding="utf-8") as handle:
                identifiers.extend(line.strip() for line in handle if line.strip())
        video_paths = [VIDEO_ROOT / f"{identifier}.avi" for identifier in sorted(set(identifiers))]

    for video_path in video_paths:
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file: {video_path}")

        relative_video = video_path.relative_to(VIDEO_ROOT)
        output_dir = RGB_FRAMES_ROOT / relative_video.with_suffix("")
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
            output_path = output_dir / DEFAULT_IMAGE_TEMPLATE.format(frame_index)
            cv2.imwrite(str(output_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            frame_index += 1
            success, frame_bgr = capture.read()
        capture.release()


class MiniUCFDataset(Dataset):
    """PyTorch dataset for RGB-frame or optical-flow miniUCF clips.

    Each item is ``(clip, label)``. By default, ``clip`` has shape
    ``[T, C, H, W]`` where ``T`` is ``num_segments`` and ``C`` is 3 for RGB or
    2 for optical flow. Set ``output_format="CTHW"`` for 3D CNNs.
    """

    def __init__(
        self,
        split: str = "train",
        modality: str = RGB,
        num_segments: int = 8,
        transform: Optional[Callable] = None,
        tensor_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        output_format: str = TCHW,
        auto_extract_rgb: bool = False,
    ) -> None:
        if modality not in {RGB, FLOW}:
            raise ValueError(f"modality must be {RGB!r} or {FLOW!r}, got {modality!r}")
        if output_format not in {TCHW, CTHW}:
            raise ValueError(f"output_format must be {TCHW!r} or {CTHW!r}, got {output_format!r}")
        if num_segments <= 0:
            raise ValueError("num_segments must be positive")

        self.split = split
        self.modality = modality
        self.num_segments = num_segments
        self.transform = transform
        self.tensor_transform = tensor_transform
        self.output_format = output_format
        self.random_sampling = split == "train"

        if split == "train":
            split_path = TRAIN_SPLIT_FILE
        elif split in {"validation", "val"}:
            split_path = VALIDATION_SPLIT_FILE
        else:
            raise ValueError("split must be 'train' or 'validation'")

        self.class_to_idx, self.idx_to_class = read_class_mapping(CLASSES_FILE)
        self.records = read_split_file(split_path, self.class_to_idx)

        if self.modality == RGB and auto_extract_rgb:
            extract_rgb_frames(split_files=[split_path])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record = self.records[index]
        frame_count = self._frame_count(record)
        indices = self._sample_indices(frame_count)

        if self.modality == RGB:
            frames = [self._load_rgb_frame(record, frame_index) for frame_index in indices]
            clip = torch.stack([self._image_to_tensor(frame, apply_transform=True) for frame in frames], dim=0)
        else:
            flow_frames = [self._load_flow_frame(record, frame_index) for frame_index in indices]
            clip = torch.stack(flow_frames, dim=0)

        if self.output_format == CTHW:
            clip = clip.permute(1, 0, 2, 3).contiguous()
        if self.tensor_transform is not None:
            clip = self.tensor_transform(clip)
        return clip, record.label

    def _frame_count(self, record: VideoRecord) -> int:
        if self.modality == RGB:
            frame_dir = RGB_FRAMES_ROOT / record.identifier
            frame_paths = sorted(frame_dir.glob("*.jpg"))
            if frame_paths:
                return len(frame_paths)
            video_path = VIDEO_ROOT / f"{record.identifier}.avi"
            raise FileNotFoundError(
                f"No extracted RGB frames found in {frame_dir}. "
                f"Run extract_rgb_frames() "
                f"or construct MiniUCFDataset(..., auto_extract_rgb=True). "
                f"Expected source video: {video_path}"
            )

        flow_dir = FLOW_ROOT / record.identifier
        flow_x_count = len(list(flow_dir.glob("flow_x_*.jpg")))
        flow_y_count = len(list(flow_dir.glob("flow_y_*.jpg")))
        frame_count = min(flow_x_count, flow_y_count)
        if frame_count == 0:
            raise FileNotFoundError(f"No optical flow frames found in {flow_dir}")
        return frame_count

    def _sample_indices(self, frame_count: int) -> List[int]:
        """Return 1-based frame indices for one clip."""

        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.num_segments == 1:
            if self.random_sampling:
                return [random.randint(1, frame_count)]
            return [(frame_count + 1) // 2]

        if frame_count < self.num_segments:
            return [
                min(frame_count, int(round(i * (frame_count - 1) / (self.num_segments - 1))) + 1)
                for i in range(self.num_segments)
            ]

        segment_size = frame_count / float(self.num_segments)
        indices: List[int] = []
        for segment_index in range(self.num_segments):
            start = int(round(segment_size * segment_index)) + 1
            end = int(round(segment_size * (segment_index + 1)))
            end = max(start, min(end, frame_count))
            if self.random_sampling:
                indices.append(random.randint(start, end))
            else:
                indices.append((start + end) // 2)
        return indices

    def _load_rgb_frame(self, record: VideoRecord, frame_index: int) -> Image.Image:
        frame_dir = RGB_FRAMES_ROOT / record.identifier
        frame_path = frame_dir / f"img_{frame_index:05d}.jpg"
        if not frame_path.exists():
            fallback = frame_dir / f"frame_{frame_index:05d}.jpg"
            frame_path = fallback if fallback.exists() else frame_path
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing RGB frame: {frame_path}")
        return Image.open(frame_path).convert("RGB")

    def _load_flow_frame(self, record: VideoRecord, frame_index: int) -> torch.Tensor:
        flow_dir = FLOW_ROOT / record.identifier
        x_path = flow_dir / f"flow_x_{frame_index:04d}.jpg"
        y_path = flow_dir / f"flow_y_{frame_index:04d}.jpg"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing optical flow pair: {x_path}, {y_path}")

        flow_x = Image.open(x_path).convert("L")
        flow_y = Image.open(y_path).convert("L")

        if self.transform is not None:
            flow_x = self.transform(flow_x)
            flow_y = self.transform(flow_y)

        return torch.cat(
            [
                self._image_to_tensor(flow_x, apply_transform=False),
                self._image_to_tensor(flow_y, apply_transform=False),
            ],
            dim=0,
        )

    def _image_to_tensor(self, image, apply_transform: bool) -> torch.Tensor:
        if apply_transform and self.transform is not None and isinstance(image, Image.Image):
            image = self.transform(image)
        if torch.is_tensor(image):
            return image.float()
        return F.to_tensor(image)


__all__ = [
    "CTHW",
    "CLASSES_FILE",
    "DATA_ROOT",
    "DEFAULT_IMAGE_TEMPLATE",
    "FLOW",
    "FLOW_ROOT",
    "RGB",
    "RGB_FRAMES_ROOT",
    "TCHW",
    "TRAIN_SPLIT_FILE",
    "VALIDATION_SPLIT_FILE",
    "VIDEO_ROOT",
    "MiniUCFDataset",
    "VideoRecord",
    "extract_rgb_frames",
    "read_class_mapping",
    "read_split_file",
]
