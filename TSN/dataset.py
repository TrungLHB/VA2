"""TSN datasets for miniUCF."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor


RGB = "rgb"
FLOW = "flow"
FUSION = "fusion"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CLASSES_FILE = DATA_ROOT / "classes.txt"
TRAIN_SPLIT_FILE = DATA_ROOT / "train.txt"
VALIDATION_SPLIT_FILE = DATA_ROOT / "validation.txt"
VIDEO_ROOT = DATA_ROOT / "mini_UCF"
RGB_FRAMES_ROOT = DATA_ROOT / "mini_UCF_frames"
FLOW_ROOT = DATA_ROOT / "mini_UCF_flow"
TSN_SAMPLE_ROOT = DATA_ROOT / "tsn_samples"

RGB_FRAME_TEMPLATE = "img_{:05d}.jpg"
FLOW_X_TEMPLATE = "flow_x_{:04d}.jpg"
FLOW_Y_TEMPLATE = "flow_y_{:04d}.jpg"


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


def rgb_train_transforms() -> transforms.Compose:
    # Resize the shorter image side to 256, then crop 224x224 because ResNet-18
    # ImageNet models are trained on 224x224 inputs. The mean/std values are the
    # standard ImageNet RGB normalization statistics.
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def rgb_eval_transforms() -> transforms.Compose:
    # Use the same 256 resize and 224x224 crop size as training, but center-crop
    # so validation/testing is deterministic.
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def flow_transforms() -> transforms.Compose:
    # Flow JPEGs are loaded as single-channel images. Pixel values become [0, 1],
    # so mean=0.5 centers them near zero and std=0.226 gives a scale close to
    # ImageNet preprocessing while applying the same normalization to all flow channels.
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.226]),
        ]
    )


class _MiniUCFTSNDataset(Dataset):
    """Shared split parsing and fixed temporal-segment sampling for TSN."""

    def __init__(
        self,
        split: str,
        num_segments: int,
        transform: Optional[Callable],
    ) -> None:
        if num_segments <= 0:
            raise ValueError("num_segments must be positive")

        self.split = split
        self.num_segments = num_segments
        self.transform = transform
        self.random_sampling = split == "train"
        self.class_to_idx, self.idx_to_class = read_class_mapping()
        self.records = read_split_file(split_file_for(split), self.class_to_idx)

    def __len__(self) -> int:
        return len(self.records)

    def _manifest_path(self, modality: str, extra_name: str = "") -> Path:
        sample_name = "random" if self.random_sampling else "middle"
        extra = f"_{extra_name}" if extra_name else ""
        filename = f"{modality}_{self.split}_segments{self.num_segments}_{sample_name}{extra}.json"
        return TSN_SAMPLE_ROOT / filename

    def _load_manifest(self, path: Path) -> Optional[Dict[str, List[int]]]:
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return {identifier: [int(index) for index in indices] for identifier, indices in manifest.items()}

    def _save_manifest(self, path: Path, manifest: Dict[str, List[int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def _sample_indices(self, frame_count: int) -> List[int]:
        """Return 1-based frame indices, one from each temporal segment."""

        if frame_count <= 0:
            raise ValueError("frame_count must be positive")

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

    def _fixed_samples_from_manifest(
        self,
        modality: str,
        counts: Dict[str, int],
        extra_name: str = "",
    ) -> Dict[str, List[int]]:
        path = self._manifest_path(modality=modality, extra_name=extra_name)
        manifest = self._load_manifest(path)
        if manifest is not None:
            return manifest

        manifest = {
            record.identifier: self._sample_indices(counts[record.identifier])
            for record in self.records
        }
        self._save_manifest(path, manifest)
        print(f"Saved fixed TSN samples: {path}")
        return manifest

    def _to_tensor(self, image) -> torch.Tensor:
        if self.transform is not None and isinstance(image, Image.Image):
            image = self.transform(image)
        if torch.is_tensor(image):
            return image.float()
        return to_tensor(image)


class MiniUCFRGBDataset(_MiniUCFTSNDataset):
    """Load one RGB frame from each temporal segment."""

    def __init__(
        self,
        split: str,
        num_segments: int,
        transform: Optional[Callable] = None,
        extract_missing: bool = True,
    ) -> None:
        if transform is None:
            transform = rgb_train_transforms() if split == "train" else rgb_eval_transforms()

        super().__init__(split=split, num_segments=num_segments, transform=transform)

        if extract_missing:
            self._extract_missing_frames()

        frame_counts = {record.identifier: self._frame_count(record) for record in self.records}
        self.sampled_frame_indices = self._fixed_samples_from_manifest(RGB, frame_counts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record = self.records[index]
        frame_indices = self.sampled_frame_indices[record.identifier]

        frames = [self._load_frame(record, frame_index) for frame_index in frame_indices]
        clip = torch.stack([self._to_tensor(frame) for frame in frames], dim=0)
        return clip, record.label

    def _extract_missing_frames(self) -> None:
        missing_records = [
            record
            for record in self.records
            if not any((RGB_FRAMES_ROOT / record.identifier).glob("*.jpg"))
        ]
        if not missing_records:
            return

        print(f"Extracting RGB frames for {len(missing_records)} missing {self.split} videos...")
        extract_rgb_frames(split_files=[split_file_for(self.split)], overwrite=False)

    @staticmethod
    def _frame_count(record: VideoRecord) -> int:
        frame_dir = RGB_FRAMES_ROOT / record.identifier
        frame_count = len(list(frame_dir.glob("*.jpg")))
        if frame_count == 0:
            video_path = VIDEO_ROOT / f"{record.identifier}.avi"
            raise FileNotFoundError(
                f"No extracted RGB frames found in {frame_dir}. "
                f"Run extract_rgb_frames() first. Source video: {video_path}"
            )
        return frame_count

    @staticmethod
    def _load_frame(record: VideoRecord, frame_index: int) -> Image.Image:
        frame_dir = RGB_FRAMES_ROOT / record.identifier
        frame_path = frame_dir / RGB_FRAME_TEMPLATE.format(frame_index)
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing RGB frame: {frame_path}")
        return Image.open(frame_path).convert("RGB")


class MiniUCFFlowDataset(_MiniUCFTSNDataset):
    """Load a stack of consecutive optical-flow pairs per temporal segment."""

    def __init__(
        self,
        split: str,
        num_segments: int,
        flow_stack_size: int,
        transform: Optional[Callable] = None,
    ) -> None:
        if flow_stack_size <= 0:
            raise ValueError("flow_stack_size must be positive")
        if transform is None:
            transform = flow_transforms()

        super().__init__(split=split, num_segments=num_segments, transform=transform)
        self.flow_stack_size = flow_stack_size
        stack_counts = {record.identifier: self._stack_count(record) for record in self.records}
        extra_name = f"stack{self.flow_stack_size}"
        self.sampled_start_indices = self._fixed_samples_from_manifest(FLOW, stack_counts, extra_name=extra_name)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record = self.records[index]
        start_indices = self.sampled_start_indices[record.identifier]

        flow_stacks = [self._load_flow_stack(record, start_index) for start_index in start_indices]
        clip = torch.stack(flow_stacks, dim=0)
        return clip, record.label

    def _stack_count(self, record: VideoRecord) -> int:
        flow_dir = FLOW_ROOT / record.identifier
        flow_pair_count = min(
            len(list(flow_dir.glob("flow_x_*.jpg"))),
            len(list(flow_dir.glob("flow_y_*.jpg"))),
        )
        if flow_pair_count == 0:
            raise FileNotFoundError(f"No optical flow frames found in {flow_dir}")

        stack_count = flow_pair_count - self.flow_stack_size + 1
        if stack_count <= 0:
            raise ValueError(
                f"Video {record.identifier} has fewer flow frames than "
                f"flow_stack_size={self.flow_stack_size}"
            )
        return stack_count

    def _load_flow_stack(self, record: VideoRecord, start_index: int) -> torch.Tensor:
        flow_pairs = [
            self._load_flow_pair(record, frame_index)
            for frame_index in range(start_index, start_index + self.flow_stack_size)
        ]
        return torch.cat(flow_pairs, dim=0)

    def _load_flow_pair(self, record: VideoRecord, frame_index: int) -> torch.Tensor:
        flow_dir = FLOW_ROOT / record.identifier
        x_path = flow_dir / FLOW_X_TEMPLATE.format(frame_index)
        y_path = flow_dir / FLOW_Y_TEMPLATE.format(frame_index)
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing optical flow pair: {x_path}, {y_path}")

        flow_x = Image.open(x_path).convert("L")
        flow_y = Image.open(y_path).convert("L")
        return torch.cat([self._to_tensor(flow_x), self._to_tensor(flow_y)], dim=0)


class MiniUCFFusionDataset(Dataset):
    """Return matching RGB and optical-flow TSN inputs for late fusion."""

    def __init__(
        self,
        split: str,
        num_segments: int,
        flow_stack_size: int,
    ) -> None:
        self.rgb_dataset = MiniUCFRGBDataset(split=split, num_segments=num_segments)
        self.flow_dataset = MiniUCFFlowDataset(
            split=split,
            num_segments=num_segments,
            flow_stack_size=flow_stack_size,
        )
        self.idx_to_class = self.rgb_dataset.idx_to_class

        if len(self.rgb_dataset) != len(self.flow_dataset):
            raise ValueError("RGB and flow datasets must have the same number of samples")

        for rgb_record, flow_record in zip(self.rgb_dataset.records, self.flow_dataset.records):
            if rgb_record.identifier != flow_record.identifier or rgb_record.label != flow_record.label:
                raise ValueError("RGB and flow datasets are not aligned")

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        rgb_clip, rgb_label = self.rgb_dataset[index]
        flow_clip, flow_label = self.flow_dataset[index]
        if rgb_label != flow_label:
            raise ValueError("RGB and flow labels do not match")
        return (rgb_clip, flow_clip), rgb_label


__all__ = [
    "FLOW",
    "FUSION",
    "RGB",
    "MiniUCFFusionDataset",
    "MiniUCFFlowDataset",
    "MiniUCFRGBDataset",
    "flow_transforms",
    "rgb_eval_transforms",
    "rgb_train_transforms",
]
