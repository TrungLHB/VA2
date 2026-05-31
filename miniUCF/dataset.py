"""PyTorch datasets for the miniUCF action-recognition subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


RGB = "rgb"
FLOW = "flow"

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
    """Extract RGB frames from AVI files into ``data/mini_UCF_frames``."""

    try:
        # noinspection PyPackageRequirements
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


__all__ = [
    "FLOW",
    "FLOW_ROOT",
    "FLOW_X_TEMPLATE",
    "FLOW_Y_TEMPLATE",
    "PROJECT_ROOT",
    "RGB",
    "RGB_FRAMES_ROOT",
    "RGB_FRAME_TEMPLATE",
    "TSN_SAMPLE_ROOT",
    "TRAIN_SPLIT_FILE",
    "VALIDATION_SPLIT_FILE",
    "VIDEO_ROOT",
    "VideoRecord",
    "extract_rgb_frames",
    "read_class_mapping",
    "read_split_file",
    "split_file_for",
]
