from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable
from zipfile import ZipFile

import cv2
import gdown

DATASET_URLS = {
    "mot_train": "https://drive.google.com/file/d/1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh/view?usp=sharing",
    "mot_val": "https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view?usp=sharing",
}

CATEGORY_NAMES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

CATEGORY_COLORS = {
    0: (128, 128, 128),
    1: (0, 255, 0),
    2: (50, 205, 50),
    3: (255, 215, 0),
    4: (0, 165, 255),
    5: (255, 140, 0),
    6: (0, 69, 255),
    7: (255, 0, 255),
    8: (255, 20, 147),
    9: (255, 255, 0),
    10: (0, 255, 255),
}


@dataclass(frozen=True)
class Detection:
    frame_id: int
    track_id: int
    x: int
    y: int
    w: int
    h: int
    score: float
    category_id: int
    truncation: int
    occlusion: int

    @property
    def label(self) -> str:
        name = CATEGORY_NAMES.get(self.category_id, f"cls{self.category_id}")
        return f"{name}#{self.track_id}"

    @property
    def color(self) -> tuple[int, int, int]:
        return CATEGORY_COLORS.get(self.category_id, (255, 255, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VisDrone MOT and export a ground-truth video preview."
    )
    parser.add_argument("--split", choices=sorted(DATASET_URLS), default="mot_val")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visdrone_preview.mp4"))
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    return parser.parse_args()


def download_dataset(split: str, data_root: Path, redownload: bool) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / f"{split}.zip"
    extract_root = data_root / split

    if redownload or not zip_path.exists():
        gdown.download(DATASET_URLS[split], str(zip_path), quiet=False, fuzzy=True)

    if redownload and extract_root.exists():
        shutil.rmtree(extract_root)

    if not extract_root.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path) as archive:
            archive.extractall(extract_root)

    return normalize_dataset_root(extract_root)


def normalize_dataset_root(extract_root: Path) -> Path:
    if (extract_root / "sequences").exists() and (extract_root / "annotations").exists():
        return extract_root

    children = [child for child in extract_root.iterdir() if child.is_dir()]
    if len(children) == 1:
        child = children[0]
        if (child / "sequences").exists() and (child / "annotations").exists():
            return child

    matches = []
    for child in extract_root.rglob("sequences"):
        candidate = child.parent
        if (candidate / "annotations").exists():
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate VisDrone MOT layout under {extract_root}. Expected sequences/ and annotations/."
    )


def choose_sequence(dataset_root: Path, requested: str | None) -> tuple[str, Path, Path]:
    sequences_dir = dataset_root / "sequences"
    annotations_dir = dataset_root / "annotations"
    available = sorted(path for path in sequences_dir.iterdir() if path.is_dir())
    if not available:
        raise FileNotFoundError(f"No sequences found in {sequences_dir}")

    if requested is None:
        sequence_dir = available[0]
    else:
        sequence_dir = sequences_dir / requested
        if not sequence_dir.exists():
            names = ", ".join(path.name for path in available[:10])
            raise FileNotFoundError(
                f"Sequence '{requested}' not found in {sequences_dir}. Example sequences: {names}"
            )

    annotation_path = annotations_dir / f"{sequence_dir.name}.txt"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    return sequence_dir.name, sequence_dir, annotation_path


def load_annotations(annotation_path: Path) -> DefaultDict[int, list[Detection]]:
    grouped: DefaultDict[int, list[Detection]] = defaultdict(list)
    with annotation_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            padded = row + ["0"] * max(0, 10 - len(row))
            detection = Detection(
                frame_id=int(float(padded[0])),
                track_id=int(float(padded[1])),
                x=int(float(padded[2])),
                y=int(float(padded[3])),
                w=int(float(padded[4])),
                h=int(float(padded[5])),
                score=float(padded[6]),
                category_id=int(float(padded[7])),
                truncation=int(float(padded[8])),
                occlusion=int(float(padded[9])),
            )
            grouped[detection.frame_id].append(detection)
    return grouped


def iter_frames(sequence_dir: Path) -> Iterable[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    frames: list[Path] = []
    for pattern in patterns:
        frames.extend(sorted(sequence_dir.glob(pattern)))
    if not frames:
        raise FileNotFoundError(f"No image frames found in {sequence_dir}")
    return frames


def render_video(
    frames: list[Path],
    annotations: DefaultDict[int, list[Detection]],
    output_path: Path,
    fps: int,
    max_seconds: int,
) -> None:
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame {frames[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    max_frame_count = min(len(frames), fps * max_seconds)

    try:
        for index, frame_path in enumerate(frames[:max_frame_count], start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame {frame_path}")
            draw_overlay(frame, index, annotations[index])
            writer.write(frame)
    finally:
        writer.release()


def draw_overlay(frame, frame_id: int, detections: list[Detection]) -> None:
    for det in detections:
        if det.category_id <= 0:
            continue
        pt1 = (det.x, det.y)
        pt2 = (det.x + det.w, det.y + det.h)
        cv2.rectangle(frame, pt1, pt2, det.color, 2)
        cv2.putText(
            frame,
            det.label,
            (det.x, max(20, det.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            det.color,
            1,
            cv2.LINE_AA,
        )

    banner = f"frame={frame_id} detections={len(detections)}"
    cv2.rectangle(frame, (10, 10), (320, 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        banner,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    dataset_root = download_dataset(args.split, args.data_root, args.redownload)
    sequence_name, sequence_dir, annotation_path = choose_sequence(dataset_root, args.sequence)
    annotations = load_annotations(annotation_path)
    frames = list(iter_frames(sequence_dir))
    render_video(frames, annotations, args.output, args.fps, args.max_seconds)

    if not args.keep_zip:
        zip_path = args.data_root / f"{args.split}.zip"
        if zip_path.exists():
            zip_path.unlink()

    print(f"Rendered sequence: {sequence_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output video: {args.output}")
    print(f"Frames used: {min(len(frames), args.fps * args.max_seconds)}")


if __name__ == "__main__":
    main()
