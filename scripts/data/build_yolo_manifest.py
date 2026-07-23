#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CAPTURE_PATTERN = re.compile(
    r"^(?P<source>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})-(?P<millisecond>\d{3})$"
)


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    image_path: str
    label_path: str
    label_status: str
    box_count: int
    class_ids: str
    source_id: str
    capture_time: str
    batch: str
    split: str = "unassigned"
    review_status: str = "unreviewed"


def _index_by_stem(paths: list[Path], kind: str) -> dict[str, Path]:
    counts = Counter(path.stem for path in paths)
    duplicates = sorted(stem for stem, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate {kind} stems: {len(duplicates)}; samples: {', '.join(duplicates[:10])}"
        )
    return {path.stem: path for path in paths}


def _parse_label(path: Path, expected_class: int) -> tuple[int, tuple[int, ...]]:
    class_ids: list[int] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{path}:{line_number}: expected 5 YOLO fields, got {len(fields)}"
            )
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric YOLO field") from exc
        if not class_value.is_integer():
            raise ValueError(f"{path}:{line_number}: class id must be an integer")
        class_id = int(class_value)
        if class_id != expected_class:
            raise ValueError(
                f"{path}:{line_number}: expected class {expected_class}, got {class_id}"
            )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{path}:{line_number}: coordinates must be finite")
        x_center, y_center, width, height = coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            raise ValueError(f"{path}:{line_number}: box center must be in [0, 1]")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: width and height must be in (0, 1]")
        class_ids.append(class_id)
    return len(class_ids), tuple(sorted(set(class_ids)))


def _capture_metadata(stem: str) -> tuple[str, str, str]:
    match = CAPTURE_PATTERN.match(stem)
    if match is None:
        return "", "", ""
    values = match.groupdict()
    source_id = values["source"]
    capture_time = (
        f"{values['date']}T{values['hour']}:{values['minute']}:"
        f"{values['second']}.{values['millisecond']}"
    )
    batch = f"{source_id}_{values['date']}_{values['hour']}"
    return source_id, capture_time, batch


def build_manifest(
    images_dir: Path,
    labels_dir: Path,
    output_path: Path,
    expected_class: int = 0,
    empty_label_status: str = "EMPTY",
) -> dict[str, int]:
    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(labels_dir)
    if empty_label_status not in {"EMPTY", "OK"}:
        raise ValueError("empty_label_status must be EMPTY or OK")

    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_paths = sorted(path for path in labels_dir.rglob("*.txt") if path.is_file())
    images_by_stem = _index_by_stem(image_paths, "image")
    labels_by_stem = _index_by_stem(label_paths, "label")

    missing_labels = sorted(images_by_stem.keys() - labels_by_stem.keys())
    orphan_labels = sorted(labels_by_stem.keys() - images_by_stem.keys())
    if missing_labels or orphan_labels:
        details = [
            f"Images without TXT: {len(missing_labels)}",
            f"TXT without image: {len(orphan_labels)}",
        ]
        if missing_labels:
            details.append("Missing TXT samples: " + ", ".join(missing_labels[:10]))
        if orphan_labels:
            details.append("Orphan TXT samples: " + ", ".join(orphan_labels[:10]))
        raise ValueError("; ".join(details))

    common_root = Path(os.path.commonpath([images_dir, labels_dir]))

    records: list[ManifestRecord] = []
    invalid_labels: list[str] = []
    for stem in sorted(images_by_stem):
        image_path = images_by_stem[stem]
        label_path = labels_by_stem[stem]
        try:
            box_count, class_ids = _parse_label(label_path, expected_class=expected_class)
        except (UnicodeError, ValueError) as exc:
            invalid_labels.append(str(exc))
            continue
        source_id, capture_time, batch = _capture_metadata(stem)
        records.append(
            ManifestRecord(
                image_id=stem,
                image_path=image_path.relative_to(common_root).as_posix(),
                label_path=label_path.relative_to(common_root).as_posix(),
                label_status="NG" if box_count else empty_label_status,
                box_count=box_count,
                class_ids=";".join(str(value) for value in class_ids),
                source_id=source_id,
                capture_time=capture_time,
                batch=batch,
            )
        )
    if invalid_labels:
        sample = "\n".join(f"- {message}" for message in invalid_labels[:20])
        raise ValueError(
            f"Invalid YOLO labels: {len(invalid_labels)}\n{sample}"
            + ("\n- ..." if len(invalid_labels) > 20 else "")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ManifestRecord.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)

    ng_count = sum(record.label_status == "NG" for record in records)
    empty_count = len(records) - ng_count
    missing_capture_metadata = sum(not record.capture_time for record in records)
    return {
        "images": len(image_paths),
        "labels": len(label_paths),
        "ng": ng_count,
        "empty": empty_count,
        "missing_capture_metadata": missing_capture_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired YOLO images/labels and write a manifest without moving data."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-class", type=int, default=0)
    parser.add_argument(
        "--empty-label-status",
        choices=("EMPTY", "OK"),
        default="EMPTY",
        help="Use OK only when every blank TXT has been explicitly reviewed as defect-free.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        images_dir=args.images,
        labels_dir=args.labels,
        output_path=args.output,
        expected_class=args.expected_class,
        empty_label_status=args.empty_label_status,
    )
    print(f"Manifest: {args.output}")
    print(f"Images: {summary['images']}")
    print(f"Labels: {summary['labels']}")
    print(f"NG / non-empty labels: {summary['ng']}")
    print(f"Empty labels: {summary['empty']}")
    print(f"Filenames without parsed capture time: {summary['missing_capture_metadata']}")
    print("YOLO label validation: passed")


if __name__ == "__main__":
    main()
