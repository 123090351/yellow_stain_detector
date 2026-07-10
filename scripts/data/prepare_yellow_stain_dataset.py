#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "training_data" / "training_data"
DEFAULT_OUTPUT = ROOT / "datasets" / "yellow_stain_v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Record:
    stem: str
    image_path: Path
    label_path: Path
    is_positive: bool


def _validate_label(label_path: Path) -> bool:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()]
    nonblank_lines = [line for line in lines if line]

    for line_number, line in enumerate(nonblank_lines, 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields, got {len(parts)}")

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: non-numeric YOLO field: {line}") from exc

        if class_id != 0:
            raise ValueError(f"{label_path}:{line_number}: expected class 0, got {class_id}")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            raise ValueError(f"{label_path}:{line_number}: YOLO coordinates must be in [0, 1]: {line}")
        if width <= 0 or height <= 0:
            raise ValueError(f"{label_path}:{line_number}: YOLO width/height must be > 0: {line}")

    return bool(nonblank_lines)


def build_records(source: Path) -> list[Record]:
    image_dir = source / "images"
    label_dir = source / "labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_paths = sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")

    images_by_stem = {path.stem: path for path in image_paths}
    labels_by_stem = {path.stem: path for path in label_paths}
    missing_labels = sorted(set(images_by_stem) - set(labels_by_stem))
    missing_images = sorted(set(labels_by_stem) - set(images_by_stem))

    if missing_labels or missing_images:
        detail = [
            f"images without labels: {len(missing_labels)}",
            f"labels without images: {len(missing_images)}",
        ]
        if missing_labels:
            detail.append("sample missing labels: " + ", ".join(missing_labels[:10]))
        if missing_images:
            detail.append("sample missing images: " + ", ".join(missing_images[:10]))
        raise ValueError("; ".join(detail))

    return [
        Record(
            stem=stem,
            image_path=images_by_stem[stem],
            label_path=labels_by_stem[stem],
            is_positive=_validate_label(labels_by_stem[stem]),
        )
        for stem in sorted(images_by_stem)
    ]


def _split_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0

    train_count = round(total * 0.70)
    val_count = round(total * 0.15)
    test_count = total - train_count - val_count

    if val_count == 0:
        train_count -= 1
        val_count = 1
    if test_count == 0:
        train_count -= 1
        test_count = 1

    return train_count, val_count, test_count


def _split_group(records: list[Record], rng: random.Random) -> dict[str, list[Record]]:
    shuffled = records[:]
    rng.shuffle(shuffled)

    train_count, val_count, test_count = _split_counts(len(shuffled))
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }


def split_records(records: Iterable[Record], seed: int = 42) -> dict[str, list[Record]]:
    rng = random.Random(seed)
    positives = [record for record in records if record.is_positive]
    negatives = [record for record in records if not record.is_positive]

    positive_splits = _split_group(positives, rng)
    negative_splits = _split_group(negatives, rng)

    splits: dict[str, list[Record]] = {}
    for split in SPLITS:
        items = positive_splits[split] + negative_splits[split]
        rng.shuffle(items)
        splits[split] = items

    return splits


def _ensure_output_dirs(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def _copy_split(output: Path, split: str, records: list[Record]) -> None:
    for record in records:
        shutil.copy2(record.image_path, output / "images" / split / record.image_path.name)
        shutil.copy2(record.label_path, output / "labels" / split / record.label_path.name)


def _write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: huangban",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_report(output: Path, source: Path, seed: int, splits: dict[str, list[Record]]) -> dict[str, int]:
    summary = {
        "total": sum(len(records) for records in splits.values()),
        "positive": sum(record.is_positive for records in splits.values() for record in records),
        "negative": sum(not record.is_positive for records in splits.values() for record in records),
    }

    report_lines = [
        "# Yellow Stain YOLO Dataset Report",
        "",
        f"source: {source}",
        f"output: {output}",
        f"seed: {seed}",
        "",
        f"total_images: {summary['total']}",
        f"positive_nonblank_labels: {summary['positive']}",
        f"negative_blank_labels: {summary['negative']}",
        "",
        "| split | total | positive | negative |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        records = splits[split]
        positive = sum(record.is_positive for record in records)
        negative = len(records) - positive
        report_lines.append(f"| {split} | {len(records)} | {positive} | {negative} |")

    (output / "dataset_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def prepare_dataset(source: Path, output: Path, seed: int = 42) -> dict[str, int]:
    records = build_records(source)
    splits = split_records(records, seed=seed)

    _ensure_output_dirs(output)
    for split, split_records_ in splits.items():
        _copy_split(output, split, split_records_)
    _write_data_yaml(output)
    return _write_report(output, source, seed, splits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the cleaned yellow-stain YOLO dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source folder containing images/ and labels/.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output YOLO dataset folder.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(args.source, args.output, seed=args.seed)
    print(f"Wrote {args.output}")
    print(
        "Summary: "
        f"total={summary['total']}, "
        f"positive={summary['positive']}, "
        f"negative={summary['negative']}"
    )


if __name__ == "__main__":
    main()
