#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_DIR = ROOT / "annotations" / "makesense_2026-06-29_030317" / "labels"
POSITIVE_IMAGE_DIR = ROOT / "pic" / "已识别_不含框"
OK_IMAGE_DIR = ROOT / "data" / "ok"
OUT = ROOT / "dataset_yolo_toy"

POSITIVE_SPLITS = {
    "train": [
        "2026#05#12_10#50#01#232",
        "2026#05#12_10#50#01#814",
        "2026#05#12_10#50#02#381",
        "2026#05#12_10#50#10#134",
        "2026#05#12_10#50#26#881",
        "2026#05#12_10#50#27#443",
        "2026#05#12_10#50#37#159",
        "2026#05#12_10#50#38#978",
        "2026#05#12_10#51#01#084",
    ],
    "val": [
        "2026#05#12_10#51#10#008",
        "2026#05#12_10#51#22#998",
    ],
    "test": [
        "2026#05#12_10#51#27#612",
        "2026#05#12_10#51#30#047",
    ],
}

OK_SPLITS = {
    "train": [
        "KN112511001_2026-06-22_08-40-05-822",
        "KN112511001_2026-06-22_08-40-24-431",
        "KN112511001_2026-06-22_08-40-25-160",
        "KN112511001_2026-06-22_08-40-48-591",
        "KN112511001_2026-06-22_08-41-38-180",
        "KN112511001_2026-06-22_08-41-39-668",
        "KN112511001_2026-06-22_08-41-40-419",
        "KN112511001_2026-06-22_08-44-02-297",
        "KN112511001_2026-06-22_08-44-03-001",
    ],
    "val": [
        "KN112511001_2026-06-22_08-45-37-646",
        "KN112511001_2026-06-22_08-45-38-385",
    ],
    "test": [
        "KN112511001_2026-06-22_08-45-39-856",
        "KN112511001_2026-06-22_08-45-40-560",
    ],
}


def ensure_dirs() -> None:
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_positive(split: str, stem: str) -> None:
    src_img = POSITIVE_IMAGE_DIR / f"{stem}.png"
    src_label = ANNOTATION_DIR / f"{stem}.txt"
    dst_img = OUT / "images" / split / src_img.name
    dst_label = OUT / "labels" / split / src_label.name
    if not src_img.exists():
        raise FileNotFoundError(src_img)
    if not src_label.exists():
        raise FileNotFoundError(src_label)
    if not dst_img.exists():
        shutil.copy2(src_img, dst_img)
    shutil.copy2(src_label, dst_label)


def copy_ok(split: str, stem: str) -> None:
    src_img = OK_IMAGE_DIR / f"{stem}.jpg"
    dst_img = OUT / "images" / split / src_img.name
    dst_label = OUT / "labels" / split / f"{stem}.txt"
    if not src_img.exists():
        raise FileNotFoundError(src_img)
    if not dst_img.exists():
        shutil.copy2(src_img, dst_img)
    dst_label.write_text("", encoding="utf-8")


def write_data_yaml() -> None:
    (OUT / "data.yaml").write_text(
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


def write_readme() -> None:
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# YOLO Toy Dataset",
                "",
                "Small practice dataset for the yellow stain / streak detection workflow.",
                "",
                "- Positive images: 13 from `pic/已识别_不含框/`, labeled in makesense.ai.",
                "- Negative images: 13 from `data/ok/`, represented by empty YOLO label files.",
                "- Split: 9 positive + 9 OK train, 2 positive + 2 OK val, 2 positive + 2 OK test.",
                "- Class: `0: huangban`.",
                "",
                "This dataset is for learning and pipeline validation only. It is too small for a reliable production metric.",
                "",
                "Regenerate it from the project root with:",
                "",
                "```bash",
                "python scripts/data/prepare_yolo_toy_dataset.py",
                "```",
                "",
                "The generated `images/`, `labels/`, and cache files are local artifacts and are ignored by Git.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    for split, stems in POSITIVE_SPLITS.items():
        for stem in stems:
            copy_positive(split, stem)
    for split, stems in OK_SPLITS.items():
        for stem in stems:
            copy_ok(split, stem)
    write_data_yaml()
    write_readme()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
