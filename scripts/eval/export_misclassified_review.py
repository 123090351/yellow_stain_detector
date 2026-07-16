#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.evaluate_image_level_ok_ng import evaluate_image_level


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ExportSummary:
    false_negatives: int
    false_positives: int

    @property
    def total(self) -> int:
        return self.false_negatives + self.false_positives


def _index_images(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in indexed:
            raise ValueError(f"Duplicate image stem '{path.stem}' in {directory}")
        indexed[path.stem] = path
    return indexed


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def export_review_package(
    *,
    images: Path,
    gt_labels: Path,
    pred_labels: Path,
    output_dir: Path,
    prediction_images: Path | None = None,
    overwrite: bool = False,
) -> ExportSummary:
    metrics = evaluate_image_level(gt_labels, pred_labels)
    image_index = _index_images(images)
    prediction_image_index = _index_images(prediction_images) if prediction_images else {}
    _prepare_output(output_dir, overwrite)

    rows: list[dict[str, str]] = []
    groups = (
        ("false_negative", "false_negatives", metrics.false_negative_stems, "NG", "OK"),
        ("false_positive", "false_positives", metrics.false_positive_stems, "OK", "NG"),
    )

    for error_type, folder_name, stems, ground_truth, prediction in groups:
        category_dir = output_dir / folder_name
        for child in ("original", "prediction", "ground_truth_label", "prediction_label"):
            (category_dir / child).mkdir(parents=True, exist_ok=True)

        for stem in stems:
            source_image = image_index.get(stem)
            if source_image is None:
                raise FileNotFoundError(f"No source image found for label stem '{stem}' in {images}")

            shutil.copy2(source_image, category_dir / "original" / source_image.name)

            prediction_image = prediction_image_index.get(stem)
            if prediction_image is not None:
                shutil.copy2(prediction_image, category_dir / "prediction" / prediction_image.name)

            gt_label = gt_labels / f"{stem}.txt"
            shutil.copy2(gt_label, category_dir / "ground_truth_label" / gt_label.name)

            pred_label = pred_labels / f"{stem}.txt"
            if pred_label.exists():
                shutil.copy2(pred_label, category_dir / "prediction_label" / pred_label.name)

            rows.append(
                {
                    "filename": source_image.name,
                    "error_type": error_type,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "review_notes": "",
                }
            )

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=("filename", "error_type", "ground_truth", "prediction", "review_notes"),
        )
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "README.txt").write_text(
        "Lucas review package\n"
        "====================\n"
        f"False negatives (true NG, predicted OK): {metrics.false_negative}\n"
        f"False positives (true OK, predicted NG): {metrics.false_positive}\n"
        f"Total images for review: {metrics.false_negative + metrics.false_positive}\n\n"
        "Add review comments to the review_notes column in manifest.csv.\n",
        encoding="utf-8",
    )

    return ExportSummary(
        false_negatives=metrics.false_negative,
        false_positives=metrics.false_positive,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export misclassified images into a review package.")
    parser.add_argument("--images", type=Path, required=True, help="Source image directory.")
    parser.add_argument("--gt-labels", type=Path, required=True, help="Ground-truth label directory.")
    parser.add_argument("--pred-labels", type=Path, required=True, help="Prediction label directory.")
    parser.add_argument(
        "--prediction-images",
        type=Path,
        help="Optional directory containing YOLO prediction preview images.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Review package output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_review_package(
        images=args.images,
        gt_labels=args.gt_labels,
        pred_labels=args.pred_labels,
        output_dir=args.output_dir,
        prediction_images=args.prediction_images,
        overwrite=args.overwrite,
    )
    print(f"Review package created: {args.output_dir}")
    print(f"False negatives: {summary.false_negatives}")
    print(f"False positives: {summary.false_positives}")
    print(f"Total: {summary.total}")


if __name__ == "__main__":
    main()
