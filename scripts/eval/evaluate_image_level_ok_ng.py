#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageLevelMetrics:
    total: int
    prediction_files_missing: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    false_positive_stems: list[str]
    false_negative_stems: list[str]

    @property
    def accuracy(self) -> float:
        return _safe_divide(self.true_positive + self.true_negative, self.total)

    @property
    def precision(self) -> float:
        return _safe_divide(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall(self) -> float:
        return _safe_divide(self.true_positive, self.true_positive + self.false_negative)

    @property
    def ok_accuracy(self) -> float:
        return _safe_divide(self.true_negative, self.true_negative + self.false_positive)

    @property
    def ng_accuracy(self) -> float:
        return self.recall


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _has_nonblank_label(label_path: Path) -> bool:
    return label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())


def evaluate_image_level(gt_labels: Path, pred_labels: Path) -> ImageLevelMetrics:
    gt_files = sorted(gt_labels.glob("*.txt"))
    if not gt_files:
        raise ValueError(f"No ground-truth .txt labels found in {gt_labels}")

    tp = fp = tn = fn = 0
    missing_pred_files = 0
    false_positives: list[str] = []
    false_negatives: list[str] = []

    for gt_file in gt_files:
        stem = gt_file.stem
        gt_ng = _has_nonblank_label(gt_file)

        pred_file = pred_labels / f"{stem}.txt"
        pred_ng = _has_nonblank_label(pred_file)
        if not pred_file.exists():
            missing_pred_files += 1

        if gt_ng and pred_ng:
            tp += 1
        elif not gt_ng and pred_ng:
            fp += 1
            false_positives.append(stem)
        elif not gt_ng and not pred_ng:
            tn += 1
        else:
            fn += 1
            false_negatives.append(stem)

    return ImageLevelMetrics(
        total=len(gt_files),
        prediction_files_missing=missing_pred_files,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        false_positive_stems=false_positives,
        false_negative_stems=false_negatives,
    )


def print_report(metrics: ImageLevelMetrics, sample_limit: int) -> None:
    print("Image-level OK/NG evaluation")
    print("--------------------------------")
    print(f"GT files: {metrics.total}")
    print(f"Prediction files missing: {metrics.prediction_files_missing}")
    print()
    print(f"TP true NG predicted NG: {metrics.true_positive}")
    print(f"FP true OK predicted NG: {metrics.false_positive}")
    print(f"TN true OK predicted OK: {metrics.true_negative}")
    print(f"FN true NG predicted OK: {metrics.false_negative}")
    print()
    print(f"accuracy:     {metrics.accuracy:.3f}")
    print(f"precision:    {metrics.precision:.3f}")
    print(f"recall/NG hit:{metrics.recall:.3f}")
    print(f"OK accuracy:  {metrics.ok_accuracy:.3f}")
    print(f"NG accuracy:  {metrics.ng_accuracy:.3f}")
    print()
    print("False negatives sample:")
    for stem in metrics.false_negative_stems[:sample_limit]:
        print(f"  {stem}")
    print()
    print("False positives sample:")
    for stem in metrics.false_positive_stems[:sample_limit]:
        print(f"  {stem}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate image-level OK/NG classification from YOLO label folders. "
            "A non-empty ground-truth label means NG; a non-empty prediction label means predicted NG."
        )
    )
    parser.add_argument("--gt-labels", type=Path, required=True, help="Ground-truth YOLO label folder.")
    parser.add_argument("--pred-labels", type=Path, required=True, help="Predicted YOLO label folder.")
    parser.add_argument("--sample-limit", type=int, default=20, help="Number of false examples to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_image_level(args.gt_labels, args.pred_labels)
    print_report(metrics, sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
