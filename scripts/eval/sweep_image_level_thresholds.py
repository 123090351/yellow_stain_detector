#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    total: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

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


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _has_nonblank_label(path: Path) -> bool:
    return path.exists() and bool(path.read_text(encoding="utf-8").strip())


def read_max_confidence(path: Path) -> float:
    if not path.exists():
        return 0.0

    confidences: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 6:
            raise ValueError(
                f"Prediction label must include confidence in column 6: {path}:{line_number}"
            )
        confidences.append(float(fields[5]))
    return max(confidences, default=0.0)


def sweep_thresholds(
    gt_labels: Path,
    pred_labels: Path,
    thresholds: list[float],
) -> list[ThresholdResult]:
    gt_files = sorted(gt_labels.glob("*.txt"))
    if not gt_files:
        raise ValueError(f"No ground-truth .txt labels found in {gt_labels}")
    if not thresholds:
        raise ValueError("At least one threshold is required")

    samples = [
        (
            _has_nonblank_label(gt_file),
            read_max_confidence(pred_labels / f"{gt_file.stem}.txt"),
        )
        for gt_file in gt_files
    ]

    results: list[ThresholdResult] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for gt_ng, max_confidence in samples:
            pred_ng = max_confidence >= threshold
            if gt_ng and pred_ng:
                tp += 1
            elif not gt_ng and pred_ng:
                fp += 1
            elif not gt_ng and not pred_ng:
                tn += 1
            else:
                fn += 1

        results.append(
            ThresholdResult(
                threshold=threshold,
                total=len(samples),
                true_positive=tp,
                false_positive=fp,
                true_negative=tn,
                false_negative=fn,
            )
        )
    return results


def choose_threshold(
    results: list[ThresholdResult],
    target_recall: float,
) -> ThresholdResult:
    if not results:
        raise ValueError("No threshold results available")

    candidates = [result for result in results if result.recall >= target_recall]
    if candidates:
        return max(
            candidates,
            key=lambda result: (
                result.ok_accuracy,
                result.accuracy,
                result.precision,
                result.threshold,
            ),
        )
    return max(
        results,
        key=lambda result: (
            result.recall,
            result.ok_accuracy,
            result.accuracy,
            result.threshold,
        ),
    )


def build_thresholds(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Threshold step must be greater than zero")
    current = Decimal(str(start))
    end = Decimal(str(stop))
    increment = Decimal(str(step))
    thresholds: list[float] = []
    while current <= end:
        thresholds.append(float(current))
        current += increment
    return thresholds


def write_csv(results: list[ThresholdResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "threshold",
                "accuracy",
                "precision",
                "ng_recall",
                "ok_accuracy",
                "tp",
                "fp",
                "tn",
                "fn",
            )
        )
        for result in results:
            writer.writerow(
                (
                    f"{result.threshold:.4f}",
                    f"{result.accuracy:.4f}",
                    f"{result.precision:.4f}",
                    f"{result.recall:.4f}",
                    f"{result.ok_accuracy:.4f}",
                    result.true_positive,
                    result.false_positive,
                    result.true_negative,
                    result.false_negative,
                )
            )


def write_plot(results: list[ThresholdResult], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    thresholds = [result.threshold for result in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, [result.recall for result in results], label="NG recall")
    plt.plot(thresholds, [result.ok_accuracy for result in results], label="OK accuracy")
    plt.plot(thresholds, [result.accuracy for result in results], label="Overall accuracy")
    plt.plot(thresholds, [result.precision for result in results], label="Precision")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep image-level OK/NG metrics over prediction confidence thresholds."
    )
    parser.add_argument("--gt-labels", type=Path, required=True)
    parser.add_argument("--pred-labels", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.01)
    parser.add_argument("--stop", type=float, default=0.20)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = build_thresholds(args.start, args.stop, args.step)
    results = sweep_thresholds(args.gt_labels, args.pred_labels, thresholds)
    selected = choose_threshold(results, target_recall=args.target_recall)
    write_csv(results, args.output_csv)
    if args.output_plot:
        write_plot(results, args.output_plot)

    print(f"Threshold results saved to: {args.output_csv}")
    if args.output_plot:
        print(f"Threshold plot saved to: {args.output_plot}")
    print()
    print(f"Recommended threshold for target recall >= {args.target_recall:.2f}:")
    print(f"threshold:   {selected.threshold:.2f}")
    print(f"accuracy:    {selected.accuracy:.3f}")
    print(f"precision:   {selected.precision:.3f}")
    print(f"NG recall:   {selected.recall:.3f}")
    print(f"OK accuracy: {selected.ok_accuracy:.3f}")
    print(f"TP/FP/TN/FN: {selected.true_positive}/{selected.false_positive}/"
          f"{selected.true_negative}/{selected.false_negative}")


if __name__ == "__main__":
    main()
