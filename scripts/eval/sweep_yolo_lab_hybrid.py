#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PredictionBox:
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float


@dataclass(frozen=True)
class ScoredBox:
    confidence: float
    lab_delta: float


@dataclass(frozen=True)
class ImageSample:
    stem: str
    gt_ng: bool
    boxes: tuple[ScoredBox, ...]

    @property
    def max_confidence(self) -> float:
        return max((box.confidence for box in self.boxes), default=0.0)


@dataclass(frozen=True)
class HybridResult:
    rescue_confidence: float
    lab_threshold: float
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


def build_float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Step must be greater than zero")
    current = Decimal(str(start))
    end = Decimal(str(stop))
    increment = Decimal(str(step))
    values: list[float] = []
    while current <= end:
        values.append(float(current))
        current += increment
    return values


def parse_prediction_boxes(path: Path) -> list[PredictionBox]:
    if not path.exists():
        return []

    boxes: list[PredictionBox] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 6:
            raise ValueError(
                f"Prediction label must include confidence in column 6: {path}:{line_number}"
            )
        boxes.append(
            PredictionBox(
                x_center=float(fields[1]),
                y_center=float(fields[2]),
                width=float(fields[3]),
                height=float(fields[4]),
                confidence=float(fields[5]),
            )
        )
    return boxes


def lab_yellow_delta(
    lab_b: np.ndarray,
    box: PredictionBox,
    percentile: float,
    baseline_mode: str = "global",
    ring_scale: float = 2.0,
) -> float:
    if lab_b.ndim != 2:
        raise ValueError("LAB b input must be a two-dimensional array")
    if not 0 < percentile <= 100:
        raise ValueError("LAB percentile must be in (0, 100]")
    if baseline_mode not in {"global", "local-ring"}:
        raise ValueError("LAB baseline mode must be 'global' or 'local-ring'")
    if ring_scale <= 1.0:
        raise ValueError("LAB ring scale must be greater than 1")

    image_height, image_width = lab_b.shape
    x1 = max(0, int(np.floor((box.x_center - box.width / 2) * image_width)))
    y1 = max(0, int(np.floor((box.y_center - box.height / 2) * image_height)))
    x2 = min(image_width, int(np.ceil((box.x_center + box.width / 2) * image_width)))
    y2 = min(image_height, int(np.ceil((box.y_center + box.height / 2) * image_height)))
    if x2 <= x1 or y2 <= y1:
        return float("-inf")

    roi = lab_b[y1:y2, x1:x2]
    if baseline_mode == "global":
        baseline = float(np.median(lab_b))
    else:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        ring_width = (x2 - x1) * ring_scale
        ring_height = (y2 - y1) * ring_scale
        ring_x1 = max(0, int(np.floor(center_x - ring_width / 2)))
        ring_y1 = max(0, int(np.floor(center_y - ring_height / 2)))
        ring_x2 = min(image_width, int(np.ceil(center_x + ring_width / 2)))
        ring_y2 = min(image_height, int(np.ceil(center_y + ring_height / 2)))

        expanded = lab_b[ring_y1:ring_y2, ring_x1:ring_x2]
        ring_mask = np.ones(expanded.shape, dtype=bool)
        ring_mask[y1 - ring_y1 : y2 - ring_y1, x1 - ring_x1 : x2 - ring_x1] = False
        ring_pixels = expanded[ring_mask]
        baseline = (
            float(np.median(ring_pixels))
            if ring_pixels.size
            else float(np.median(lab_b))
        )
    return float(np.percentile(roi, percentile) - baseline)


def index_images(images_dir: Path) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in image_paths:
            raise ValueError(
                f"Duplicate image stem '{path.stem}': {image_paths[path.stem]} and {path}"
            )
        image_paths[path.stem] = path
    if not image_paths:
        raise ValueError(f"No supported images found in {images_dir}")
    return image_paths


def load_samples(
    images_dir: Path,
    gt_labels: Path,
    pred_labels: Path,
    lab_percentile: float,
    lab_baseline: str = "global",
    lab_ring_scale: float = 2.0,
) -> list[ImageSample]:
    gt_files = sorted(gt_labels.glob("*.txt"))
    if not gt_files:
        raise ValueError(f"No ground-truth .txt labels found in {gt_labels}")

    image_paths = index_images(images_dir)
    samples: list[ImageSample] = []
    for index, gt_file in enumerate(gt_files, start=1):
        image_path = image_paths.get(gt_file.stem)
        if image_path is None:
            raise ValueError(f"No source image found for ground-truth label: {gt_file.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {image_path}")
        lab_b = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 2].astype(np.float32)
        prediction_boxes = parse_prediction_boxes(pred_labels / f"{gt_file.stem}.txt")
        scored_boxes = tuple(
            ScoredBox(
                confidence=box.confidence,
                lab_delta=lab_yellow_delta(
                    lab_b,
                    box,
                    percentile=lab_percentile,
                    baseline_mode=lab_baseline,
                    ring_scale=lab_ring_scale,
                ),
            )
            for box in prediction_boxes
        )
        samples.append(
            ImageSample(
                stem=gt_file.stem,
                gt_ng=_has_nonblank_label(gt_file),
                boxes=scored_boxes,
            )
        )
        if index % 25 == 0 or index == len(gt_files):
            print(f"Scored LAB-b candidates: {index}/{len(gt_files)}")
    return samples


def classify_sample(
    sample: ImageSample,
    direct_confidence: float,
    rescue_confidence: float,
    lab_threshold: float,
) -> tuple[bool, bool, bool]:
    direct_ng = sample.max_confidence >= direct_confidence
    lab_rescue = any(
        rescue_confidence <= box.confidence < direct_confidence
        and box.lab_delta >= lab_threshold
        for box in sample.boxes
    )
    return direct_ng or lab_rescue, direct_ng, lab_rescue


def evaluate_configuration(
    samples: list[ImageSample],
    direct_confidence: float,
    rescue_confidence: float,
    lab_threshold: float,
) -> HybridResult:
    tp = fp = tn = fn = 0
    for sample in samples:
        pred_ng, _, _ = classify_sample(
            sample,
            direct_confidence=direct_confidence,
            rescue_confidence=rescue_confidence,
            lab_threshold=lab_threshold,
        )
        if sample.gt_ng and pred_ng:
            tp += 1
        elif not sample.gt_ng and pred_ng:
            fp += 1
        elif not sample.gt_ng and not pred_ng:
            tn += 1
        else:
            fn += 1
    return HybridResult(
        rescue_confidence=rescue_confidence,
        lab_threshold=lab_threshold,
        total=len(samples),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def sweep_hybrid(
    samples: list[ImageSample],
    direct_confidence: float,
    rescue_confidences: list[float],
    lab_thresholds: list[float],
) -> list[HybridResult]:
    if not samples:
        raise ValueError("At least one image sample is required")
    if not rescue_confidences or not lab_thresholds:
        raise ValueError("Rescue confidence and LAB threshold grids cannot be empty")
    if any(value < 0 or value >= direct_confidence for value in rescue_confidences):
        raise ValueError("Every rescue confidence must be >= 0 and below direct confidence")

    return [
        evaluate_configuration(
            samples,
            direct_confidence=direct_confidence,
            rescue_confidence=rescue_confidence,
            lab_threshold=lab_threshold,
        )
        for rescue_confidence in rescue_confidences
        for lab_threshold in lab_thresholds
    ]


def choose_result(results: list[HybridResult], target_recall: float) -> HybridResult:
    if not results:
        raise ValueError("No hybrid results available")
    candidates = [result for result in results if result.recall >= target_recall]
    pool = candidates or results
    return max(
        pool,
        key=lambda result: (
            result.recall >= target_recall,
            result.recall if not candidates else result.ok_accuracy,
            result.accuracy,
            result.precision,
            result.rescue_confidence,
            result.lab_threshold,
        ),
    )


def write_results_csv(results: list[HybridResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "rescue_confidence",
                "lab_threshold",
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
                    f"{result.rescue_confidence:.4f}",
                    f"{result.lab_threshold:.4f}",
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


def write_decisions_csv(
    samples: list[ImageSample],
    direct_confidence: float,
    selected: HybridResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "stem",
                "gt_ng",
                "max_confidence",
                "max_eligible_lab_delta",
                "direct_ng",
                "lab_rescue",
                "pred_ng",
                "outcome",
            )
        )
        for sample in samples:
            pred_ng, direct_ng, lab_rescue = classify_sample(
                sample,
                direct_confidence=direct_confidence,
                rescue_confidence=selected.rescue_confidence,
                lab_threshold=selected.lab_threshold,
            )
            eligible_scores = [
                box.lab_delta
                for box in sample.boxes
                if selected.rescue_confidence <= box.confidence < direct_confidence
            ]
            if sample.gt_ng and pred_ng:
                outcome = "TP"
            elif not sample.gt_ng and pred_ng:
                outcome = "FP"
            elif not sample.gt_ng and not pred_ng:
                outcome = "TN"
            else:
                outcome = "FN"
            writer.writerow(
                (
                    sample.stem,
                    int(sample.gt_ng),
                    f"{sample.max_confidence:.6f}",
                    f"{max(eligible_scores):.4f}" if eligible_scores else "",
                    int(direct_ng),
                    int(lab_rescue),
                    int(pred_ng),
                    outcome,
                )
            )


def print_result(title: str, result: HybridResult) -> None:
    print(title)
    print(f"rescue confidence: {result.rescue_confidence:.4f}")
    print(f"LAB-b threshold:   {result.lab_threshold:.4f}")
    print(f"accuracy:          {result.accuracy:.3f}")
    print(f"precision:         {result.precision:.3f}")
    print(f"NG recall:         {result.recall:.3f}")
    print(f"OK accuracy:       {result.ok_accuracy:.3f}")
    print(
        "TP/FP/TN/FN:      "
        f"{result.true_positive}/{result.false_positive}/"
        f"{result.true_negative}/{result.false_negative}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a recall-first image-level rule: direct YOLO detections are kept, "
            "and lower-confidence boxes may be rescued by relative LAB-b yellow evidence."
        )
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--gt-labels", type=Path, required=True)
    parser.add_argument("--pred-labels", type=Path, required=True)
    parser.add_argument("--direct-conf", type=float, required=True)
    parser.add_argument(
        "--rescue-conf",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09],
    )
    parser.add_argument("--lab-start", type=float, default=0.0)
    parser.add_argument("--lab-stop", type=float, default=40.0)
    parser.add_argument("--lab-step", type=float, default=0.5)
    parser.add_argument("--lab-percentile", type=float, default=90.0)
    parser.add_argument(
        "--lab-baseline",
        choices=("global", "local-ring"),
        default="global",
        help=(
            "Reference for the LAB-b delta. 'global' preserves the original whole-image "
            "median; 'local-ring' uses pixels around each YOLO box."
        ),
    )
    parser.add_argument(
        "--lab-ring-scale",
        type=float,
        default=2.0,
        help="Width and height multiplier for the local background ring (must be > 1).",
    )
    parser.add_argument("--target-recall", type=float, default=1.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-decisions", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.target_recall <= 1:
        raise ValueError("Target recall must be between 0 and 1")
    if args.direct_conf <= 0:
        raise ValueError("Direct confidence must be greater than zero")
    if args.lab_ring_scale <= 1:
        raise ValueError("LAB ring scale must be greater than 1")

    samples = load_samples(
        images_dir=args.images,
        gt_labels=args.gt_labels,
        pred_labels=args.pred_labels,
        lab_percentile=args.lab_percentile,
        lab_baseline=args.lab_baseline,
        lab_ring_scale=args.lab_ring_scale,
    )
    baseline = evaluate_configuration(
        samples,
        direct_confidence=args.direct_conf,
        rescue_confidence=0.0,
        lab_threshold=float("inf"),
    )
    lab_thresholds = build_float_range(args.lab_start, args.lab_stop, args.lab_step)
    results = sweep_hybrid(
        samples,
        direct_confidence=args.direct_conf,
        rescue_confidences=args.rescue_conf,
        lab_thresholds=lab_thresholds,
    )
    selected = choose_result(results, target_recall=args.target_recall)
    write_results_csv(results, args.output_csv)
    write_decisions_csv(
        samples,
        direct_confidence=args.direct_conf,
        selected=selected,
        output_path=args.output_decisions,
    )

    print()
    print_result("YOLO-only baseline", baseline)
    print()
    print_result(
        f"Selected hybrid result for target recall >= {args.target_recall:.3f}",
        selected,
    )
    print()
    print(f"Sweep CSV:     {args.output_csv}")
    print(f"Decisions CSV: {args.output_decisions}")


if __name__ == "__main__":
    main()
