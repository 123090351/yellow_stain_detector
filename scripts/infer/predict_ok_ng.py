#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def collect_images(source: Path) -> list[Path]:
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found under: {source}")
    return images


def classify_confidences(
    confidences: Iterable[float], threshold: float
) -> tuple[str, int, float]:
    accepted = [float(value) for value in confidences if float(value) >= threshold]
    if not accepted:
        return "OK", 0, 0.0
    return "NG", len(accepted), max(accepted)


def relative_output_path(image: Path, source: Path) -> Path:
    source = source.resolve()
    image = image.resolve()
    if source.is_dir():
        return image.relative_to(source)
    return Path(image.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Yellow Stain v3 inference and write image-level OK/NG CSV."
    )
    parser.add_argument("--model", type=Path, default=Path("/model/best.pt"))
    parser.add_argument("--source", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--threshold", type=float, default=0.381)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    images = collect_images(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output / "annotated"

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    predictions = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.threshold,
        device=args.device,
        stream=True,
        save=False,
        verbose=False,
    )

    rows: list[dict[str, str | int]] = []
    counts = {"OK": 0, "NG": 0}
    for image, result in zip(images, predictions, strict=True):
        confidences = (
            result.boxes.conf.detach().cpu().tolist()
            if result.boxes is not None
            else []
        )
        decision, detections, max_confidence = classify_confidences(
            confidences, args.threshold
        )
        counts[decision] += 1
        relative_path = relative_output_path(image, args.source)
        rows.append(
            {
                "image": relative_path.as_posix(),
                "decision": decision,
                "detections": detections,
                "max_confidence": f"{max_confidence:.6f}",
                "threshold": f"{args.threshold:.6f}",
            }
        )
        if args.save_images:
            destination = annotated_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(destination))

    csv_path = args.output / "decisions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "image",
                "decision",
                "detections",
                "max_confidence",
                "threshold",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": str(args.model.resolve()),
        "source": str(args.source.resolve()),
        "images": len(rows),
        "ok": counts["OK"],
        "ng": counts["NG"],
        "imgsz": args.imgsz,
        "threshold": args.threshold,
        "device": str(args.device),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Decisions: {csv_path}")


if __name__ == "__main__":
    main()
