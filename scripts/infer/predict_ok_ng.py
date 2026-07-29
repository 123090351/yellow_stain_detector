#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CSV_FIELDS = (
    "image",
    "decision",
    "detections",
    "max_confidence",
    "threshold",
)


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


def load_completed(csv_path: Path) -> tuple[set[str], dict[str, int]]:
    completed: set[str] = set()
    counts = {"OK": 0, "NG": 0}
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return completed, counts
    with csv_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            image = row.get("image", "")
            decision = row.get("decision", "")
            if image and decision in counts and image not in completed:
                completed.add(image)
                counts[decision] += 1
    return completed, counts


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    processed: int,
    total: int,
    counts: dict[str, int],
) -> None:
    summary = {
        "model": str(args.model.resolve()),
        "source": str(args.source.resolve()),
        "images": processed,
        "total_images": total,
        "remaining_images": total - processed,
        "complete": processed == total,
        "ok": counts["OK"],
        "ng": counts["NG"],
        "imgsz": args.imgsz,
        "threshold": args.threshold,
        "device": str(args.device),
        "batch_size": args.batch_size,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum images handed to Ultralytics at once (default: 4).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in decisions.csv.",
    )
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    images = collect_images(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output / "annotated"
    csv_path = args.output / "decisions.csv"
    summary_path = args.output / "summary.json"

    completed, counts = (
        load_completed(csv_path) if args.resume else (set(), {"OK": 0, "NG": 0})
    )
    pending = [
        image
        for image in images
        if relative_output_path(image, args.source).as_posix() not in completed
    ]
    print(
        f"Found {len(images)} images; completed={len(completed)}, "
        f"remaining={len(pending)}, batch_size={args.batch_size}",
        flush=True,
    )

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    csv_mode = (
        "a"
        if args.resume and csv_path.is_file() and csv_path.stat().st_size
        else "w"
    )
    with csv_path.open(csv_mode, newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        if csv_mode == "w":
            writer.writeheader()
            output.flush()

        for start in range(0, len(pending), args.batch_size):
            batch_images = pending[start : start + args.batch_size]
            results = model.predict(
                source=[str(path) for path in batch_images],
                imgsz=args.imgsz,
                conf=args.threshold,
                device=args.device,
                batch=args.batch_size,
                stream=False,
                save=False,
                verbose=False,
            )
            if len(results) != len(batch_images):
                raise RuntimeError(
                    f"Expected {len(batch_images)} results, got {len(results)}"
                )

            for image, result in zip(batch_images, results):
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
                writer.writerow(
                    {
                        "image": relative_path.as_posix(),
                        "decision": decision,
                        "detections": detections,
                        "max_confidence": f"{max_confidence:.6f}",
                        "threshold": f"{args.threshold:.6f}",
                    }
                )
                completed.add(relative_path.as_posix())
                if args.save_images:
                    destination = annotated_dir / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    result.save(filename=str(destination))

            output.flush()
            os.fsync(output.fileno())
            write_summary(
                summary_path,
                args=args,
                processed=len(completed),
                total=len(images),
                counts=counts,
            )
            print(
                f"Processed {len(completed)}/{len(images)} "
                f"(OK={counts['OK']}, NG={counts['NG']})",
                flush=True,
            )

    write_summary(
        summary_path,
        args=args,
        processed=len(completed),
        total=len(images),
        counts=counts,
    )
    print(summary_path.read_text(encoding="utf-8"), end="", flush=True)
    print(f"Decisions: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
