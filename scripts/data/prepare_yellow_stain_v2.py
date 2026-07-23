#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetRecord:
    image_id: str
    image_path: Path
    label_path: Path
    label_status: str
    source: str
    batch: str
    split: str


def _index_by_stem(paths: list[Path], kind: str) -> dict[str, Path]:
    counts = Counter(path.stem for path in paths)
    duplicates = sorted(stem for stem, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate {kind} stems: {len(duplicates)}; samples: {', '.join(duplicates[:10])}"
        )
    return {path.stem: path for path in paths}


def _label_status(path: Path) -> str:
    return "NG" if path.read_text(encoding="utf-8").strip() else "OK"


def load_v1_records(v1_dataset: Path) -> list[DatasetRecord]:
    v1_dataset = v1_dataset.resolve()
    records: list[DatasetRecord] = []
    for split in SPLITS:
        image_dir = v1_dataset / "images" / split
        label_dir = v1_dataset / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        if not label_dir.is_dir():
            raise FileNotFoundError(label_dir)
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        )
        images_by_stem = _index_by_stem(images, f"v1 {split} image")
        labels_by_stem = _index_by_stem(labels, f"v1 {split} label")
        if images_by_stem.keys() != labels_by_stem.keys():
            missing_labels = sorted(images_by_stem.keys() - labels_by_stem.keys())
            missing_images = sorted(labels_by_stem.keys() - images_by_stem.keys())
            raise ValueError(
                f"v1 {split} pairing mismatch: images without TXT={len(missing_labels)}, "
                f"TXT without image={len(missing_images)}"
            )
        records.extend(
            DatasetRecord(
                image_id=stem,
                image_path=images_by_stem[stem],
                label_path=labels_by_stem[stem],
                label_status=_label_status(labels_by_stem[stem]),
                source="v1",
                batch="existing_v1_split",
                split=split,
            )
            for stem in sorted(images_by_stem)
        )
    return records


def _parse_hours(values: list[int]) -> set[int]:
    hours = set(values)
    invalid = sorted(hour for hour in hours if not 0 <= hour <= 23)
    if invalid:
        raise ValueError(f"Hours must be in [0, 23]: {invalid}")
    return hours


def _hour_from_capture_time(capture_time: str) -> int | None:
    if len(capture_time) < 13 or capture_time[10] != "T":
        return None
    try:
        hour = int(capture_time[11:13])
    except ValueError:
        return None
    return hour if 0 <= hour <= 23 else None


def load_new_records(
    manifest_path: Path,
    hour_splits: dict[int, str],
    unparsed_split: str | None,
) -> list[DatasetRecord]:
    manifest_path = manifest_path.resolve()
    required = {
        "image_id",
        "image_path",
        "label_path",
        "label_status",
        "capture_time",
        "batch",
    }
    records: list[DatasetRecord] = []
    unparsed: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8") as manifest:
        reader = csv.DictReader(manifest)
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "Manifest is missing columns: " + ", ".join(sorted(missing_columns))
            )
        for row in reader:
            image_id = row["image_id"]
            if row["label_status"] != "NG":
                raise ValueError(
                    f"New dataset must contain only reviewed NG labels; "
                    f"{image_id} is {row['label_status']!r}"
                )
            image_path = (manifest_path.parent / row["image_path"]).resolve()
            label_path = (manifest_path.parent / row["label_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            hour = _hour_from_capture_time(row["capture_time"])
            if hour is None:
                unparsed.append(image_id)
                if unparsed_split is None:
                    continue
                split = unparsed_split
            else:
                try:
                    split = hour_splits[hour]
                except KeyError as exc:
                    raise ValueError(
                        f"No split configured for capture hour {hour:02d}: {image_id}"
                    ) from exc
            records.append(
                DatasetRecord(
                    image_id=image_id,
                    image_path=image_path,
                    label_path=label_path,
                    label_status="NG",
                    source="v2_new",
                    batch=row["batch"] or "unparsed",
                    split=split,
                )
            )
    if unparsed and unparsed_split is None:
        raise ValueError(
            f"Filenames without capture time: {len(unparsed)}; "
            f"samples: {', '.join(unparsed[:10])}. "
            "Inspect them, then pass --unparsed-split train if they may be used for training."
        )
    return records


def build_plan(
    v1_dataset: Path,
    new_manifest: Path,
    train_hours: list[int],
    val_hours: list[int],
    test_hours: list[int],
    unparsed_split: str | None,
) -> list[DatasetRecord]:
    split_hours = {
        "train": _parse_hours(train_hours),
        "val": _parse_hours(val_hours),
        "test": _parse_hours(test_hours),
    }
    all_hours = [hour for hours in split_hours.values() for hour in hours]
    duplicates = sorted(
        hour for hour, count in Counter(all_hours).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Capture hours assigned to multiple splits: {duplicates}")
    hour_splits = {
        hour: split for split, hours in split_hours.items() for hour in hours
    }

    old_records = load_v1_records(v1_dataset)
    new_records = load_new_records(new_manifest, hour_splits, unparsed_split)
    old_stems = {record.image_id for record in old_records}
    new_stems = {record.image_id for record in new_records}
    overlap = sorted(old_stems & new_stems)
    if overlap:
        raise ValueError(
            f"Image stems overlap between v1 and new data: {len(overlap)}; "
            f"samples: {', '.join(overlap[:10])}"
        )
    if len(new_stems) != len(new_records):
        raise ValueError("New manifest contains duplicate image_id values")
    return old_records + new_records


def summarize(records: list[DatasetRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_records = [record for record in records if record.split == split]
        ng_count = sum(record.label_status == "NG" for record in split_records)
        new_count = sum(record.source == "v2_new" for record in split_records)
        summary[split] = {
            "total": len(split_records),
            "ng": ng_count,
            "ok": len(split_records) - ng_count,
            "new_ng": new_count,
        }
    return summary


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    print("split,total,NG,OK,new_NG")
    for split in SPLITS:
        values = summary[split]
        print(
            f"{split},{values['total']},{values['ng']},"
            f"{values['ok']},{values['new_ng']}"
        )


def _materialize(source: Path, target: Path, link_mode: str) -> None:
    if link_mode == "copy":
        shutil.copy2(source, target)
    elif link_mode == "hardlink":
        os.link(source, target)
    else:
        target.symlink_to(source)


def _write_metadata(
    output: Path,
    final_output: Path,
    records: list[DatasetRecord],
    summary: dict[str, dict[str, int]],
) -> None:
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
    (output / "data_server.yaml").write_text(
        "\n".join(
            [
                f"path: {final_output.resolve()}",
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
    report = [
        "# Yellow Stain v2 Dataset Report",
        "",
        "| split | total | NG | OK | new NG |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        values = summary[split]
        report.append(
            f"| {split} | {values['total']} | {values['ng']} | "
            f"{values['ok']} | {values['new_ng']} |"
        )
    report.extend(
        [
            "",
            "The existing v1 split is preserved.",
            "New NG images are assigned by capture hour, never per-image random split.",
            "",
        ]
    )
    (output / "dataset_report.md").write_text("\n".join(report), encoding="utf-8")

    fieldnames = [
        "image_id",
        "image_path",
        "label_path",
        "label_status",
        "source",
        "batch",
        "split",
    ]
    with (output / "dataset_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item.split, item.image_id)):
            row = asdict(record)
            row["image_path"] = str(record.image_path)
            row["label_path"] = str(record.label_path)
            writer.writerow(row)


def materialize_dataset(
    output: Path,
    records: list[DatasetRecord],
    link_mode: str,
    overwrite: bool,
) -> None:
    output = output.resolve()
    staging = output.with_name(f"{output.name}.building")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    if staging.exists():
        if not overwrite:
            raise FileExistsError(
                f"{staging} exists from an earlier run; pass --overwrite to replace it"
            )
        shutil.rmtree(staging)
    try:
        for split in SPLITS:
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(records, start=1):
            image_target = staging / "images" / record.split / record.image_path.name
            label_target = staging / "labels" / record.split / record.label_path.name
            if image_target.exists() or label_target.exists():
                raise ValueError(f"Output filename collision: {record.image_id}")
            _materialize(record.image_path, image_target, link_mode)
            _materialize(record.label_path, label_target, link_mode)
            if index % 250 == 0:
                print(f"Materialized {index}/{len(records)}")
        summary = summarize(records)
        _write_metadata(staging, output, records, summary)
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build yellow_stain_v2 by preserving v1 splits and assigning new NG by capture hour."
    )
    parser.add_argument("--v1-dataset", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-hours", type=int, nargs="+", default=[14, 15])
    parser.add_argument("--val-hours", type=int, nargs="+", default=[13])
    parser.add_argument("--test-hours", type=int, nargs="+", default=[16])
    parser.add_argument("--unparsed-split", choices=SPLITS)
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_plan(
        v1_dataset=args.v1_dataset,
        new_manifest=args.new_manifest,
        train_hours=args.train_hours,
        val_hours=args.val_hours,
        test_hours=args.test_hours,
        unparsed_split=args.unparsed_split,
    )
    summary = summarize(records)
    print_summary(summary)
    if args.dry_run:
        print("Dry run only; no files were written.")
        return
    materialize_dataset(
        output=args.output,
        records=records,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(f"Wrote dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
