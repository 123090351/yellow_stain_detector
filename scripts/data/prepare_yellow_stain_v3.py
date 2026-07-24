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


@dataclass(frozen=True)
class BuildPlan:
    records: list[DatasetRecord]
    skipped_overlap: int


def _index_by_stem(paths: list[Path], kind: str) -> dict[str, Path]:
    counts = Counter(path.stem for path in paths)
    duplicates = sorted(stem for stem, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate {kind} stems: {len(duplicates)}; "
            f"samples: {', '.join(duplicates[:10])}"
        )
    return {path.stem: path for path in paths}


def _label_status(path: Path) -> str:
    return "NG" if path.read_text(encoding="utf-8").strip() else "OK"


def load_base_records(base_dataset: Path) -> list[DatasetRecord]:
    base_dataset = base_dataset.resolve()
    records: list[DatasetRecord] = []
    seen: set[str] = set()
    for split in SPLITS:
        image_dir = base_dataset / "images" / split
        label_dir = base_dataset / "labels" / split
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
        images_by_stem = _index_by_stem(images, f"base {split} image")
        labels_by_stem = _index_by_stem(labels, f"base {split} label")
        if images_by_stem.keys() != labels_by_stem.keys():
            raise ValueError(
                f"Base {split} pairing mismatch: "
                f"images without TXT={len(images_by_stem.keys() - labels_by_stem.keys())}, "
                f"TXT without image={len(labels_by_stem.keys() - images_by_stem.keys())}"
            )
        overlap = seen & images_by_stem.keys()
        if overlap:
            raise ValueError(
                f"Base image stems occur in multiple splits: {len(overlap)}; "
                f"samples: {', '.join(sorted(overlap)[:10])}"
            )
        seen.update(images_by_stem)
        records.extend(
            DatasetRecord(
                image_id=stem,
                image_path=images_by_stem[stem],
                label_path=labels_by_stem[stem],
                label_status=_label_status(labels_by_stem[stem]),
                source="v2_base",
                batch="existing_v2_split",
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


def _hour_split_map(
    train_hours: list[int],
    val_hours: list[int],
    test_hours: list[int],
) -> dict[int, str]:
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
    return {hour: split for split, hours in split_hours.items() for hour in hours}


def build_plan(
    base_dataset: Path,
    ok_manifest: Path,
    train_hours: list[int],
    val_hours: list[int],
    test_hours: list[int],
) -> BuildPlan:
    base_records = load_base_records(base_dataset)
    base_by_stem = {record.image_id: record for record in base_records}
    hour_splits = _hour_split_map(train_hours, val_hours, test_hours)
    ok_manifest = ok_manifest.resolve()
    required = {
        "image_id",
        "image_path",
        "label_path",
        "label_status",
        "capture_time",
        "batch",
    }
    new_records: list[DatasetRecord] = []
    manifest_stems: set[str] = set()
    skipped_overlap = 0

    with ok_manifest.open(newline="", encoding="utf-8") as manifest:
        reader = csv.DictReader(manifest)
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "Manifest is missing columns: " + ", ".join(sorted(missing_columns))
            )
        for row in reader:
            image_id = row["image_id"]
            if image_id in manifest_stems:
                raise ValueError(f"Duplicate image_id in OK manifest: {image_id}")
            manifest_stems.add(image_id)
            if row["label_status"] != "OK":
                raise ValueError(
                    f"Expected reviewed OK manifest; "
                    f"{image_id} is {row['label_status']!r}"
                )

            existing = base_by_stem.get(image_id)
            if existing is not None:
                if existing.label_status != "OK":
                    raise ValueError(
                        f"Overlapping stem is NG in base dataset: {image_id}"
                    )
                skipped_overlap += 1
                continue

            image_path = (ok_manifest.parent / row["image_path"]).resolve()
            label_path = (ok_manifest.parent / row["label_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            if label_path.read_text(encoding="utf-8").strip():
                raise ValueError(f"New OK label is not empty: {label_path}")

            hour = _hour_from_capture_time(row["capture_time"])
            if hour is None:
                raise ValueError(f"OK filename has no parseable capture time: {image_id}")
            try:
                split = hour_splits[hour]
            except KeyError as exc:
                raise ValueError(
                    f"No split configured for capture hour {hour:02d}: {image_id}"
                ) from exc
            new_records.append(
                DatasetRecord(
                    image_id=image_id,
                    image_path=image_path,
                    label_path=label_path,
                    label_status="OK",
                    source="v3_new_ok",
                    batch=row["batch"],
                    split=split,
                )
            )

    return BuildPlan(
        records=base_records + new_records,
        skipped_overlap=skipped_overlap,
    )


def summarize(records: list[DatasetRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        selected = [record for record in records if record.split == split]
        ng = sum(record.label_status == "NG" for record in selected)
        new_ok = sum(record.source == "v3_new_ok" for record in selected)
        summary[split] = {
            "total": len(selected),
            "ng": ng,
            "ok": len(selected) - ng,
            "new_ok": new_ok,
        }
    return summary


def print_summary(
    summary: dict[str, dict[str, int]], skipped_overlap: int
) -> None:
    print(f"Skipped OK stems already present in base dataset: {skipped_overlap}")
    print("split,total,NG,OK,new_OK")
    for split in SPLITS:
        values = summary[split]
        print(
            f"{split},{values['total']},{values['ng']},"
            f"{values['ok']},{values['new_ok']}"
        )


def _materialize(source: Path, target: Path, link_mode: str) -> None:
    if link_mode == "copy":
        shutil.copy2(source, target)
    elif link_mode == "hardlink":
        os.link(source, target)
    else:
        target.symlink_to(source)


def _write_metadata(
    staging: Path,
    final_output: Path,
    records: list[DatasetRecord],
    summary: dict[str, dict[str, int]],
    skipped_overlap: int,
) -> None:
    yaml_lines = [
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        "  0: huangban",
        "",
    ]
    (staging / "data.yaml").write_text(
        "\n".join(["path: .", *yaml_lines]), encoding="utf-8"
    )
    (staging / "data_server.yaml").write_text(
        "\n".join([f"path: {final_output.resolve()}", *yaml_lines]),
        encoding="utf-8",
    )

    report = [
        "# Yellow Stain v3 Dataset Report",
        "",
        f"Skipped OK stems already present in v2: {skipped_overlap}",
        "",
        "| split | total | NG | OK | new OK |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        values = summary[split]
        report.append(
            f"| {split} | {values['total']} | {values['ng']} | "
            f"{values['ok']} | {values['new_ok']} |"
        )
    report.extend(
        [
            "",
            "The existing v2 split is preserved.",
            "New unique OK images are assigned by continuous capture-hour blocks.",
            "",
        ]
    )
    (staging / "dataset_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    fieldnames = [
        "image_id",
        "image_path",
        "label_path",
        "label_status",
        "source",
        "batch",
        "split",
    ]
    with (staging / "dataset_manifest.csv").open(
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
    plan: BuildPlan,
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
        for index, record in enumerate(plan.records, start=1):
            image_target = staging / "images" / record.split / record.image_path.name
            label_target = staging / "labels" / record.split / record.label_path.name
            if image_target.exists() or label_target.exists():
                raise ValueError(f"Output filename collision: {record.image_id}")
            _materialize(record.image_path, image_target, link_mode)
            _materialize(record.label_path, label_target, link_mode)
            if index % 250 == 0:
                print(f"Materialized {index}/{len(plan.records)}")
        summary = summarize(plan.records)
        _write_metadata(
            staging, output, plan.records, summary, plan.skipped_overlap
        )
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build yellow_stain_v3 by extending v2 with deduplicated OK images."
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--ok-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-hours", type=int, nargs="+", default=[8, 9, 10, 11, 13]
    )
    parser.add_argument("--val-hours", type=int, nargs="+", default=[14])
    parser.add_argument("--test-hours", type=int, nargs="+", default=[15, 16])
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
    plan = build_plan(
        base_dataset=args.base_dataset,
        ok_manifest=args.ok_manifest,
        train_hours=args.train_hours,
        val_hours=args.val_hours,
        test_hours=args.test_hours,
    )
    summary = summarize(plan.records)
    print_summary(summary, plan.skipped_overlap)
    if args.dry_run:
        print("Dry run only; no files were written.")
        return
    materialize_dataset(
        output=args.output,
        plan=plan,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(f"Wrote dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
