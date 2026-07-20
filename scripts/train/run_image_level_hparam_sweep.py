#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.sweep_image_level_thresholds import (  # noqa: E402
    ThresholdResult,
    build_thresholds,
    choose_threshold,
    sweep_thresholds,
    write_csv as write_threshold_csv,
    write_plot as write_threshold_plot,
)


@dataclass(frozen=True)
class SweepRecord:
    run_name: str
    imgsz: int
    freeze: int
    threshold: float
    accuracy: float
    precision: float
    ng_recall: float
    ok_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    target_met: bool
    best_weights: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an imgsz x freeze grid sequentially and rank each run by "
            "image-level OK/NG validation metrics."
        )
    )
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/yellow_stain_v1/data_server.yaml"),
    )
    parser.add_argument(
        "--val-images",
        type=Path,
        default=Path("datasets/yellow_stain_v1/images/val"),
    )
    parser.add_argument(
        "--val-labels",
        type=Path,
        default=Path("datasets/yellow_stain_v1/labels/val"),
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Base output directory. It is resolved to an absolute path.",
    )
    parser.add_argument("--sweep-name", default="imgsz_freeze_sweep")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[512, 640, 768])
    parser.add_argument("--freeze", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--prediction-conf", type=float, default=0.001)
    parser.add_argument("--threshold-start", type=float, default=0.001)
    parser.add_argument("--threshold-stop", type=float, default=0.100)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rerun outputs managed by this sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without loading Ultralytics or training.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_from_root(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def planned_runs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if not args.imgsz or not args.freeze:
        raise ValueError("At least one imgsz and one freeze value are required")
    if any(value <= 0 for value in args.imgsz):
        raise ValueError("All imgsz values must be greater than zero")
    if any(value < 0 for value in args.freeze):
        raise ValueError("Freeze values cannot be negative")
    return list(product(args.imgsz, args.freeze))


def record_from_result(
    run_name: str,
    imgsz: int,
    freeze: int,
    result: ThresholdResult,
    target_recall: float,
    best_weights: Path,
) -> SweepRecord:
    return SweepRecord(
        run_name=run_name,
        imgsz=imgsz,
        freeze=freeze,
        threshold=result.threshold,
        accuracy=result.accuracy,
        precision=result.precision,
        ng_recall=result.recall,
        ok_accuracy=result.ok_accuracy,
        tp=result.true_positive,
        fp=result.false_positive,
        tn=result.true_negative,
        fn=result.false_negative,
        target_met=result.recall >= target_recall,
        best_weights=str(best_weights),
    )


def ranking_key(record: SweepRecord) -> tuple[float, ...]:
    if record.target_met:
        return (
            1.0,
            record.ok_accuracy,
            record.accuracy,
            record.precision,
            record.threshold,
        )
    return (
        0.0,
        record.ng_recall,
        record.ok_accuracy,
        record.accuracy,
        record.threshold,
    )


def write_leaderboard(records: list[SweepRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(records, key=ranking_key, reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "rank",
                "run_name",
                "imgsz",
                "freeze",
                "threshold",
                "accuracy",
                "precision",
                "ng_recall",
                "ok_accuracy",
                "tp",
                "fp",
                "tn",
                "fn",
                "target_met",
                "best_weights",
            )
        )
        for rank, record in enumerate(ranked, start=1):
            writer.writerow(
                (
                    rank,
                    record.run_name,
                    record.imgsz,
                    record.freeze,
                    f"{record.threshold:.4f}",
                    f"{record.accuracy:.4f}",
                    f"{record.precision:.4f}",
                    f"{record.ng_recall:.4f}",
                    f"{record.ok_accuracy:.4f}",
                    record.tp,
                    record.fp,
                    record.tn,
                    record.fn,
                    str(record.target_met).lower(),
                    record.best_weights,
                )
            )


def load_record(path: Path) -> SweepRecord:
    return SweepRecord(**json.loads(path.read_text(encoding="utf-8")))


def save_record(record: SweepRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")


def release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_one(
    args: argparse.Namespace,
    imgsz: int,
    freeze: int,
    sweep_dir: Path,
    thresholds: list[float],
) -> SweepRecord:
    from ultralytics import YOLO

    run_name = f"imgsz{imgsz}_freeze{freeze}_seed{args.seed}"
    train_dir = sweep_dir / "train" / run_name
    predict_dir = sweep_dir / "predict" / run_name
    eval_dir = sweep_dir / "evaluation" / run_name
    result_json = eval_dir / "result.json"

    if result_json.exists() and not args.overwrite:
        print(f"[{run_name}] Reusing completed result")
        return load_record(result_json)

    if args.overwrite:
        for managed_path in (train_dir, predict_dir, eval_dir):
            if managed_path.exists():
                shutil.rmtree(managed_path)

    best_weights = train_dir / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"[{run_name}] Training")
        model = YOLO(args.model)
        model.train(
            data=str(resolve_from_root(args.data)),
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            freeze=freeze,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            bgr=0.0,
            mosaic=0.0,
            close_mosaic=0,
            erasing=0.0,
            project=str(train_dir.parent),
            name=train_dir.name,
            exist_ok=False,
            verbose=args.verbose,
        )
        del model
        release_gpu_memory()
    else:
        print(f"[{run_name}] Reusing {best_weights}")

    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not create {best_weights}")

    prediction_marker = predict_dir / ".complete"
    if not prediction_marker.exists():
        if predict_dir.exists():
            raise RuntimeError(
                f"Incomplete prediction directory exists: {predict_dir}. "
                "Rerun with --overwrite."
            )
        print(f"[{run_name}] Predicting validation images at conf={args.prediction_conf}")
        predictor = YOLO(str(best_weights))
        predictor.predict(
            source=str(resolve_from_root(args.val_images)),
            imgsz=imgsz,
            conf=args.prediction_conf,
            augment=False,
            device=args.device,
            save=False,
            save_txt=True,
            save_conf=True,
            project=str(predict_dir.parent),
            name=predict_dir.name,
            exist_ok=False,
            verbose=args.verbose,
        )
        predict_dir.mkdir(parents=True, exist_ok=True)
        (predict_dir / "labels").mkdir(parents=True, exist_ok=True)
        prediction_marker.write_text("complete\n", encoding="utf-8")
        del predictor
        release_gpu_memory()

    print(f"[{run_name}] Sweeping image-level thresholds")
    threshold_results = sweep_thresholds(
        resolve_from_root(args.val_labels),
        predict_dir / "labels",
        thresholds,
    )
    selected = choose_threshold(threshold_results, args.target_recall)
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_threshold_csv(threshold_results, eval_dir / "threshold_sweep.csv")
    write_threshold_plot(threshold_results, eval_dir / "threshold_sweep.png")

    record = record_from_result(
        run_name,
        imgsz,
        freeze,
        selected,
        args.target_recall,
        best_weights,
    )
    save_record(record, result_json)
    print(
        f"[{run_name}] threshold={record.threshold:.4f} "
        f"recall={record.ng_recall:.3f} FP={record.fp} FN={record.fn}"
    )
    return record


def validate_inputs(args: argparse.Namespace) -> None:
    for label, path in (
        ("data YAML", resolve_from_root(args.data)),
        ("validation images", resolve_from_root(args.val_images)),
        ("validation labels", resolve_from_root(args.val_labels)),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not 0.0 <= args.target_recall <= 1.0:
        raise ValueError("target-recall must be between 0 and 1")
    if not 0.0 < args.prediction_conf <= 1.0:
        raise ValueError("prediction-conf must be between 0 and 1")


def build_sweep_config(
    args: argparse.Namespace,
    runs: list[tuple[int, int]],
) -> dict[str, object]:
    return {
        "model": args.model,
        "data": str(resolve_from_root(args.data)),
        "val_images": str(resolve_from_root(args.val_images)),
        "val_labels": str(resolve_from_root(args.val_labels)),
        "runs": [{"imgsz": imgsz, "freeze": freeze} for imgsz, freeze in runs],
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "device": args.device,
        "prediction_conf": args.prediction_conf,
        "threshold_start": args.threshold_start,
        "threshold_stop": args.threshold_stop,
        "threshold_step": args.threshold_step,
        "target_recall": args.target_recall,
        "augmentations": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "bgr": 0.0,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "erasing": 0.0,
        },
    }


def prepare_sweep_directory(
    sweep_dir: Path,
    config: dict[str, object],
    overwrite: bool,
) -> None:
    config_path = sweep_dir / "config.json"
    if overwrite and sweep_dir.exists():
        shutil.rmtree(sweep_dir)

    if sweep_dir.exists() and not config_path.exists():
        raise RuntimeError(
            f"Output directory exists without a sweep config: {sweep_dir}. "
            "Use a different --sweep-name or explicitly rerun with --overwrite."
        )

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                f"Sweep settings differ from {config_path}. Use a different "
                "--sweep-name or explicitly rerun with --overwrite."
            )

    sweep_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = planned_runs(args)
    project = resolve_from_root(args.project)
    sweep_dir = project / args.sweep_name

    print(f"Planned runs: {len(runs)}")
    print(f"Output: {sweep_dir}")
    for index, (imgsz, freeze) in enumerate(runs, start=1):
        print(f"  {index:>2}. imgsz={imgsz}, freeze={freeze}, seed={args.seed}")
    if args.dry_run:
        return

    validate_inputs(args)
    prepare_sweep_directory(sweep_dir, build_sweep_config(args, runs), args.overwrite)
    thresholds = build_thresholds(
        args.threshold_start,
        args.threshold_stop,
        args.threshold_step,
    )
    records: list[SweepRecord] = []
    failures: list[tuple[str, str]] = []

    for index, (imgsz, freeze) in enumerate(runs, start=1):
        run_name = f"imgsz{imgsz}_freeze{freeze}_seed{args.seed}"
        print(f"\n=== Run {index}/{len(runs)}: {run_name} ===")
        try:
            records.append(run_one(args, imgsz, freeze, sweep_dir, thresholds))
            write_leaderboard(records, sweep_dir / "leaderboard.csv")
        except Exception as error:  # keep an unattended queue moving
            failures.append((run_name, str(error)))
            print(f"[{run_name}] FAILED: {error}", file=sys.stderr)
            traceback.print_exc()
            release_gpu_memory()

    if failures:
        failure_path = sweep_dir / "failures.csv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("run_name", "error"))
            writer.writerows(failures)
        print(f"\nFailures: {len(failures)} (see {failure_path})")
    else:
        failure_path = sweep_dir / "failures.csv"
        if failure_path.exists():
            failure_path.unlink()

    if not records:
        raise RuntimeError("All sweep runs failed")

    ranked = sorted(records, key=ranking_key, reverse=True)
    best = ranked[0]
    print(f"\nLeaderboard: {sweep_dir / 'leaderboard.csv'}")
    print("Best validation configuration:")
    print(f"  run:         {best.run_name}")
    print(f"  imgsz:       {best.imgsz}")
    print(f"  freeze:      {best.freeze}")
    print(f"  threshold:   {best.threshold:.4f}")
    print(f"  accuracy:    {best.accuracy:.3f}")
    print(f"  precision:   {best.precision:.3f}")
    print(f"  NG recall:   {best.ng_recall:.3f}")
    print(f"  OK accuracy: {best.ok_accuracy:.3f}")
    print(f"  TP/FP/TN/FN: {best.tp}/{best.fp}/{best.tn}/{best.fn}")


if __name__ == "__main__":
    main()
