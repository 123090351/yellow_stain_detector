import csv
import tempfile
import unittest
from pathlib import Path

from scripts.data.prepare_yellow_stain_v3 import (
    build_plan,
    materialize_dataset,
    summarize,
)


class PrepareYellowStainV3Test(unittest.TestCase):
    def _write_pair(
        self, image_dir: Path, label_dir: Path, stem: str, ng: bool
    ) -> None:
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{stem}.png").write_bytes(stem.encode())
        (label_dir / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n" if ng else "", encoding="utf-8"
        )

    def _make_base(self, root: Path) -> Path:
        dataset = root / "v2"
        self._write_pair(
            dataset / "images" / "train",
            dataset / "labels" / "train",
            "old_ng",
            True,
        )
        self._write_pair(
            dataset / "images" / "val",
            dataset / "labels" / "val",
            "existing_ok",
            False,
        )
        self._write_pair(
            dataset / "images" / "test",
            dataset / "labels" / "test",
            "old_test_ok",
            False,
        )
        return dataset

    def _make_manifest(self, root: Path) -> Path:
        raw = root / "raw"
        rows = []
        for stem, hour in (
            ("existing_ok", 8),
            ("new_train_ok", 9),
            ("new_val_ok", 14),
            ("new_test_ok", 15),
        ):
            self._write_pair(raw / "ok", raw / "labels", stem, False)
            rows.append(
                {
                    "image_id": stem,
                    "image_path": f"ok/{stem}.png",
                    "label_path": f"labels/{stem}.txt",
                    "label_status": "OK",
                    "capture_time": f"2026-06-22T{hour:02d}:00:00.000",
                    "batch": f"camera_2026-06-22_{hour:02d}",
                }
            )
        manifest = raw / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_skips_existing_ok_and_assigns_new_ok_by_hour(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan(
                self._make_base(root),
                self._make_manifest(root),
                train_hours=[8, 9, 10, 11, 13],
                val_hours=[14],
                test_hours=[15, 16],
            )

            self.assertEqual(1, plan.skipped_overlap)
            summary = summarize(plan.records)
            self.assertEqual(
                {"total": 2, "ng": 1, "ok": 1, "new_ok": 1},
                summary["train"],
            )
            self.assertEqual(
                {"total": 2, "ng": 0, "ok": 2, "new_ok": 1},
                summary["val"],
            )

            output = root / "v3"
            materialize_dataset(output, plan, link_mode="copy", overwrite=False)
            self.assertIn(
                f"path: {output.resolve()}",
                (output / "data_server.yaml").read_text(encoding="utf-8"),
            )
            self.assertTrue((output / "dataset_manifest.csv").is_file())

    def test_rejects_overlap_that_is_ng_in_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._make_base(root)
            manifest = self._make_manifest(root)
            with manifest.open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[0]["image_id"] = "old_ng"
            rows[0]["image_path"] = "ok/existing_ok.png"
            rows[0]["label_path"] = "labels/existing_ok.txt"
            with manifest.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "is NG in base"):
                build_plan(
                    base,
                    manifest,
                    train_hours=[8, 9, 10, 11, 13],
                    val_hours=[14],
                    test_hours=[15, 16],
                )

    def test_rejects_hour_assigned_to_multiple_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                build_plan(
                    self._make_base(root),
                    self._make_manifest(root),
                    train_hours=[9],
                    val_hours=[9, 14],
                    test_hours=[15],
                )


if __name__ == "__main__":
    unittest.main()
