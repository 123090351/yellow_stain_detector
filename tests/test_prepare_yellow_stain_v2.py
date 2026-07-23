import csv
import tempfile
import unittest
from pathlib import Path

from scripts.data.prepare_yellow_stain_v2 import (
    build_plan,
    materialize_dataset,
    summarize,
)


class PrepareYellowStainV2Test(unittest.TestCase):
    def _write_pair(
        self, image_dir: Path, label_dir: Path, stem: str, ng: bool
    ) -> None:
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{stem}.png").write_bytes(b"image")
        (label_dir / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n" if ng else "", encoding="utf-8"
        )

    def _make_v1(self, root: Path) -> Path:
        dataset = root / "v1"
        self._write_pair(
            dataset / "images" / "train",
            dataset / "labels" / "train",
            "old_train_ng",
            True,
        )
        self._write_pair(
            dataset / "images" / "val",
            dataset / "labels" / "val",
            "old_val_ok",
            False,
        )
        self._write_pair(
            dataset / "images" / "test",
            dataset / "labels" / "test",
            "old_test_ok",
            False,
        )
        return dataset

    def _make_new_manifest(self, root: Path, include_unparsed: bool) -> Path:
        raw = root / "raw"
        rows = []
        for hour in (13, 14, 15, 16):
            stem = f"camera_2026-06-24_{hour:02d}-00-00-000"
            self._write_pair(raw / "yellow", raw / "labels", stem, True)
            rows.append(
                {
                    "image_id": stem,
                    "image_path": f"yellow/{stem}.png",
                    "label_path": f"labels/{stem}.txt",
                    "label_status": "NG",
                    "capture_time": f"2026-06-24T{hour:02d}:00:00.000",
                    "batch": f"camera_2026-06-24_{hour:02d}",
                }
            )
        if include_unparsed:
            stem = "manual_name"
            self._write_pair(raw / "yellow", raw / "labels", stem, True)
            rows.append(
                {
                    "image_id": stem,
                    "image_path": f"yellow/{stem}.png",
                    "label_path": f"labels/{stem}.txt",
                    "label_status": "NG",
                    "capture_time": "",
                    "batch": "",
                }
            )
        manifest = raw / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_builds_expected_hour_split_and_materializes_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = build_plan(
                self._make_v1(root),
                self._make_new_manifest(root, include_unparsed=True),
                train_hours=[14, 15],
                val_hours=[13],
                test_hours=[16],
                unparsed_split="train",
            )

            summary = summarize(records)

            self.assertEqual(
                {"total": 4, "ng": 4, "ok": 0, "new_ng": 3},
                summary["train"],
            )
            self.assertEqual(
                {"total": 2, "ng": 1, "ok": 1, "new_ng": 1},
                summary["val"],
            )
            output = root / "v2"
            materialize_dataset(output, records, link_mode="copy", overwrite=False)
            self.assertTrue((output / "data_server.yaml").is_file())
            self.assertIn(
                f"path: {output.resolve()}",
                (output / "data_server.yaml").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                4, len(list((output / "images" / "train").iterdir()))
            )

    def test_requires_explicit_handling_for_unparsed_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "without capture time: 1"):
                build_plan(
                    self._make_v1(root),
                    self._make_new_manifest(root, include_unparsed=True),
                    train_hours=[14, 15],
                    val_hours=[13],
                    test_hours=[16],
                    unparsed_split=None,
                )

    def test_rejects_hour_assigned_to_multiple_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                build_plan(
                    self._make_v1(root),
                    self._make_new_manifest(root, include_unparsed=False),
                    train_hours=[14],
                    val_hours=[14],
                    test_hours=[16],
                    unparsed_split=None,
                )


if __name__ == "__main__":
    unittest.main()
