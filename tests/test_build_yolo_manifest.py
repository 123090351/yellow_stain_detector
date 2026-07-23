import csv
import tempfile
import unittest
from pathlib import Path

from scripts.data.build_yolo_manifest import build_manifest


class BuildYoloManifestTest(unittest.TestCase):
    def make_dirs(self, root: Path) -> tuple[Path, Path]:
        images = root / "raw" / "images"
        labels = root / "raw" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        return images, labels

    def test_writes_manifest_and_parses_capture_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = self.make_dirs(root)
            ng_stem = "KN112511001_2026-06-24_13-57-06-468"
            ok_stem = "KN112511001_2026-06-24_13-57-07-225"
            (images / f"{ng_stem}.png").write_bytes(b"image")
            (images / f"{ok_stem}.jpg").write_bytes(b"image")
            (labels / f"{ng_stem}.txt").write_text(
                "0 0.5 0.5 0.1 0.2\n", encoding="utf-8"
            )
            (labels / f"{ok_stem}.txt").write_text("", encoding="utf-8")
            output = root / "raw" / "manifest.csv"

            summary = build_manifest(images, labels, output)

            self.assertEqual(
                {
                    "images": 2,
                    "labels": 2,
                    "ng": 1,
                    "empty": 1,
                    "missing_capture_metadata": 0,
                },
                summary,
            )
            with output.open(newline="", encoding="utf-8") as manifest:
                rows = list(csv.DictReader(manifest))
            self.assertEqual(2, len(rows))
            self.assertEqual("NG", rows[0]["label_status"])
            self.assertEqual("EMPTY", rows[1]["label_status"])
            self.assertEqual("2026-06-24T13:57:06.468", rows[0]["capture_time"])
            self.assertEqual("KN112511001_2026-06-24_13", rows[0]["batch"])
            self.assertEqual("unassigned", rows[0]["split"])

    def test_rejects_invalid_yolo_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = self.make_dirs(root)
            (images / "bad.png").write_bytes(b"image")
            (labels / "bad.txt").write_text("0 1.2 0.5 0.1 0.1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "box center must be in"):
                build_manifest(images, labels, root / "manifest.csv")

    def test_rejects_missing_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = self.make_dirs(root)
            (images / "missing.png").write_bytes(b"image")

            with self.assertRaisesRegex(ValueError, "Images without TXT: 1"):
                build_manifest(images, labels, root / "manifest.csv")


if __name__ == "__main__":
    unittest.main()
