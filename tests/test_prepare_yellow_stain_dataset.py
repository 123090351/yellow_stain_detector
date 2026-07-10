import tempfile
import unittest
from pathlib import Path

from scripts.data.prepare_yellow_stain_dataset import (
    build_records,
    prepare_dataset,
    split_records,
)


class PrepareYellowStainDatasetTest(unittest.TestCase):
    def make_source(self, root: Path, positives: int = 10, negatives: int = 10) -> Path:
        source = root / "training_data" / "training_data"
        images = source / "images"
        labels = source / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        for index in range(positives):
            stem = f"ng_{index:03d}"
            (images / f"{stem}.png").write_bytes(b"fake image")
            (labels / f"{stem}.txt").write_text(
                "0 0.500000 0.500000 0.100000 0.200000\n",
                encoding="utf-8",
            )

        for index in range(negatives):
            stem = f"ok_{index:03d}"
            (images / f"{stem}.jpg").write_bytes(b"fake image")
            (labels / f"{stem}.txt").write_text("", encoding="utf-8")

        return source

    def test_build_records_classifies_blank_labels_as_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp), positives=3, negatives=4)

            records = build_records(source)

            self.assertEqual(7, len(records))
            self.assertEqual(3, sum(record.is_positive for record in records))
            self.assertEqual(4, sum(not record.is_positive for record in records))

    def test_split_records_is_stratified_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp), positives=20, negatives=20)
            records = build_records(source)

            splits = split_records(records, seed=7)

            self.assertEqual({"train", "val", "test"}, set(splits))
            self.assertEqual(40, sum(len(items) for items in splits.values()))
            for split_name, items in splits.items():
                self.assertGreater(sum(record.is_positive for record in items), 0, split_name)
                self.assertGreater(sum(not record.is_positive for record in items), 0, split_name)

            all_stems = [record.stem for items in splits.values() for record in items]
            self.assertEqual(len(all_stems), len(set(all_stems)))

    def test_prepare_dataset_writes_yolo_layout_yaml_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = self.make_source(tmp_path, positives=10, negatives=10)
            output = tmp_path / "datasets" / "yellow_stain_v1"

            summary = prepare_dataset(source, output, seed=3)

            self.assertEqual(20, summary["total"])
            self.assertEqual(10, summary["positive"])
            self.assertEqual(10, summary["negative"])
            self.assertTrue((output / "data.yaml").exists())
            self.assertTrue((output / "dataset_report.txt").exists())

            for split in ("train", "val", "test"):
                image_count = len(list((output / "images" / split).iterdir()))
                label_count = len(list((output / "labels" / split).iterdir()))
                self.assertEqual(image_count, label_count, split)
                self.assertGreater(image_count, 0, split)

            data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("train: images/train", data_yaml)
            self.assertIn("0: huangban", data_yaml)


if __name__ == "__main__":
    unittest.main()
