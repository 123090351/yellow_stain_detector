import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.eval.export_misclassified_review import export_review_package


class ExportMisclassifiedReviewTest(unittest.TestCase):
    def test_cli_help_runs_from_script_path(self):
        script = Path(__file__).parents[1] / "scripts" / "eval" / "export_misclassified_review.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--output-dir", result.stdout)

    def test_exports_false_negatives_false_positives_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            gt_labels = root / "gt"
            pred_labels = root / "pred"
            prediction_images = root / "prediction_images"
            output = root / "review"
            for directory in (images, gt_labels, pred_labels, prediction_images):
                directory.mkdir()

            (images / "fn.png").write_bytes(b"fn-original")
            (images / "fp.jpg").write_bytes(b"fp-original")
            (images / "correct.png").write_bytes(b"correct")
            (prediction_images / "fn.png").write_bytes(b"fn-preview")
            (prediction_images / "fp.jpg").write_bytes(b"fp-preview")

            (gt_labels / "fn.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (gt_labels / "fp.txt").write_text("", encoding="utf-8")
            (gt_labels / "correct.txt").write_text("", encoding="utf-8")
            (pred_labels / "fp.txt").write_text("0 0.5 0.5 0.1 0.1 0.7\n", encoding="utf-8")

            summary = export_review_package(
                images=images,
                gt_labels=gt_labels,
                pred_labels=pred_labels,
                output_dir=output,
                prediction_images=prediction_images,
            )

            self.assertEqual(1, summary.false_negatives)
            self.assertEqual(1, summary.false_positives)
            self.assertEqual(
                b"fn-original",
                (output / "false_negatives" / "original" / "fn.png").read_bytes(),
            )
            self.assertEqual(
                b"fp-preview",
                (output / "false_positives" / "prediction" / "fp.jpg").read_bytes(),
            )
            self.assertFalse((output / "false_positives" / "original" / "correct.png").exists())

            with (output / "manifest.csv").open(newline="", encoding="utf-8") as manifest:
                rows = list(csv.DictReader(manifest))

            self.assertEqual(["false_negative", "false_positive"], [row["error_type"] for row in rows])
            self.assertEqual(["fn.png", "fp.jpg"], [row["filename"] for row in rows])


if __name__ == "__main__":
    unittest.main()
