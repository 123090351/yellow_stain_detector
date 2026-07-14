import tempfile
import unittest
from pathlib import Path

from scripts.eval.evaluate_image_level_ok_ng import evaluate_image_level


class EvaluateImageLevelOkNgTest(unittest.TestCase):
    def test_evaluates_image_level_ok_ng_from_label_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root / "gt"
            pred = root / "pred"
            gt.mkdir()
            pred.mkdir()

            (gt / "tp.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (pred / "tp.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n", encoding="utf-8")

            (gt / "fp.txt").write_text("", encoding="utf-8")
            (pred / "fp.txt").write_text("0 0.5 0.5 0.1 0.1 0.7\n", encoding="utf-8")

            (gt / "tn.txt").write_text("", encoding="utf-8")

            (gt / "fn.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

            metrics = evaluate_image_level(gt, pred)

            self.assertEqual(4, metrics.total)
            self.assertEqual(2, metrics.prediction_files_missing)
            self.assertEqual(1, metrics.true_positive)
            self.assertEqual(1, metrics.false_positive)
            self.assertEqual(1, metrics.true_negative)
            self.assertEqual(1, metrics.false_negative)
            self.assertEqual(0.5, metrics.accuracy)
            self.assertEqual(0.5, metrics.precision)
            self.assertEqual(0.5, metrics.recall)
            self.assertEqual(["fn"], metrics.false_negative_stems)
            self.assertEqual(["fp"], metrics.false_positive_stems)


if __name__ == "__main__":
    unittest.main()
