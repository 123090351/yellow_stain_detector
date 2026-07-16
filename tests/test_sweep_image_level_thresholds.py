import tempfile
import unittest
from pathlib import Path

from scripts.eval.sweep_image_level_thresholds import (
    choose_threshold,
    read_max_confidence,
    sweep_thresholds,
)


class SweepImageLevelThresholdsTest(unittest.TestCase):
    def test_reads_highest_confidence_from_prediction_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "sample.txt"
            label.write_text(
                "0 0.5 0.5 0.1 0.1 0.12\n"
                "0 0.4 0.4 0.2 0.2 0.35\n",
                encoding="utf-8",
            )

            self.assertEqual(0.35, read_max_confidence(label))

    def test_sweeps_image_level_metrics_and_selects_target_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root / "gt"
            pred = root / "pred"
            gt.mkdir()
            pred.mkdir()

            (gt / "ng_high.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (gt / "ng_low.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (gt / "ok_high.txt").write_text("", encoding="utf-8")
            (gt / "ok_low.txt").write_text("", encoding="utf-8")

            (pred / "ng_high.txt").write_text("0 0.5 0.5 0.1 0.1 0.90\n", encoding="utf-8")
            (pred / "ng_low.txt").write_text("0 0.5 0.5 0.1 0.1 0.20\n", encoding="utf-8")
            (pred / "ok_high.txt").write_text("0 0.5 0.5 0.1 0.1 0.30\n", encoding="utf-8")
            (pred / "ok_low.txt").write_text("0 0.5 0.5 0.1 0.1 0.05\n", encoding="utf-8")

            results = sweep_thresholds(gt, pred, thresholds=[0.10, 0.25, 0.40])

            self.assertEqual((2, 1, 1, 0), (
                results[0].true_positive,
                results[0].false_positive,
                results[0].true_negative,
                results[0].false_negative,
            ))
            self.assertEqual((1, 1, 1, 1), (
                results[1].true_positive,
                results[1].false_positive,
                results[1].true_negative,
                results[1].false_negative,
            ))
            self.assertEqual((1, 0, 2, 1), (
                results[2].true_positive,
                results[2].false_positive,
                results[2].true_negative,
                results[2].false_negative,
            ))

            selected = choose_threshold(results, target_recall=0.50)
            self.assertEqual(0.40, selected.threshold)


if __name__ == "__main__":
    unittest.main()
