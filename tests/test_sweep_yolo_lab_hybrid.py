import unittest

import numpy as np

from scripts.eval.sweep_yolo_lab_hybrid import (
    ImageSample,
    PredictionBox,
    ScoredBox,
    choose_result,
    lab_yellow_delta,
    sweep_hybrid,
)


class SweepYoloLabHybridTest(unittest.TestCase):
    def test_lab_delta_measures_yellow_excess_inside_box(self):
        lab_b = np.full((10, 10), 128, dtype=np.float32)
        lab_b[3:7, 3:7] = 158
        box = PredictionBox(
            x_center=0.5,
            y_center=0.5,
            width=0.4,
            height=0.4,
            confidence=0.05,
        )

        self.assertEqual(30.0, lab_yellow_delta(lab_b, box, percentile=90.0))

    def test_selects_lab_rescue_that_meets_target_recall_with_fewer_false_positives(self):
        samples = [
            ImageSample("ng_direct", True, (ScoredBox(0.20, 0.0),)),
            ImageSample("ng_rescue", True, (ScoredBox(0.05, 10.0),)),
            ImageSample("ok_neutral", False, (ScoredBox(0.04, 2.0),)),
            ImageSample("ok_yellow", False, (ScoredBox(0.03, 9.0),)),
        ]

        results = sweep_hybrid(
            samples,
            direct_confidence=0.10,
            rescue_confidences=[0.02],
            lab_thresholds=[5.0, 11.0],
        )
        selected = choose_result(results, target_recall=1.0)

        self.assertEqual(5.0, selected.lab_threshold)
        self.assertEqual((2, 1, 1, 0), (
            selected.true_positive,
            selected.false_positive,
            selected.true_negative,
            selected.false_negative,
        ))


if __name__ == "__main__":
    unittest.main()
