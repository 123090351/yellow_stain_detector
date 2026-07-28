import tempfile
import unittest
from pathlib import Path

from scripts.infer.predict_ok_ng import (
    classify_confidences,
    collect_images,
    relative_output_path,
)


class PredictOkNgTest(unittest.TestCase):
    def test_collect_images_recurses_and_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "b.JPG").write_bytes(b"image")
            (root / "nested" / "a.png").write_bytes(b"image")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            images = collect_images(root)

            self.assertEqual(
                [
                    (root / "b.JPG").resolve(),
                    (root / "nested" / "a.png").resolve(),
                ],
                images,
            )

    def test_classifies_any_detection_at_threshold_as_ng(self):
        self.assertEqual(
            ("NG", 2, 0.9),
            classify_confidences([0.2, 0.381, 0.9], threshold=0.381),
        )
        self.assertEqual(
            ("OK", 0, 0.0),
            classify_confidences([0.2, 0.38], threshold=0.381),
        )

    def test_relative_output_path_preserves_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            image = source / "batch" / "image.png"
            image.parent.mkdir()
            image.write_bytes(b"image")

            self.assertEqual(
                Path("batch/image.png"),
                relative_output_path(image, source),
            )


if __name__ == "__main__":
    unittest.main()
