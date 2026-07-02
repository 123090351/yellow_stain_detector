# YOLO Toy Dataset

Small practice dataset for the yellow stain / streak detection workflow.

- Positive images: 13 from `pic/已识别_不含框/`, labeled in makesense.ai.
- Negative images: 13 from `data/ok/`, represented by empty YOLO label files.
- Split: 9 positive + 9 OK train, 2 positive + 2 OK val, 2 positive + 2 OK test.
- Class: `0: huangban`.

This dataset is for learning and pipeline validation only. It is too small for a reliable production metric.

Regenerate it from the project root with:

```bash
python scripts/data/prepare_yolo_toy_dataset.py
```

The generated `images/`, `labels/`, and cache files are local artifacts and are ignored by Git.
