# Yellow Stain Detection - Current Results

**Updated:** 28 July 2026
**Primary objective:** Image-level OK/NG classification with priority on
minimizing missed NG defects.

## Executive Summary

The current candidate is a YOLO11m detector trained on Yellow Stain v3. On the
locked 389-image test split, it detected all 136 NG images and incorrectly
flagged 5 of 253 OK images:

| Accuracy | Precision | NG Recall | OK Accuracy | TP / FP / TN / FN |
|---:|---:|---:|---:|---:|
| **0.987** | **0.965** | **1.000** | **0.980** | **136 / 5 / 248 / 0** |

Three training seeds produced validation accuracy between 0.993 and 1.000,
providing evidence that the selected `imgsz=768, freeze=3` configuration is not
dependent on one lucky random seed.

## Dataset

Yellow Stain v3 contains 3,174 images:

| Split | Total | NG | OK | New OK |
|---|---:|---:|---:|---:|
| Train | 2,512 | 1,596 | 916 | 684 |
| Validation | 273 | 110 | 163 | 113 |
| Test | 389 | 136 | 253 | 204 |
| **Total** | **3,174** | **1,842** | **1,332** | **1,001** |

The v2 split was preserved. Of 1,332 reviewed OK images, 331 stems were already
present in v2 with empty labels and were skipped. The 1,001 unique new OK
images were assigned by continuous capture-hour blocks:

```text
08:00, 09:00, 10:00, 11:00, 13:00 -> train
14:00                                -> validation
15:00, 16:00                         -> test
```

This time-based assignment reduces the risk of adjacent frames from the same
capture period being randomly split across train and evaluation sets.

## Current Candidate

```text
model: YOLO11m, initialized from yolo11m.pt
task: object detection
class: 0 = huangban
imgsz: 768
freeze: 3
epochs: 100 maximum
batch: 16
patience: 15
workers: 12
seed: 21
deterministic: true
device: CUDA device 0
photo-wise confidence threshold: 0.381
LAB-b rescue: disabled
```

Explicitly disabled augmentations:

```text
hsv_h=0.0
hsv_s=0.0
hsv_v=0.0
bgr=0.0
mosaic=0.0
close_mosaic=0
erasing=0.0
```

Other Ultralytics defaults, including possible flip, translation, and scale
augmentation, were not explicitly disabled.

Checkpoint:

```text
runs/detect/v3_baseline_img768_freeze3_seed21/train/
  imgsz768_freeze3_seed21/weights/best.pt
```

The model was trained on `train`, selected using `validation`, and saved as
`best.pt`. Prediction scores were generated at confidence 0.001, then validation
thresholds from 0.001 to 0.500 were swept in steps of 0.001. Threshold 0.381 was
selected under the requirement `NG recall >= 0.975` and locked before test
prediction.

At deployment time:

```text
at least one huangban detection with confidence >= 0.381 -> NG
otherwise                                                -> OK
```

## Image-Level Results

### Seed 21 Validation

Validation contained 273 images: 110 NG and 163 OK.

| Threshold | Accuracy | Precision | NG Recall | OK Accuracy | TP / FP / TN / FN |
|---:|---:|---:|---:|---:|---:|
| **0.381** | **0.993** | **1.000** | **0.982** | **1.000** | **108 / 0 / 163 / 2** |

### Locked Seed 21 Test

The checkpoint and threshold were fixed before running the 389-image test.

| Threshold | Accuracy | Precision | NG Recall | OK Accuracy | TP / FP / TN / FN |
|---:|---:|---:|---:|---:|---:|
| **0.381** | **0.987** | **0.965** | **1.000** | **0.980** | **136 / 5 / 248 / 0** |

All 136 NG images were detected. The five false-positive OK images were:

```text
KN112511001_2026-06-22_15-14-01-557
KN112511001_2026-06-22_15-20-00-452
KN112511001_2026-06-22_15-38-55-847
KN112511001_2026-06-22_15-45-45-083
KN112511001_2026-06-22_15-45-57-164
```

## Seed Stability

The same architecture, data split, and training settings were repeated with
seeds 0, 21, and 42. Each checkpoint received its own validation-selected
threshold because confidence calibration changes between training runs.

| Seed | Threshold | Accuracy | Precision | NG Recall | OK Accuracy | TP / FP / TN / FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 110 / 0 / 163 / 0 |
| 21 | 0.381 | 0.993 | 1.000 | 0.982 | 1.000 | 108 / 0 / 163 / 2 |
| 42 | 0.253 | 1.000 | 1.000 | 1.000 | 1.000 | 110 / 0 / 163 / 0 |

Across the three seeds:

```text
mean validation accuracy: approximately 0.998
mean validation NG recall: approximately 0.994
worst validation accuracy: 0.993
worst validation NG recall: 0.982
maximum FP: 0
maximum FN: 2
```

Seed 42 stopped at epoch 94 after 15 epochs without improvement; its `best.pt`
came from epoch 79. This is normal early-stopping behavior. Seeds 0 and 42 were
used only as validation stability checks. The test split was not used to select
among them, and seed 21 remains the locked candidate.

## 640-Pixel Challenger

A colleague-provided configuration was reproduced on the same v3 train and
validation splits:

```text
model: YOLO11m
imgsz: 640
freeze: 3
epochs: 200 maximum
batch: 16
patience: 30
workers: 16
seed: 141803
validation threshold: 0.203
```

Its image-level validation result was perfect:

| Threshold | Accuracy | Precision | NG Recall | OK Accuracy | TP / FP / TN / FN |
|---:|---:|---:|---:|---:|---:|
| 0.203 | 1.000 | 1.000 | 1.000 | 1.000 | 110 / 0 / 163 / 0 |

Its validation box result was:

| Model | Box P | Box R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 768, seed 42 | **0.937** | 0.803 | **0.873** | 0.687 |
| 640, seed 141803 | 0.919 | **0.811** | 0.869 | **0.688** |

The box differences are negligible and both configurations reached perfect
image-level validation. The 640 model has not been run on the locked test and
is retained only as a lighter challenger for a future independent batch. It
does not replace the tested seed-21 768-pixel candidate.

## Box-Level Results

The v3 seed-21 checkpoint was evaluated on the test split at `conf=0.001` to
calculate the full box precision-recall curve:

| Images | Backgrounds | Instances | Box P | Box R | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 389 | 253 | 157 | **0.876** | **0.892** | **0.926** | **0.721** |

Comparison with the previous v2 freeze-3 checkpoint:

| Model | Box P | Box R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| v2 YOLO11m, freeze 3 | **0.936** | 0.873 | 0.921 | 0.697 |
| **v3 YOLO11m, freeze 3** | 0.876 | **0.892** | **0.926** | **0.721** |

The v3 test adds 204 OK/background images while retaining the same 157 labelled
test instances. This creates more opportunities for false detections and
partly explains the lower box precision. Box recall and both mAP measures
improved.

Image-level NG recall can be 1.000 while box recall is 0.892 because one correct
detection is sufficient to reject an NG image; the detector does not need to
find every labelled stain in that image for the OK/NG business decision.

## Limitations

- The 1,001 additions in v3 are all OK images. NG generalization still depends
  on the existing NG evaluation set.
- Images come from continuous capture periods. Exact duplicate controls and
  time-block splitting reduce leakage risk but do not eliminate near-duplicate
  visual content.
- The historical test data has been viewed during earlier development. The
  current locked result is stronger evidence than validation alone, but it
  should not be presented as final production acceptance.
- Seeds 0 and 42 must not be tested and compared on the current test split to
  select a new winner.
- Final acceptance requires a future untouched production batch containing
  both OK and NG images.

## Current Decision

Use the seed-21 YOLO-only checkpoint with image threshold 0.381 as the current
candidate. Do not add LAB-b rescue or continue tuning against the current test
set. Archive the checkpoint, training arguments, dataset report, threshold, and
SHA-256 checksum, then evaluate once on a future independent OK+NG batch.

## Delivery Artifacts

The handoff consists of separate artifacts with different responsibilities:

### `best.pt`

`best.pt` is the trained PyTorch/Ultralytics checkpoint selected at the best
validation epoch. It contains the learned YOLO11m weights and model metadata.
It does not contain the training images, Python environment, source repository,
or Docker runtime. The current file is approximately 40.5 MB and must be
delivered with a SHA-256 checksum.

### `model.yaml`

`delivery/model.yaml` is a small human-readable configuration and model-card
summary. It records the class name, `imgsz=768`, image-level
`threshold=0.381`, training parameters, environment versions, and locked test
metrics. It does not contain neural-network weights and cannot perform
inference without `best.pt`.

### Inference Docker Image

The inference Docker image is the reproducible Linux/Python/CUDA/Ultralytics
runtime built from `docker/Dockerfile.inference`. It includes
`scripts/infer/predict_ok_ng.py` but intentionally excludes client data and
`best.pt`. At runtime the recipient mounts:

```text
/model/best.pt -> trained checkpoint
/input         -> unlabelled original images
/output        -> decisions.csv, summary.json, and optional annotated images
```

The image therefore packages the software environment, while `best.pt`
packages the learned model and `model.yaml` packages the locked parameters.

### Inference Without Labels

YOLO can directly process original PNG/JPG images without any TXT labels.
Labels are required only for training or for computing evaluation metrics
against known ground truth. During inference:

1. the model reads an original image;
2. it predicts zero or more `huangban` boxes and confidence scores;
3. boxes at or above confidence 0.381 are drawn on the optional annotated copy;
4. any such box makes the image-level decision `NG`;
5. no qualifying box makes the decision `OK`.

The delivered inference script writes one row per input image to
`decisions.csv` and can save images with the predicted yellow-stain boxes. No
same-name label file is needed for this production prediction workflow.
