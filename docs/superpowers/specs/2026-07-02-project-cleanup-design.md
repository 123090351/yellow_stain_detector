# Project Cleanup Design

## Goal

Prepare the yellow-stain detection project for GitHub by removing local path assumptions, separating reusable project files from local data artifacts, and documenting a stable directory layout.

## Scope

This cleanup reorganizes project-owned scripts, documentation, model weights, and third-party source code. It keeps customer images, annotations, generated datasets, enhanced images, and training runs local and ignored by Git.

## Directory Layout

The cleaned layout should use:

```text
.
├── README.md
├── ai.md
├── configs/
├── docs/
│   ├── client/
│   ├── notes/
│   └── superpowers/
├── scripts/
│   ├── data/
│   ├── experiments/
│   └── train/
├── models/
│   └── pretrained/
├── external/
│   └── ultralytics/
├── data/
├── annotations/
├── datasets/
├── runs/
├── enhanced/
├── enhanced_v2/
└── reveal_out/
```

## Path Rules

Python scripts should derive the project root from their own location instead of assuming the current working directory. Generated YOLO `data.yaml` files should avoid machine-specific absolute paths when possible. Documentation should use `<project-root>` and placeholder paths instead of real local user paths.

## GitHub Rules

The repository should commit reusable code, project documentation, lightweight configs, and README files. It should ignore customer images, local annotations, generated datasets, model weights, training outputs, enhanced outputs, archives, caches, and third-party checkouts.

## Implementation Notes

The existing local data directories remain in place unless moving them is necessary for clarity. The important GitHub boundary is `.gitignore`: local data should not be staged. The `dataset_yolo_toy` folder may keep its README and config, but generated images, labels, and caches remain ignored.

## Validation

After the cleanup:

- Searching for real local user paths such as `/Users/<name>` should return no functional config paths.
- Scripts should still be readable and point to the new layout.
- `prepare_yolo_toy_dataset.py` should be runnable from the project root and write a portable `data.yaml`.
- A dry-run Git listing should show only reusable files, not images, weights, runs, or third-party source.
