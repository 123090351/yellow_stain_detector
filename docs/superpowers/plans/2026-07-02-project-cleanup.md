# Project Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the project into a GitHub-ready structure and remove local hardcoded paths.

**Architecture:** Reusable files move into `scripts/`, `docs/`, `configs/`, `models/`, and `external/`. Local data and generated artifacts stay ignored by Git. Python scripts compute the project root from their script location and write portable YOLO configuration.

**Tech Stack:** Python `pathlib`, Ultralytics YOLO CLI conventions, Markdown documentation, `.gitignore`.

---

### Task 1: Create Stable Project Directories

**Files:**
- Create directories: `docs/client/`, `docs/notes/`, `scripts/data/`, `scripts/experiments/`, `scripts/train/`, `configs/`, `models/pretrained/`, `external/`

- [ ] **Step 1: Create directories**

Run:

```bash
mkdir -p docs/client docs/notes scripts/data scripts/experiments scripts/train configs models/pretrained external
```

Expected: directories exist and no output is required.

### Task 2: Move Files Into Their New Locations

**Files:**
- Move client docs to `docs/client/`
- Move notes to `docs/notes/`
- Move scripts to `scripts/data/`, `scripts/experiments/`, and `scripts/train/`
- Move `yolo11n.pt` to `models/pretrained/`
- Move `ultralytics/` to `external/ultralytics/`

- [ ] **Step 1: Move files**

Run non-destructive `mv` commands for existing files only.

Expected: root contains fewer mixed files; ignored local data remains local.

### Task 3: Fix Script Paths

**Files:**
- Modify: `scripts/data/prepare_yolo_toy_dataset.py`
- Modify: `scripts/experiments/threshold_experiment.py`
- Modify: `scripts/train/try_predict.py`
- Modify: `scripts/experiments/enhance_for_labeling.py`
- Modify: `scripts/experiments/enhance_v2.py`
- Modify: `scripts/experiments/reveal_streaks.py`
- Modify: `scripts/data/convert_markdown_to_pdf.py`

- [ ] **Step 1: Update root detection**

Use `Path(__file__).resolve().parents[2]` for scripts two folders below the root.

- [ ] **Step 2: Update generated `data.yaml`**

Write `path: .` for generated dataset configs so they are portable when commands run against the dataset directory, or use the generated dataset path relative to the project root in documentation.

- [ ] **Step 3: Update model and input paths**

Point sample prediction to `models/pretrained/yolo11n.pt` and local image paths through `ROOT`.

### Task 4: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `ai.md`
- Modify: `dataset_yolo_toy/README.md`

- [ ] **Step 1: Update directory map**

Reflect the new `docs/`, `scripts/`, `models/`, and `external/` layout.

- [ ] **Step 2: Update commands**

Use script paths such as `python scripts/data/prepare_yolo_toy_dataset.py` and model paths such as `models/pretrained/yolo11n.pt`.

- [ ] **Step 3: Generalize local Windows paths**

Replace real user paths with `<miniconda-env-path>`.

### Task 5: Update Ignore Rules

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Ignore reorganized artifacts**

Ensure `models/`, `external/ultralytics/`, `data/`, `pic/`, `huangban/`, `OK/`, `annotations/`, `datasets/`, `runs/`, enhanced outputs, archives, and caches stay ignored.

- [ ] **Step 2: Keep reusable files trackable**

Do not ignore `scripts/`, `docs/`, `configs/`, root README files, or `dataset_yolo_toy/README.md`.

### Task 6: Validate Cleanup

**Files:**
- Inspect all modified files.

- [ ] **Step 1: Search local hardcoded paths**

Run:

```bash
rg -n "/Users/[A-Za-z0-9_-]+" README.md ai.md docs scripts configs dataset_yolo_toy/README.md dataset_yolo_toy/data.yaml .gitignore
```

Expected: no functional config path remains.

- [ ] **Step 2: Regenerate toy dataset config**

Run:

```bash
python scripts/data/prepare_yolo_toy_dataset.py
```

Expected: `dataset_yolo_toy/data.yaml` is written with portable path settings.

- [ ] **Step 3: Check ignored files**

If a Git repository exists, run:

```bash
git status --short
```

Expected: customer data, weights, external source, and generated outputs are not listed for commit.

If no Git repository exists, run:

```bash
find . -maxdepth 2 -type f | sort
```

Expected: reusable files are visibly separated from local ignored data directories.
