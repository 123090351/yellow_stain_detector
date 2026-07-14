# HKPC Yellow Stain Detection / HKPC 黄斑检测项目

This repository is for developing a computer vision model to detect yellow stains or yellow streaks on white foam bra cups.

本项目用于开发海绵胸杯表面「黄斑 / 黄条」缺陷视觉检测模型。

Current practical route:

当前实用路线：

```text
Collect images -> annotate yellow stain boxes -> build YOLO dataset -> train YOLO detection -> evaluate -> review failures -> add data and retrain
收集图片 -> 标注黄斑矩形框 -> 生成 YOLO 数据集 -> 训练 YOLO 检测模型 -> 评估 -> 分析错例 -> 补数据重训
```

The business goal is image-level `OK / NG` classification, but the current implementation uses YOLO detection so that the model can also show the defect location.

业务目标是整图 `OK / NG` 分类，但当前用 YOLO detection 实现，因为它可以同时给出黄斑位置。

## Table of Contents / 目录

1. [Project Background / 项目背景](#1-project-background--项目背景)
2. [Goal / 项目目标](#2-goal--项目目标)
3. [Current Status / 当前状态](#3-current-status--当前状态)
4. [Document Map / 文档索引](#4-document-map--文档索引)
5. [Current Directory Layout / 当前目录结构](#5-current-directory-layout--当前目录结构)
6. [Directory Guide / 目录说明](#6-directory-guide--目录说明)
7. [Recommended Cleanup Direction / 后续整理方向](#7-recommended-cleanup-direction--后续整理方向)
8. [Annotation Standard / 标注规范](#8-annotation-standard--标注规范)
9. [makesense.ai Annotation / makesense.ai 标注](#9-makesenseai-annotation--makesenseai-标注)
10. [Export YOLO Labels / 导出 YOLO 标签](#10-export-yolo-labels--导出-yolo-标签)
11. [YOLO Dataset Format / YOLO 数据格式](#11-yolo-dataset-format--yolo-数据格式)
12. [Training / 训练](#12-training--训练)
13. [GPU Platform and Docker Workflow / GPU 平台与 Docker 工作流](#13-gpu-platform-and-docker-workflow--gpu-平台与-docker-工作流)
14. [Evaluation / 评估指标](#14-evaluation--评估指标)
15. [Open Items / 待推进事项](#15-open-items--待推进事项)
16. [Delivery and Risks / 交付与风险](#16-delivery-and-risks--交付与风险)

## 1. Project Background / 项目背景

The client already has a black-dot defect detection system based on YOLO, with an existing annotation and batch inference platform. This project adds a new capability: detecting yellow stains or yellow streaks.

需求方已有基于 YOLO 的黑点缺陷检测系统，并配套标注与批量推理平台。本项目是在该系统基础上新增「黄斑 / 黄条」识别能力。

Yellow stains are not obvious black dots. They are weak yellowish streaks on white foam. First-stage image analysis found:

黄斑不是明显黑点，而是白色海绵上的偏黄色条状 / 线状污渍。第一阶段图像分析结论：

- Type A stains are more visible and can be enhanced by LAB `b` channel or similar methods. / A 型黄条较明显，可通过 LAB `b` 通道等方法辅助观察。
- Type B stains are very weak and require quality-inspection experts to provide example annotations. / B 型黄条极弱，需要质检专家给出示范标注。
- Traditional color thresholding is useful for analysis but is not stable enough as the main solution. / 传统颜色阈值法只能辅助分析，不适合作为主方案。

## 2. Goal / 项目目标

The practical target is:

当前实用目标：

- Classify each image as `OK` or `NG`. / 判断每张图是 `OK` 合格品还是 `NG` 不合格品。
- If the image is `NG`, show the approximate yellow-stain location. / 如果是 `NG`，给出黄斑大致位置，方便人工复核。
- Build a working end-to-end baseline first, then improve accuracy by adding data and refining training. / 先打通端到端 baseline，再通过补数据和调训练提升效果。

Current model logic:

当前模型逻辑：

```text
If YOLO detects a huangban box above the confidence threshold -> NG
If YOLO detects no huangban box -> OK

如果 YOLO 检出置信度超过阈值的 huangban 框 -> NG
如果 YOLO 没有检出 huangban 框 -> OK
```

Current class list:

当前类别：

```text
0: huangban
```

## 3. Current Status / 当前状态

As of 2026-07-13, GitHub contains reusable code, documentation, dataset preparation scripts, tests, and small YOLO config/report files only. Local client images, training image data, model weights, training runs, and third-party source checkouts are ignored by Git and should be shared separately through a drive, shared folder, or the GPU platform upload workflow.

截至 2026-07-13，GitHub 只保留可复用代码、文档、数据集准备脚本、测试和小型 YOLO 配置/报告文件。客户图片、训练图片数据、模型权重、训练输出和第三方源码副本默认被 Git 忽略，应通过网盘、共享目录或 GPU 平台上传流程单独交付。

Repository contents on GitHub:

GitHub 当前包含：

- `README.md`, `.gitignore`, and client/project documents under `docs/`. / `README.md`、`.gitignore` 和 `docs/` 下的项目文档。
- YOLO helper scripts under `scripts/`, especially `scripts/data/prepare_yellow_stain_dataset.py` and `scripts/train/try_predict.py`. / `scripts/` 下的 YOLO 辅助脚本，重点是正式数据集生成脚本和推理测试脚本。
- `datasets/yellow_stain_v1/data.yaml` and `datasets/yellow_stain_v1/dataset_report.txt`; generated images and labels are local artifacts and are not committed. / `datasets/yellow_stain_v1/data.yaml` 和 `datasets/yellow_stain_v1/dataset_report.txt`；生成的图片和标签是本地产物，不提交 Git。

Current cleaned dataset summary:

当前清洗后数据概况：

- `training_data/training_data/images/`: `598` cleaned images. / `598` 张清洗后的图片。
- `training_data/training_data/labels/`: `598` same-stem YOLO labels. / `598` 个同名 YOLO 标签。
- Positive / NG nonblank labels: `267`. / 非空标签正样本 `267` 个。
- Negative / OK blank labels: `331`. / 空标签 OK 负样本 `331` 个。
- Formal YOLO dataset: `datasets/yellow_stain_v1/`. / 正式 YOLO 数据集：`datasets/yellow_stain_v1/`。
- Split: train `419` (`187` NG + `232` OK), val `90` (`40` NG + `50` OK), test `89` (`40` NG + `49` OK). / 划分：train `419`（`187` NG + `232` OK），val `90`（`40` NG + `50` OK），test `89`（`40` NG + `49` OK）。

The toy YOLO dataset has been removed from the active workflow. The current next step is to run the first GPU baseline on `datasets/yellow_stain_v1/data.yaml`.

toy YOLO 数据集已从当前工作流中移除。当前下一步是在 GPU 上基于 `datasets/yellow_stain_v1/data.yaml` 跑第一版 baseline。

New batch added on 2026-07-02:

2026-07-02 已新增一批数据：

- `data/ok/batch_2026-07-02_ok_1_2/`: `331` OK images. / `331` 张 OK 图。
- `data/defect/batch_2026-07-02_huangban/compound_original/`: `158` compound yellow-stain originals. / `158` 张复合黄斑原图。
- `data/defect/batch_2026-07-02_huangban/single_original/`: `61` single yellow-stain originals. / `61` 张单一黄斑原图。
- `data/defect/batch_2026-07-02_huangban/missed_visible_original/`: `20` machine-missed but human-visible yellow-stain originals. / `20` 张机器漏检但人眼可见黄斑原图。
- `data/defect/batch_2026-07-02_huangban/missed_invisible_original/`: `148` machine-missed and hard-to-see originals. Keep for analysis; do not use for YOLO detection training until boxes are confirmed. / `148` 张机器漏检且难以目视确认的原图，先保留分析，确认框之前不要直接进 YOLO detection 训练。
- `data/reference_with_boxes/batch_2026-07-02_huangban/`: `219` boxed reference images. These are visual references only, not training images. / `219` 张带框参考图，只作参考，不能当训练图。

Current annotation target:

当前标注目标：

- Train as one YOLO class only: `0: huangban`. Do not split compound/single/missed into separate model classes yet. / 训练时只用一个类别 `0: huangban`，暂不把复合/单一/漏检拆成多个模型类别。
- First annotation batch: `compound_original` + `single_original` + `missed_visible_original` = `239` images. / 第一轮新增标注目标为 `239` 张。
- `missed_invisible_original` has `148` images; keep them for analysis and expert review before training. / `missed_invisible_original` 共 `148` 张，先保留分析，等专家确认位置后再进入训练。
- Existing machine-readable labels: `13` YOLO `.txt` files. / 当前已有机器可读 YOLO 标注 `13` 个。
- Estimated first-round manual labeling work: about `273` images if the older unlabelled defect set is included. / 如果把旧的未标缺陷图也纳入第一轮，预计还需手动标注约 `273` 张。

Current technical decisions:

当前技术决策：

- Use YOLO detection as the first baseline. / 第一版 baseline 使用 YOLO detection。
- Use one class only: `huangban`. / 只标一个类别：`huangban`。
- Use horizontal rectangles for now. / 当前只用水平矩形框。
- Do not start with rotated boxes, polygons, or segmentation masks. / 暂不使用旋转框、多边形或分割 mask。
- Annotate original images and train on original images. / 标原图，训练也先用原图。
- Enhanced images are only visual aids unless the whole pipeline is changed to use the same preprocessing. / 增强图只作辅助观察；除非全流程都统一预处理，否则不直接用增强图训练。

## 4. Document Map / 文档索引

| Path / 路径 | Purpose / 用途 |
|---|---|
| `README.md` | Main project entry for all team members. / 项目总入口，给所有协作者看。 |
| `docs/client/yellow_stain_detection_plan_client.md` | Chinese client-facing project plan. / 中文客户计划书。 |
| `docs/client/yellow_stain_detection_plan_client.pdf` | PDF version of the Chinese client plan. / 中文客户计划书 PDF。 |
| `docs/client/yellow_stain_detection_plan_client.html` | HTML version of the Chinese client plan. / 中文客户计划书 HTML。 |
| `docs/client/yellow_stain_detection_plan_client_EN.md` | English client-facing project plan. / 英文客户计划书。 |
| `docs/client/yellow_stain_detection_plan_client_EN.pdf` | PDF version of the English client plan. / 英文客户计划书 PDF。 |
| `docs/client/yellow_stain_detection_plan_client_EN.html` | HTML version of the English client plan. / 英文客户计划书 HTML。 |
| `docs/notes/HANDOFF_黄条显现.md` | Image enhancement and yellow-streak visibility notes. / 黄条显现和图像增强交接说明。 |
| `datasets/yellow_stain_v1/dataset_report.txt` | Generated split summary for the formal YOLO dataset. / 正式 YOLO 数据集切分报告。 |

## 5. Current Directory Layout / 当前目录结构

This is the current working layout after cleanup. Local data and generated artifacts are ignored by Git.

这是整理后的当前目录结构。客户数据和生成产物默认被 Git 忽略。

```text
.
├── README.md
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
├── training_data/
│   └── training_data/
│       ├── images/
│       └── labels/
├── datasets/
│   └── yellow_stain_v1/
│       ├── images/
│       ├── labels/
│       ├── data.yaml
│       └── dataset_report.txt
└── runs/
    └── detect/
```

## 6. Directory Guide / 目录说明

| Path / 路径 | Meaning / 说明 |
|---|---|
| `training_data/training_data/` | Cleaned source images and same-stem YOLO labels. Upload separately; do not commit to GitHub. / 清洗后的源图片和同名 YOLO 标签。单独上传，不提交 GitHub。 |
| `datasets/yellow_stain_v1/` | Generated formal YOLO dataset. `data.yaml` and report are tracked; images and labels are local artifacts. / 生成的正式 YOLO 数据集。`data.yaml` 和报告被跟踪；图片和标签是本地产物。 |
| `runs/detect/` | YOLO training and prediction outputs. / YOLO 训练和预测结果。 |
| `scripts/data/` | Dataset preparation scripts. / 数据集整理脚本。 |
| `scripts/experiments/` | Image enhancement and analysis experiments. / 图像增强和分析实验脚本。 |
| `scripts/train/` | Training and prediction helper scripts. / 训练和推理辅助脚本。 |
| `models/pretrained/` | Local pretrained weights, ignored by Git. / 本地预训练权重，Git 默认忽略。 |
| `external/ultralytics/` | Local Ultralytics source/code copy. Usually do not edit. / 本地 Ultralytics 代码，一般不改。 |

Ultralytics YOLO is not vendored in this repository. Install it separately:

本仓库不内置 Ultralytics YOLO 框架源码，使用时单独安装：

```bash
pip install ultralytics
```

Official repository:

官方仓库：

```text
https://github.com/ultralytics/ultralytics
```

## Generate LAB_b Enhancement Images / 生成 LAB_b 增强图

Use LAB_b enhancement as the main labeling aid:

用 LAB_b 增强图作为主要标注辅助：

```bash
python scripts/experiments/enhance_for_labeling.py data/defect
```

Output:

输出：

```text
enhanced/LAB_b/
```

The script reads image folders recursively and keeps the relative folder structure. Annotate by looking at the enhanced image, but train YOLO with the original image and the corresponding YOLO `.txt` label.

脚本会递归读取子目录，并保留相对目录结构。标注时可以看增强图，但训练 YOLO 时仍使用原图和对应的 YOLO `.txt` 标签。

## 7. Repository Cleanup Rules / 仓库整理规则

The project is organized so GitHub keeps reusable code and documentation while local data stays on this machine.

当前项目已按“代码文档可上传、客户数据本地保留”的原则整理。

Current repository-oriented layout:

当前面向 GitHub 的整理结构：

```text
.
├── README.md
├── docs/
│   ├── client/
│   └── notes/
├── data/
├── annotations/
├── datasets/
├── scripts/
│   ├── data/
│   ├── train/
│   └── experiments/
├── models/
│   ├── pretrained/
│   └── exported/
├── runs/
└── external/
    └── ultralytics/
```

Cleanup rules:

整理规则：

- Keep raw images immutable. Do not overwrite original data. / 原始图片不覆盖、不改名，尽量保持可追溯。
- Put generated datasets under `datasets/`. / 生成的数据集放到 `datasets/`。
- Put one-off experiment outputs under `runs/` or `outputs/`. / 实验输出放到 `runs/` 或 `outputs/`。
- Put reusable scripts under `scripts/`. / 可复用脚本放到 `scripts/`。
- Put client-facing documents under `docs/client/`. / 对外客户文档放到 `docs/client/`。
- Put third-party source code under `external/`. / 第三方源码放到 `external/`。

GitHub rules:

GitHub 上传规则：

- Do not commit local absolute paths such as `/Users/name/Desktop/project` or `C:\Users\name\project`. Use `<project-root>` in documentation. / 不要提交本机绝对路径，例如 `/Users/name/Desktop/project` 或 `C:\Users\name\project`。文档里统一写 `<project-root>`。
- Do not commit client images, raw datasets, training runs, model weights, or cache files unless the team explicitly agrees. / 未经团队确认，不要提交客户图片、原始数据集、训练输出、模型权重或缓存文件。
- Do not commit local AI handoff notes such as `ai.md` or `CLAUDE.md`. / 不提交本地 AI 交接上下文，例如 `ai.md` 或 `CLAUDE.md`。
- Keep reusable code, documentation, dataset-conversion scripts, and small config files in Git. / Git 里保留可复用代码、文档、数据集转换脚本和小型配置文件。
- Use `.gitignore` to block large or sensitive local artifacts. / 用 `.gitignore` 阻止大文件和敏感本地产物被误传。

## 8. Annotation Standard / 标注规范

Class name:

类别名：

```text
huangban
```

Rules:

规则：

- Annotate original images only. / 只标原图。
- Use rectangle boxes only for now. / 当前只用矩形框。
- Do not use polygon, line, point, rotated box, or segmentation mask yet. / 暂不使用多边形、线、点、旋转框或分割 mask。
- If the stain is diagonal, use a horizontal rectangle that covers the full stain. / 如果黄斑是斜线，用水平矩形把整条黄斑包住。
- If one image has multiple stains, draw multiple boxes, all labeled `huangban`. / 一张图有多个黄斑就画多个框，每个框都标 `huangban`。
- Do not box the whole cup. Box only the yellow-stain area. / 不要框整个杯子，只框黄斑 / 黄条区域。
- OK images should have no boxes. / OK 图不画框。
- Use exactly `huangban`; do not mix names like `yellow_stain`, `黄斑`, or `ng`. / 标签名必须统一为 `huangban`，不要混用其他名字。

Recommended annotation order:

推荐标注顺序：

1. `pic/已识别_不含框/`: first batch, clear images.
2. `huangban/`: second batch, earlier raw defect images.
3. `pic/huangban-未识别/`: hard cases; annotate after expert confirmation.
4. `data/ok/`: no boxes; use as negative samples.

## 9. makesense.ai Annotation / makesense.ai 标注

Current annotation tool:

当前标注工具：

```text
makesense.ai
```

Labelme is not part of the current workflow. Do not create new Labelme JSON annotations unless the team explicitly changes the workflow.

当前流程不使用 Labelme。除非团队重新改流程，否则不要再新增 Labelme JSON 标注。

Recommended workflow:

推荐流程：

1. Generate LAB_b enhancement images. / 先生成 LAB_b 增强图。
2. Upload the LAB_b images to makesense.ai for easier visual labeling. / 把 LAB_b 增强图上传到 makesense.ai，方便肉眼看黄斑位置。
3. Draw rectangle boxes around yellow stains. / 用矩形框标黄斑。
4. Use exactly one label name: `huangban`. / 标签名固定为 `huangban`。
5. Export labels in YOLO format. / 导出 YOLO 格式标签。
6. Train YOLO using original images plus the exported `.txt` labels. / 训练时使用原图 + 导出的 `.txt` 标签。

Important:

注意：

- The enhanced image and original image must have the same pixel size. / 增强图和原图必须像素尺寸一致。
- Keep the same filename stem between enhanced image, original image, and label file. / 增强图、原图、标签文件的文件名主体必须一致。
- Boxed reference images are visual references only; do not use them as training images. / 带框参考图只作参考，不能当训练图。

## 10. Export YOLO Labels / 导出 YOLO 标签

When exporting from makesense.ai, choose YOLO format. Each image should get a same-stem `.txt` label file:

从 makesense.ai 导出时选择 YOLO 格式。每张图对应一个同名 `.txt` 标签文件：

```text
image_name.jpg
image_name.txt
```

YOLO label line format:

YOLO 标签每一行格式：

```text
0 x_center y_center width height
```

For this project:

本项目中：

```text
0 = huangban
```

OK images should have empty `.txt` label files when building a YOLO detection dataset.

构建 YOLO detection 数据集时，OK 图应对应空的 `.txt` 标签文件。

## 11. YOLO Dataset Format / YOLO 数据格式

Standard YOLO detection layout:

YOLO detection 标准格式：

```text
dataset/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  data.yaml
```

Each image needs a label file with the same base name:

每张图片需要一个同名标签文件：

```text
0 x_center y_center width height
```

Notes:

说明：

- `0` is the class id for `huangban`. / `0` 是 `huangban` 的类别 id。
- Coordinates are normalized to `[0, 1]`, not pixel coordinates. / 坐标是 0 到 1 的归一化值，不是像素值。
- Multiple boxes mean multiple lines. / 多个框就写多行。
- OK images should have empty `.txt` files. / OK 图对应空 `.txt` 文件。

Common split:

常见划分：

```text
train / val / test = 70 / 15 / 15
```

With the current small dataset, metrics are only for learning and debugging, not formal acceptance.

当前数据很少，指标只能用于学习和调试，不能作为正式验收结论。

## 12. Training / 训练

Prepare the formal dataset from cleaned source data:

从清洗后的源数据生成正式数据集：

```bash
python scripts/data/prepare_yellow_stain_dataset.py
```

Current formal dataset config:

当前正式数据集配置：

```text
datasets/yellow_stain_v1/data.yaml
```

First GPU baseline:

第一版 GPU baseline：

```bash
yolo detect train \
  model=yolo11n.pt \
  data=datasets/yellow_stain_v1/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/detect \
  name=yellow_stain_v1_yolo11n_640
```

If Ultralytics resolves `path: .` from the repository root on the GPU container and reports a missing path such as `<repo>/images/val`, create a container-local YAML with an absolute dataset root:

如果 GPU container 中 Ultralytics 把 `path: .` 解析到仓库根目录，并报缺少 `<repo>/images/val`，在容器内创建一个使用绝对数据集根目录的本地 YAML：

```bash
cat > datasets/yellow_stain_v1/data_server.yaml <<'EOF'
path: /data/greya/yellow_stain_detector/datasets/yellow_stain_v1
train: images/train
val: images/val
test: images/test

names:
  0: huangban
EOF
```

Then train with the server YAML:

然后使用服务器 YAML 训练：

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/data/greya/yellow_stain_detector/datasets/yellow_stain_v1/data_server.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=4 \
  project=/data/greya/yellow_stain_detector/runs/detect \
  name=yellow_stain_v1_yolo11n_640
```

Current first-run result: the `yolo11n` 640 and 1280 baselines run successfully, but recall is still low. The next practical experiment is a larger model at 1280, then review prediction images before further tuning.

当前第一轮结果：`yolo11n` 的 640 和 1280 baseline 都已跑通，但 recall 仍偏低。下一步更实际的是用 1280 跑更大的模型，然后先看预测图再继续调参。

```bash
yolo detect train \
  model=yolo11m.pt \
  data=/data/greya/yellow_stain_detector/datasets/yellow_stain_v1/data_server.yaml \
  epochs=100 \
  imgsz=1280 \
  batch=8 \
  device=0 \
  workers=4 \
  project=/data/greya/yellow_stain_detector/runs/detect \
  name=yellow_stain_v1_yolo11m_1280
```

## 13. GPU Platform and Docker Workflow / GPU 平台与 Docker 工作流

The company AI/HPC platform runs containers from Docker images. The recommended workflow is to build a dedicated NVIDIA CUDA/PyTorch/YOLO environment image, push or upload it to the company Container Registry, then clone the GitHub source code inside the running GPU container.

公司 AI/HPC 平台从 Docker image 启动容器。当前推荐流程是先构建一个专用 NVIDIA CUDA/PyTorch/YOLO 环境镜像，推送或上传到公司 Container Registry，然后在运行中的 GPU 容器里 clone GitHub 源码。

The image is an environment image only. Do not bake client images, generated datasets, training outputs, or private notes into it.

该镜像只作为环境镜像使用。不要把客户图片、生成数据集、训练输出或私有笔记打进镜像。

Build the environment image for the GPU platform:

构建 GPU 平台使用的环境镜像：

```bash
docker buildx create --use --name yellow-builder
docker buildx build --platform linux/amd64 -t yellow-stain-yolo:0.1 --load .
```

Push the image to the company Container Registry using the exact registry path and push command shown by the platform.

使用公司平台显示的准确 registry 路径和 push command，将镜像推送到公司 Container Registry。

GPU container workflow:

GPU 容器流程：

```text
create one GPU container with yellow-stain-yolo:0.1
  -> open Web Terminal / Jupyter terminal
  -> git clone this repository
  -> upload datasets/yellow_stain_v1/ or upload training_data/ and regenerate the dataset
  -> run yolo detect train
```

For the HKPC interactive container tested on 2026-07-14, the cloned repository path was:

2026-07-14 测试的 HKPC 交互式容器中，仓库路径为：

```text
/workspace/yellow_stain_detector
```

The validated data-transfer route was:

已验证的数据传输路线：

```text
package datasets/yellow_stain_v1 as yellow_stain_v1_data.zip
  -> upload to a private Hugging Face dataset repository
  -> download in the GPU container
  -> unzip from the cloned repository root
```

Download example:

下载示例：

```bash
cd /workspace/yellow_stain_detector
export HF_TOKEN='<hugging-face-read-token>'
curl -L \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/datasets/greyaliu/yellow-stain-v1-data/resolve/main/yellow_stain_v1_data.zip" \
  -o yellow_stain_v1_data.zip
unzip -t yellow_stain_v1_data.zip
unzip yellow_stain_v1_data.zip
find datasets/yellow_stain_v1/images -type f | wc -l
find datasets/yellow_stain_v1/labels -type f | wc -l
```

Expected counts:

预期数量：

```text
598 images
598 labels
```

Suggested platform settings:

建议平台参数：

```text
container image: yellow-stain-yolo:0.1
image pull policy: IfNotPresent
privileged container: false
command: sleep
args: 99999999
GPU physical cards: 1
GPU compute: 100%
GPU memory: 80Gi, or the maximum available
instances / parallelism / successful pods: 1
restart policy for first interactive run: do not restart
retry count for first interactive run: 0 or 1
```

Verify CUDA inside the running container before training:

训练前先在容器内验证 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Do not confuse GitHub source code with a Docker image. A normal GitHub repository stores source code. A Docker image must be pushed to a Docker Registry such as the company Registry, Docker Hub, Quay.io, or GitHub Container Registry.

不要把 GitHub 源码仓库和 Docker image 混淆。普通 GitHub 仓库存源码；Docker image 需要推到 Docker Registry，例如公司 Registry、Docker Hub、Quay.io 或 GitHub Container Registry。

## 14. Evaluation / 评估指标

Key metrics:

关键指标：

- `precision`: how many predicted defects are real. Low precision means many false positives. / 模型报黄斑时，有多少是真的。低 precision 表示误检多。
- `recall`: how many real defects are detected. Low recall means many missed defects. / 真实黄斑里，有多少被找到了。低 recall 表示漏检多。
- `IoU`: overlap between predicted box and ground-truth box. / 预测框和人工框的重合程度。
- `mAP50`: detection score at IoU >= 0.5. / IoU >= 0.5 时的检测综合指标。
- `mAP50-95`: stricter mAP across multiple IoU thresholds. / 更严格的综合指标。
- `conf`: confidence threshold. Lower `conf` usually reduces missed defects but increases false positives. / 置信度阈值。降低 `conf` 通常减少漏检但增加误检。

Important result files:

重点查看文件：

```text
runs/detect/.../results.csv
runs/detect/.../results.png
runs/detect/.../val_batch0_labels.jpg
runs/detect/.../val_batch0_pred.jpg
runs/detect/.../confusion_matrix.png
```

`val_batch0_labels.jpg` shows human labels. `val_batch0_pred.jpg` shows model predictions. Compare them to find missed detections, false positives, and bad box placement.

`val_batch0_labels.jpg` 是人工标注，`val_batch0_pred.jpg` 是模型预测。对比两者可以看漏检、误检和框偏移。

## 15. Open Items / 待推进事项

Priority list:

优先级清单：

1. Get more confirmed defect images, especially weak and borderline cases. / 补充更多已确认黄斑图，尤其是弱信号和边界案例。
2. Get more OK images. / 补充更多 OK 合格品图。
3. Ask experts to annotate Type B weak yellow streaks. / 请质检专家示范标注 B 型弱黄条。
4. Confirm whether the client platform can export machine-readable labels. / 确认对方平台能否导出机器可读标签。
5. Confirm whether the platform can train and export `.evo`. / 确认平台是否能训练并导出 `.evo`。
6. Confirm acceptance test set, metrics, and false-negative / false-positive priority. / 确认验收测试集、指标以及漏检 / 误检优先级。
7. If the platform does not train the model, prepare cloud GPU or client GPU server. / 如果平台不负责训练，准备云 GPU 或需求方 GPU 服务器。

## 16. Delivery and Risks / 交付与风险

The client-facing plan currently treats `.evo` as the core delivery format, but the `.evo` generation path is not confirmed yet.

对外客户计划书目前把 `.evo` 模型文件作为核心交付，但 `.evo` 生成链路尚未确认。

If `.evo` cannot be generated directly, the team must confirm an alternative delivery format, such as YOLO weights, ONNX, training code, annotated dataset, and technical documentation.

如果无法直接生成 `.evo`，需要另行确认替代交付格式，例如 YOLO 权重、ONNX、训练代码、标注数据集和技术说明。

Main risks:

主要风险：

- Type B yellow streaks are too weak for stable annotation. / B 型黄条太弱，人工也难以稳定标注。
- The dataset is still too small for reliable metrics. / 样本量仍太小，指标不稳定。
- Inconsistent annotation rules will directly hurt model quality. / 标注口径不一致会直接影响模型效果。
- The `.evo` export path is unclear. / `.evo` 导出链路尚不明确。
- Local CPU is enough for learning and small experiments, but formal training should use an NVIDIA GPU. / 本机 CPU 适合学习和小实验，正式训练建议使用 NVIDIA GPU。
