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
13. [Evaluation / 评估指标](#13-evaluation--评估指标)
14. [Open Items / 待推进事项](#14-open-items--待推进事项)
15. [Delivery and Risks / 交付与风险](#15-delivery-and-risks--交付与风险)

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

As of 2026-06-30:

截至 2026-06-30：

- `47` yellow-stain / yellow-streak defect images have been organized. / 已整理 `47` 张黄斑 / 黄条缺陷图像。
- `22` OK images have been organized. / 已整理 `22` 张 OK 合格品图像。
- `13` reference images with boxes are available, but they are visual references only. / 已整理 `13` 张带框参考图，但只能参考，不能直接作为训练标签。
- `13` clearer yellow-stain images were annotated with makesense.ai and exported as YOLO `.txt`. / 已用 makesense.ai 标注 `13` 张较清晰黄斑图，并导出 YOLO `.txt`。
- A toy YOLO dataset has been generated under `dataset_yolo_toy/`. / 已生成小型 YOLO toy 数据集 `dataset_yolo_toy/`。
- YOLO training has run successfully for a 3-epoch smoke test and a 50-epoch toy run. / 已跑通 3 epoch smoke test 和 50 epoch toy training。
- The toy model is not usable yet. The pipeline works, but data volume and label quality are the bottlenecks. / toy 模型效果还不能用；训练链路已通，瓶颈是数据量和标注质量。

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
| `ai.md` | AI assistant handoff context with latest decisions and experiment notes. / AI 助手上下文，记录最新决策和实验结果。 |
| `CLAUDE.md` | Older AI context, kept for history. `ai.md` is preferred. / 旧 AI 上下文，保留作历史参考，优先看 `ai.md`。 |
| `docs/client/yellow_stain_detection_plan_client.md` | Chinese client-facing project plan. / 中文客户计划书。 |
| `docs/client/yellow_stain_detection_plan_client.pdf` | PDF version of the Chinese client plan. / 中文客户计划书 PDF。 |
| `docs/client/yellow_stain_detection_plan_client.html` | HTML version of the Chinese client plan. / 中文客户计划书 HTML。 |
| `docs/client/yellow_stain_detection_plan_client_EN.md` | English client-facing project plan. / 英文客户计划书。 |
| `docs/client/yellow_stain_detection_plan_client_EN.pdf` | PDF version of the English client plan. / 英文客户计划书 PDF。 |
| `docs/client/yellow_stain_detection_plan_client_EN.html` | HTML version of the English client plan. / 英文客户计划书 HTML。 |
| `docs/notes/HANDOFF_黄条显现.md` | Image enhancement and yellow-streak visibility notes. / 黄条显现和图像增强交接说明。 |
| `dataset_yolo_toy/README.md` | Notes for the toy YOLO dataset. / toy YOLO 数据集说明。 |

## 5. Current Directory Layout / 当前目录结构

This is the current working layout after cleanup. Local data and generated artifacts are ignored by Git.

这是整理后的当前目录结构。客户数据和生成产物默认被 Git 忽略。

```text
.
├── README.md
├── ai.md
├── CLAUDE.md
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
│   ├── defect/
│   ├── ok/
│   └── reference_with_boxes/
├── pic/
│   ├── 已识别_不含框/
│   ├── 已识别_含框/
│   └── huangban-未识别/
├── huangban/
│   └── jpg/
├── OK/
├── annotations/
│   ├── labelme/
│   └── makesense_2026-06-29_030317/
├── dataset_yolo_toy/
│   ├── images/
│   ├── labels/
│   ├── data.yaml
│   └── README.md
├── enhanced/
│   └── LAB_b/
├── enhanced_v2/
│   ├── CLAHE/
│   └── Deviation/
├── reveal_out/
├── runs/
│   └── detect/
```

## 6. Directory Guide / 目录说明

| Path / 路径 | Meaning / 说明 |
|---|---|
| `data/defect/` | Organized yellow-stain defect images. Use these for formal training after annotation. / 整理后的黄斑缺陷图，标注后用于正式训练。 |
| `data/ok/` | Organized OK images. In YOLO detection, these should have empty label files. / 整理后的 OK 图，在 YOLO detection 中对应空标签。 |
| `data/reference_with_boxes/` | Reference images with boxes burned into pixels. Do not use as labels. / 带框参考图，框已烧进像素，不能直接当标签。 |
| `pic/已识别_不含框/` | First recommended folder for annotation practice and first clean labels. / 推荐优先标注的清晰黄斑原图。 |
| `pic/已识别_含框/` | Visual reference only. / 只作位置参考。 |
| `pic/huangban-未识别/` | Hard weak-signal images. Wait for expert guidance before annotation. / 弱信号难例，等专家确认后再标。 |
| `huangban/` | Earlier raw yellow-stain images and `.info` files. Can be used as a second annotation batch. / 早期黄斑原图和 `.info` 文件，可作为第二批标注。 |
| `OK/` | Original OK image folder. `data/ok/` is the organized copy. / 原始 OK 图目录，`data/ok/` 是整理版。 |
| `annotations/labelme/` | Legacy/unused Labelme JSON output folder. Current workflow does not use Labelme. / 历史遗留或备用目录；当前流程不使用 Labelme。 |
| `annotations/makesense_2026-06-29_030317/` | Archived makesense.ai YOLO labels for 13 images. / makesense.ai 导出的 13 张图 YOLO 标签归档。 |
| `dataset_yolo_toy/` | Small YOLO dataset for pipeline testing, not final training. / 小型 YOLO 测试数据集，不是正式训练集。 |
| `enhanced/`, `enhanced_v2/`, `reveal_out/` | Enhancement outputs for visual analysis only. / 图像增强输出，只用于辅助观察。 |
| `runs/detect/` | YOLO training and prediction outputs. / YOLO 训练和预测结果。 |
| `scripts/data/` | Dataset preparation and document conversion scripts. / 数据集整理和文档转换脚本。 |
| `scripts/experiments/` | Image enhancement and analysis experiments. / 图像增强和分析实验脚本。 |
| `scripts/train/` | Training and prediction helper scripts. / 训练和推理辅助脚本。 |
| `models/pretrained/` | Local pretrained weights, ignored by Git. / 本地预训练权重，Git 默认忽略。 |
| `external/ultralytics/` | Local Ultralytics source/code copy. Usually do not edit. / 本地 Ultralytics 代码，一般不改。 |

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
├── ai.md
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

Smoke test that has already run successfully:

已跑通的 smoke test：

```bash
yolo detect train \
  model=models/pretrained/yolo11n.pt \
  data=dataset_yolo_toy/data.yaml \
  epochs=3 \
  imgsz=640 \
  batch=2 \
  device=cpu \
  project=runs_yolo_toy \
  name=smoke_yolo11n \
  exist_ok=True
```

Typical GPU training command:

正式 GPU 训练命令通常类似：

```bash
yolo detect train \
  model=models/pretrained/yolo11n.pt \
  data=dataset_yolo_toy/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0
```

Before formal training, replace `dataset_yolo_toy/` with a real dataset generated from confirmed labels.

正式训练前，需要把 `dataset_yolo_toy/` 换成由正式标注生成的数据集。

## 13. Evaluation / 评估指标

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

## 14. Open Items / 待推进事项

Priority list:

优先级清单：

1. Get more confirmed defect images, especially weak and borderline cases. / 补充更多已确认黄斑图，尤其是弱信号和边界案例。
2. Get more OK images. / 补充更多 OK 合格品图。
3. Ask experts to annotate Type B weak yellow streaks. / 请质检专家示范标注 B 型弱黄条。
4. Confirm whether the client platform can export machine-readable labels. / 确认对方平台能否导出机器可读标签。
5. Confirm whether the platform can train and export `.evo`. / 确认平台是否能训练并导出 `.evo`。
6. Confirm acceptance test set, metrics, and false-negative / false-positive priority. / 确认验收测试集、指标以及漏检 / 误检优先级。
7. If the platform does not train the model, prepare cloud GPU or client GPU server. / 如果平台不负责训练，准备云 GPU 或需求方 GPU 服务器。

## 15. Delivery and Risks / 交付与风险

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
