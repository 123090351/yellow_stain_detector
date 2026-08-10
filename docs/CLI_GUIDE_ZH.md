# 黄斑检测 CLI 操作指南

这份指南用于 HKPC GPU container / Web VS Code 终端。所有命令都从项目目录执行。
为避免反斜杠换行导致 `--model: command not found`，主要命令均写成单行。

## 1. 每次打开新终端先执行

```bash
cd /data/greya/yellow_stain_detector
DS=/data/greya/new_data/xabat_huangban_data_20260731/huangban_v3
MODEL=/data/greya/yellow_stain_detector/runs/detect/xabat_img640_freeze3_seed100_e200_b16/weights/best.pt
RUNS=/data/greya/yellow_stain_detector/runs/detect
```

`DS`、`MODEL` 和 `RUNS` 只是当前终端中的路径简称。新开终端或 Pod 重建后需要重新设置。
`/path/to/...` 只表示“替换成真实路径”，不能直接复制运行。

确认路径：

```bash
echo "$DS"
ls -lh "$MODEL"
find "$DS/images/test" -type f | wc -l
```

当前 Xabat 数据集应有：

```text
train: 2224 images
val:    476 images
test:   474 images
```

## 2. 检查运行环境

```bash
yolo version
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
df -h /data/greya
df -h /dev/shm
```

`nvidia-smi` 正常且 `torch.cuda.is_available()` 返回 `True`，才可使用 `device=0`。

## 3. Pretrained 与训练的关系

本项目没有单独执行从零开始的 pre-training。`yolo11m.pt` 是通用预训练权重，
正式训练是在黄斑数据上进行 fine-tuning：

```text
yolo11m.pt -> yellow stain training -> best.pt
```

首次使用 `yolo11m.pt` 时，Ultralytics 会在联网环境下自动下载。
目录中的 `yolo26n.pt` 不是当前正式模型。

## 4. 正式训练

先确认数据 YAML 的真实名称：

```bash
find "$DS" -maxdepth 2 -type f \( -name '*.yaml' -o -name '*.yml' \) -print
```

假设结果是 `$DS/data.yaml`，当前 seed 100 训练命令为：

```bash
yolo detect train model=yolo11m.pt data="$DS/data.yaml" epochs=200 imgsz=640 batch=16 freeze=3 patience=30 device=0 workers=16 seed=100 dropout=0.0 hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 bgr=0.0 mosaic=0.0 close_mosaic=0 erasing=0.0 project="$RUNS" name=xabat_img640_freeze3_seed100_e200_b16
```

如果 YAML 文件名不同，只替换 `data=` 后面的路径。

查看训练是否仍在运行：

```bash
pgrep -af 'yolo.*train'
nvidia-smi
tail -n 5 "$RUNS/xabat_img640_freeze3_seed100_e200_b16/results.csv"
```

训练结束后使用：

```text
best.pt  最佳 validation checkpoint，用于评估和部署
last.pt  最后一轮 checkpoint，主要用于恢复中断训练
```

恢复意外中断的训练：

```bash
yolo detect train resume model="$RUNS/xabat_img640_freeze3_seed100_e200_b16/weights/last.pt"
```

## 5. Validation 上选择图片级 threshold

不要在 test 上反复调整 threshold。先在 validation 上以低阈值保存所有候选框：

```bash
yolo detect predict model="$MODEL" source="$DS/images/val" imgsz=640 conf=0.001 device=0 save=False save_txt=True save_conf=True project="$RUNS" name=xabat_seed100_val_lowconf exist_ok=True
```

扫描 threshold：

```bash
python scripts/eval/sweep_image_level_thresholds.py --gt-labels "$DS/labels/val" --pred-labels "$RUNS/xabat_seed100_val_lowconf/labels" --start 0.001 --stop 0.500 --step 0.001 --target-recall 0.99 --output-csv "$RUNS/xabat_seed100_val_lowconf/threshold_sweep.csv" --output-plot "$RUNS/xabat_seed100_val_lowconf/threshold_sweep.png"
```

当前交付 threshold 为 `0.175`。它表示图片中只要至少一个黄斑框的 confidence
不低于 0.175，就把整张图片判为 NG。

## 6. 锁定后运行 test

用 PT 模型预测固定 test split：

```bash
yolo detect predict model="$MODEL" source="$DS/images/test" imgsz=640 conf=0.175 device=0 save=False save_txt=True save_conf=True project="$RUNS" name=xabat_seed100_pt_test_conf0175 exist_ok=True
```

计算图片级 OK/NG 指标：

```bash
python scripts/eval/evaluate_image_level_ok_ng.py --gt-labels "$DS/labels/test" --pred-labels "$RUNS/xabat_seed100_pt_test_conf0175/labels"
```

如果 Ultralytics 显示实际保存到了 `runs/detect/runs/detect/...`，应以终端打印的
`Results saved to` 为准，把该真实目录下的 `labels` 传给 `--pred-labels`。

## 7. Box-level test

Box-level 指标评估框的位置和检测质量：

```bash
yolo detect val model="$MODEL" data="$DS/data.yaml" split=test imgsz=640 conf=0.001 device=0 workers=12 plots=True project="$RUNS" name=xabat_seed100_box_test exist_ok=True
```

重点输出包括 Box Precision、Box Recall、mAP50 和 mAP50-95。图片级 OK/NG 指标
与 box-level 指标用途不同，生产目标应同时记录两组结果。

## 8. 对无标签生产图片进行推理

输入目录不需要 label：

```bash
INPUT=/data/greya/inference_input
OUTPUT=/data/greya/inference_output
```

```bash
python scripts/infer/predict_ok_ng.py --model "$MODEL" --source "$INPUT" --output "$OUTPUT" --imgsz 640 --threshold 0.175 --device 0 --save-images
```

输出包括：

```text
decisions.csv   每张图片的 OK/NG、框数量和最高 confidence
summary.json    本批次总数、OK 数和 NG 数
annotated/      保存画框后的图片（仅在使用 --save-images 时生成）
```

## 9. 导出与测试 ONNX

导出只是转换格式，不会重新训练：

```bash
yolo export model="$MODEL" format=onnx imgsz=640 simplify=True dynamic=False
```

查找导出的模型：

```bash
find "$RUNS/xabat_img640_freeze3_seed100_e200_b16" -type f -name '*.onnx' -print
```

设置真实 ONNX 路径：

```bash
ONNX="$RUNS/xabat_img640_freeze3_seed100_e200_b16/weights/best.onnx"
ls -lh "$ONNX"
```

使用 ONNX 进行图片级推理：

```bash
python scripts/infer/predict_ok_ng.py --model "$ONNX" --source "$DS/images/test" --output /data/greya/onnx_test_output --imgsz 640 --threshold 0.175 --device cpu --save-images
```

也可以通过 YOLO CLI 保存带 confidence 的标签，再使用相同评估脚本：

```bash
yolo detect predict model="$ONNX" source="$DS/images/test" imgsz=640 conf=0.175 device=cpu save=False save_txt=True save_conf=True project="$RUNS" name=xabat_seed100_onnx_test_conf0175 exist_ok=True
python scripts/eval/evaluate_image_level_ok_ng.py --gt-labels "$DS/labels/test" --pred-labels "$RUNS/xabat_seed100_onnx_test_conf0175/labels"
```

已记录的 ONNX test 结果：

```text
test images: 474
TP / FP / TN / FN: 275 / 8 / 191 / 0
accuracy: 0.983
precision: 0.972
NG recall: 1.000
OK accuracy: 0.960
```

## 10. 常见错误

### `FileNotFoundError: /path/to/...`

`/path/to` 是示例，必须替换为真实路径。

### `FileNotFoundError: /images/test`

说明 `$DS` 为空。重新执行第 1 节的变量设置，并运行 `echo "$DS"` 检查。

### `--model: command not found`

多行命令的反斜杠后存在空格或空行。改用本指南提供的单行命令。

### `Invalid CUDA device=0` 或 `torch.cuda.is_available(): False`

当前 Pod 没有正常获得 GPU。先运行 `nvidia-smi`；若仍失败，需要重建或重启 Pod，
而不是重新训练模型。模型和数据只要位于 `/data/greya` PVC 就不会因此丢失。

### `pgrep -af 'yolo.*train'` 没有输出

表示当前没有训练进程，不是报错。

