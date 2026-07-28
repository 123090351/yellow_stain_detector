# 黄斑检测模型交付说明

**版本日期：** 2026-07-28  
**候选模型：** Yellow Stain v3 / YOLO11m / seed 21

## 1. 当前运行环境是什么

当前开发环境运行在公司内部 Kubernetes GPU 平台。外部交付文档不记录内部
cluster、namespace、Deployment、Service 或 PVC 标识。可公开复现的环境信息为：

```text
development image: docker.io/greyaliu/yellow-stain-yolo:0.2
image digest: sha256:34c59244dc25a843c824259beb6fae675148252d73df5257701b2c62f565479a
```

开发镜像基于：

```text
pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
Python 3.11.9
PyTorch 2.4.1+cu121
Ultralytics 8.4.93
CUDA 12.1
code-server 4.128.0
```

Pod、container 和 PVC 不是交付物。它们只是当前平台上的运行实例和内部存储。
另一个团队需要的是源代码、模型文件、参数配置和可重建的 Docker image。

## 2. 正式交付内容

建议交付目录：

```text
yellow-stain-v3-delivery/
├── source/
│   └── yellow_stain_detector.tar.gz
├── model/
│   ├── best.pt
│   ├── model.yaml
│   └── sha256.txt
├── environment/
│   └── yellow-stain-inference-1.0.0.tar.gz
├── docs/
│   ├── DELIVERY_ZH.md
│   └── result.md
└── samples/
    ├── input/
    └── expected/
```

不应交付：

- 完整训练 dataset，除非已经得到跨公司数据授权；
- Hugging Face、DockerHub、GitHub token；
- code-server 密码；
- HKPC 内网地址、PVC 凭据或 Kubernetes secret；
- 所有历史 checkpoint 和 `runs/`；
- 带有未脱敏生产信息的样例图。

## 3. 锁定模型参数

配置文件为 `delivery/model.yaml`。关键参数：

```text
model: YOLO11m
checkpoint: best.pt
imgsz: 768
confidence threshold: 0.381
decision rule: 任意 huangban 框 confidence >= 0.381 时判定 NG，否则判定 OK
LAB-b rescue: disabled
```

锁定 test 结果：

```text
389 images
TP / FP / TN / FN: 136 / 5 / 248 / 0
accuracy: 0.987
precision: 0.965
NG recall: 1.000
OK accuracy: 0.980
```

## 4. 从 HKPC 归档模型

在 Web VS Code 服务器终端执行：

```bash
cd /data/greya/yellow_stain_detector
MODEL=$(find /data/greya/yellow_stain_detector/runs/detect -type f -path '*v3_baseline_img768_freeze3_seed21*/train/imgsz768_freeze3_seed21/weights/best.pt' | head -n 1)
DELIVERY=/data/greya/delivery/yellow-stain-v3-20260728
mkdir -p "$DELIVERY/model" "$DELIVERY/docs"
cp "$MODEL" "$DELIVERY/model/best.pt"
cp delivery/model.yaml "$DELIVERY/model/model.yaml"
cp result.md "$DELIVERY/docs/result.md"
cp docs/DELIVERY_ZH.md "$DELIVERY/docs/DELIVERY_ZH.md"
sha256sum "$DELIVERY/model/best.pt" | tee "$DELIVERY/model/sha256.txt"
```

检查模型：

```bash
ls -lh "$DELIVERY/model/best.pt"
sha256sum -c "$DELIVERY/model/sha256.txt"
```

## 5. 打包源代码

只交付已提交、可追溯的 Git 代码，不要直接压缩整个 PVC：

```bash
cd /data/greya/yellow_stain_detector
git status --short
git rev-parse HEAD
git archive --format=tar.gz --output=/data/greya/delivery/yellow-stain-v3-20260728/source.tar.gz greya
sha256sum /data/greya/delivery/yellow-stain-v3-20260728/source.tar.gz
```

应在交付记录中写明 branch、commit SHA 和 Docker image digest。

## 6. 构建生产推理镜像

仓库中的原始 `Dockerfile` 是开发镜像，会启动 code-server。正式交付使用：

```text
docker/Dockerfile.inference
```

它不包含训练数据、不包含模型、不启动 Web VS Code。模型通过只读 volume
挂载，方便后续只更新 `best.pt`。

在有 Docker 的 Linux、Windows 或 macOS 电脑执行：

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.inference \
  -t greyaliu/yellow-stain-inference:1.0.0 \
  --load .
```

本机验证：

```bash
docker run --rm --gpus all \
  -v "$PWD/delivery/model:/model:ro" \
  -v "$PWD/samples/input:/input:ro" \
  -v "$PWD/samples/output:/output" \
  greyaliu/yellow-stain-inference:1.0.0 \
  --imgsz 768 \
  --threshold 0.381 \
  --device 0 \
  --save-images
```

输出：

```text
/output/decisions.csv
/output/summary.json
/output/annotated/
```

## 7. 在线或离线交付 Docker image

如果对方可以访问双方认可的私有 registry：

```bash
docker push greyaliu/yellow-stain-inference:1.0.0
docker buildx imagetools inspect greyaliu/yellow-stain-inference:1.0.0
```

把完整 digest 记录到交付单。正式环境应通过 digest 拉取，避免 tag 被覆盖。

如果对方不能访问 registry，导出离线镜像：

```bash
mkdir -p delivery/environment
docker save greyaliu/yellow-stain-inference:1.0.0 \
  | gzip > delivery/environment/yellow-stain-inference-1.0.0.tar.gz
sha256sum delivery/environment/yellow-stain-inference-1.0.0.tar.gz \
  > delivery/environment/sha256.txt
```

对方导入：

```bash
gunzip -c yellow-stain-inference-1.0.0.tar.gz | docker load
```

## 8. 对方环境要求

GPU 推理需要：

```text
Linux amd64
NVIDIA GPU
兼容 CUDA 12.1 的 NVIDIA driver
Docker Engine
NVIDIA Container Toolkit
```

对方先验证：

```bash
docker run --rm --gpus all \
  greyaliu/yellow-stain-inference:1.0.0 \
  --help
```

如果使用 Kubernetes，还需要另外提供 Deployment YAML，将模型挂载到
`/model/best.pt`，输入输出挂载到 `/input` 和 `/output`。Kubernetes YAML
不是模型本身的一部分，应根据对方公司的 namespace、GPU resource name、
storage class 和安全策略重新配置。

## 9. 验收要求

交付双方应共同确认：

1. `best.pt` SHA-256 校验通过；
2. Docker image digest 与交付记录一致；
3. 样例图片的 `decisions.csv` 与预期一致；
4. `imgsz=768`、`threshold=0.381` 未被修改；
5. 对方环境能识别 NVIDIA GPU；
6. 使用一批未参与训练、调参和历史 test 的新 OK+NG 图片进行最终验收；
7. 重点报告 FN，其次报告 FP、延迟和显存。

当前结果是开发候选，不应直接表述为生产环境永久准确率 98.7%。正式验收需要
未来独立批次，并保留原图、模型 digest、配置、输出 CSV 和审核记录。
