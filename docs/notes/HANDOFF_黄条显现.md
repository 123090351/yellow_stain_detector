# 黄条 / 黄斑 图像显现 · 工作交接说明

> **用途**:把网页端对话里梳理出的上下文,交接给终端 / Claude Code 会话,以便在本机真实文件上直接操作。
> **面向读者**:接手的终端 Claude(下文称"你")。
> **配套文件**:`reveal_streaks.py`(同目录;全文见本文末附录)、`yellow_stain_detection_plan_client.md`(完整项目计划,如在目录中)。

---

## 1. 项目背景(一句话)

为「海绵胸杯」做缺陷视觉检测。现有**黑点**的 YOLO 系统识别率已 99%+;本项目新增**「黄斑」**识别能力——但其实际形态是**黄条 / 线状污渍**(细长、边界模糊、偏黄),不是团块。最终交付物是训练好的 `.evo` 模型。完整计划见项目计划书。

## 2. 当前卡点 → 第一步只做「图像显现」

黄条**肉眼几乎看不见**:约一半勉强可见(一条偏黄的细线),另一半与乳白底完全融为一体、看不出来。**看不见就没法标注**。

所以现在**不要急着训练或选模型**。第一步是:**用图像处理把黄条显现出来,显出来之后再人工标注**。这份交接的核心任务就是这一步。

## 3. 关键资产:配对的 `.info`(已含黄斑框)【必读】

每张 `*.png` 都配一个**同名** `*.info`,是这套检测系统输出的**自定义 JSON**(后缀不通用,但内容是纯文本 JSON,直接当文本读即可)。一个 `.info` 记录了该图所有检出目标,**其中已经包含「黄斑」框**。

字段结构(以实际样本为准):

| 字段 | 含义 |
|---|---|
| `iObjNum` | 该图检出目标总数 |
| `iResNum` | 经可见性过滤后的目标数(≈ `bVisible:true` 的个数;样本中 9 检出、6 可见,与之吻合) |
| `pstObjectInfo[]` | 目标数组,每个元素如下 |
| `cn_tag` / `tag` | 类别。**用 `cn_tag` 区分更稳**:`黑点` / `黄斑`。`tag` 不一致:黑点是通用的 `quexian`,黄斑是 `huangban` |
| `fProb` | 置信度 |
| `fArea` | 面积(px²) |
| `bVisible` | 是否显示 / 计入(见下方阈值线索) |
| `stPosi` | 正外接矩形:`x,y`(左上角)+ `width,height` |
| `stRotPosi` | **旋转矩形四角点** `stP1..stP4`。系统用的是**带角度的检测框(OBB)** |
| `fOffsetAngle` | 旋转角(黄条样本 64.6°;小黑点默认 90°) |
| `pstrHash` / `pstrVersion` | 记录哈希 / 格式版本(`1.0.0`) |

**这份资产的两个用途**:
1. **量化验证**:`.info` 给了**已知黄条的位置**。可据此量化"每种处理方法把黄条与干净背景拉开了多少"(d′ 分数),客观排名,不靠肉眼猜。
2. **数据驱动黄轴**:从已知黄条学出其**真实颜色方向**,投影到整张图,看**没被框的隐形黄条会不会一并显现**。

> **【重要警告】** `.info` 里的黄斑框**大概率来自"正要被替换的旧黄斑模型"**(样本里那条黄斑 `fProb` 仅约 0.49,黑点也普遍是 0.1~0.2 的低分)。**当作候选标签 / 弱标签,务必人工核验,不要当成真值直接训练**。它能定位"可见的一半",但**很可能也漏了隐形的一半**。

> **阈值线索**:样本里 3 个黑点 `bVisible:false`,其 `fProb` 都极低(<0.04)。`bVisible × fProb` 的关系藏着现有系统的判定阈值策略,对后续"漏检 / 误检"调参有参考价值——多收集几个 `.info` 可反推。

## 4. 数据与环境约束(请遵守)

- **图像**:PNG / 无损 / sRGB / **8-bit per channel** / 2448×2048 / 约 6.5MB。
- **无损**:没有 JPEG 的色度二次采样损失——细黄线属于"高频色度细节",若存成 JPEG 早被压掉了;PNG 把它完整保留。信号只要当时被采进来,现在就还在文件里。
- **采集已固定,无法重拍**。**后处理是唯一杠杆**——不要建议换光源 / UV / 偏振 / 漫射光 / 重拍,这些方案已被排除。
- **8-bit 是天花板来源**:最弱的黄条在蓝通道可能只差两三级灰阶,量化误差就是上限。不致命,但决定"可达指标"。
  - 唯一可争取的余量:若这批 PNG 是从 RAW 导出的,用同一次曝光重导一份 **16-bit TIFF** 会显著增加弱信号余量(这不算重拍)。**可向用户确认是否有 RAW**。

## 5. 工具:`reveal_streaks.py`(已写好,并在合成数据上验证可跑)

吃一张 `png` + 配对 `.info`,跑 8 种处理方法并排对比,并用 `.info` 的黄斑框做可分性打分。

**8 种方法**:原图 RGB / LAB b 通道(固定黄轴)/ **数据驱动黄轴投影**(最该盯)/ **去相关拉伸 DStretch**(显现最直观)/ 背景压平(减曲面梯度、只看局部偏离)/ CLAHE 局部对比 / **Frangi 线状结构**(用"成线"几何把弱黄信号从噪声里捞出)/ 颜色 × 线状(两条件 AND)。

**运行**:
```bash
pip install numpy opencv-python scikit-image matplotlib
python reveal_streaks.py <图>.png <配对>.info
# 不带 .info 也能跑(仅无监督处理,会跳过打分与数据驱动方向)
```

**输出**(写到 `./reveal_out/`):
- `comparison_zoom.png` —— 已知黄条附近,各方法并排(**先看这张**:哪种方法最显)
- `comparison_overview.png` —— 整图并排(**再看这张**:隐形黄条有没有在别处一并冒出来)
- `1x_*.png` —— 每种方法的全分辨率图(看图器里自由放大,直接对照标注;与原图同分辨率,框可 1:1 回原图)
- `00_original_with_boxes.png` —— 原图叠 `.info` 已知框(绿=黄斑,青=黑点)
- 终端打印 **d′ 排名** —— 哪种方法在真图上客观最强

> 注:在不同图上"谁赢"会变。若污渍的黄偏离标准 b 轴,**数据驱动投影**通常反超固定 b 通道;Frangi 只有在颜色图本身够干净时才帮得上。一切以 d′ 排名 + 肉眼对照为准。

## 6. 即时任务(建议执行顺序)

1. **定位数据**:找到 png 与配对 `.info` 所在目录;确认命名对应关系(用户把某张重命名成了 `1.png`,其原名与对应 `.info` 同源)。装依赖。
2. **选样本跑**:挑 **可见黄条 / 几乎看不见 / 干净合格品** 各 2~3 张,逐张跑 `reveal_streaks.py`。
3. **判读**:对照 `comparison_zoom.png`、`comparison_overview.png`、终端 d′ 排名。
4. **出结论**(这正是项目计划书"第一周可分性评估"要的东西):
   - **信号清晰可分** → 数据驱动投影 / DStretch 等能稳定显现 → 走"在增强图上标注 + 监督式分割 / 检测"。
   - **信号弱 / 不稳** → 多通道、颜色归一化,可能转向异常检测;并建议与需求方再确认目标指标。
   - **挖不出来** → 多半是信号没被采进来,后处理到头,需回到采集端或确认 RAW / 16-bit。
5. **(若可分)定标注工作流**:在**最显的那种增强图**上标注;坐标与原图一致,可直接回贴。考虑**用 `.info` 已有黄斑框做标注种子**(再人工补隐形的、剔除误框)。
6. **回报用户**:哪种方法最有效、隐形黄条能否显现、是否需要 RAW / 16-bit、可分性结论。

## 7. 下游方向(知道即可,**先别做**)

- **路线选择**(语义分割 / 异常检测 / 检测 / 分类)取决于第 6 步的可分性结论与黄斑样本量——见项目计划书第六节。
- **`.evo` 导出对齐**:若仓库里有现有黑点系统的代码 / `.evo` 生成脚本,先记下位置供后续对齐,本步用不到。
- **真值来源问题(项目级,需向需求方确认)**:隐形那半黄条的"标准答案"从哪来?若唯一真值是"人盯 RGB 照片",隐形的按定义就无法标也无法验收——需要更强真值(实物在良好光照下复检 / 已确诊残次品清单)。这件事在所有图像处理的上游,值得早问。

---

## 附录:`reveal_streaks.py` 全文

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黄条/黄斑 显现 + 方法对比工具

目的:在"照片已定、肉眼看不见黄条"的前提下,尝试多种图像处理方法把黄条显现出来,
方便后续标注。配对的 .info(JSON)若提供,会用其中已知的黄斑框来:
  1) 量化每种方法的"可分性"(d′ 分数),客观排名;
  2) 用已知黄条的颜色方向构造"数据驱动黄轴",应用到整张图,
     看未被框出的隐形黄条会不会一并显现。

用法:
    python reveal_streaks.py 1.png 1.info
    python reveal_streaks.py 1.png            # 无 info,做无监督处理(跳过打分与数据驱动方向)

依赖:
    pip install numpy opencv-python scikit-image matplotlib

输出(写到 ./reveal_out/):
    00_original_with_boxes.png   原图叠加已知框(绿=黄斑, 青=黑点)
    1x_<method>.png              每种方法的全分辨率结果(可在看图器里自由放大、对照标注)
    comparison_overview.png      所有方法整图并排
    comparison_zoom.png          所有方法在"已知黄条"附近的放大并排
    并在终端打印各方法可分性排名
"""

import sys
import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.filters import frangi
    HAVE_FRANGI = True
except Exception:
    HAVE_FRANGI = False


def setup_cjk_font():
    """尝试设置中文字体(跨 mac/win/linux)。成功返回 True,否则 False(标题回退英文)。"""
    import matplotlib.font_manager as fm
    candidates = ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti",
                  "Arial Unicode MS", "Hiragino Sans GB",
                  "Microsoft YaHei", "SimHei",
                  "Noto Sans CJK SC", "Noto Sans CJK JP",
                  "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK_OK = setup_cjk_font()


# ----------------------------- 基础工具 -----------------------------

def norm01(x, p_low=1, p_high=99):
    """按百分位裁剪后归一化到 [0,1],抗离群点。"""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def to_u8(x):
    return (norm01(x) * 255).astype(np.uint8)


def load_info_boxes(path):
    """解析 .info,返回框列表。每个框含旋转四角点 pts、类别、置信度、是否可见。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    for o in data.get("pstObjectInfo", []):
        rp = o.get("stRotPosi", {})
        try:
            pts = np.array([
                [rp["stP1"]["x"], rp["stP1"]["y"]],
                [rp["stP2"]["x"], rp["stP2"]["y"]],
                [rp["stP3"]["x"], rp["stP3"]["y"]],
                [rp["stP4"]["x"], rp["stP4"]["y"]],
            ], dtype=np.int32)
        except (KeyError, TypeError):
            continue
        boxes.append({
            "tag": o.get("tag", ""),
            "cn": o.get("cn_tag", ""),
            "prob": o.get("fProb", None),
            "visible": o.get("bVisible", None),
            "pts": pts,
        })
    return boxes


def is_yellow(box):
    return box["tag"] == "huangban" or box["cn"] == "黄斑"


def poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m > 0


def clean_mask(shape, boxes, dilate=40):
    """所有框(膨胀后)之外的区域,作为'干净背景'采样。"""
    occ = np.zeros(shape[:2], np.uint8)
    for b in boxes:
        cv2.fillPoly(occ, [b["pts"].astype(np.int32)], 255)
    if dilate > 0 and occ.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        occ = cv2.dilate(occ, k)
    return occ == 0


def dprime(scalar_map, defect_mask, clean_msk):
    """可分性: (黄条均值 - 干净均值) / 干净标准差。越大越好分。"""
    d = scalar_map[defect_mask].astype(np.float32)
    c = scalar_map[clean_msk].astype(np.float32)
    if d.size == 0 or c.size == 0:
        return float("nan")
    return float((d.mean() - c.mean()) / (c.std() + 1e-6))


# ----------------------------- 处理方法 -----------------------------

def lab_b(img_rgb):
    """LAB 的 b 通道(固定蓝-黄轴),>128 偏黄。"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[:, :, 2]


def chroma_projection(img_rgb, defect_mask, clean_msk):
    """数据驱动黄轴:在 LAB 的 (a,b) 色度平面上,取已知黄条与干净背景的均值差方向,
    把每个像素投影上去。比固定 b 通道更贴合本批污渍的真实颜色方向。
    返回 (投影图, 方向向量)。无黄条样本时回退到 b 轴。"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ab = lab[:, :, 1:3]
    if defect_mask is None or defect_mask.sum() == 0:
        return ab[:, :, 1], np.array([0.0, 1.0])
    mu_d = ab[defect_mask].reshape(-1, 2).mean(0)
    mu_c = ab[clean_msk].reshape(-1, 2).mean(0)
    w = mu_d - mu_c
    n = np.linalg.norm(w)
    w = (w / n) if n > 1e-6 else np.array([0.0, 1.0])
    proj = ab @ w
    return proj, w


def dstretch(img_rgb):
    """去相关拉伸(DStretch):对颜色做 PCA 去相关 + 归一化方差 + 投回。
    把肉眼难辨的细微色差极度夸张化,显现/标注辅助利器。"""
    X = img_rgb.reshape(-1, 3).astype(np.float32)
    mu = X.mean(0)
    Xc = X - mu
    cov = np.cov(Xc.T) + np.eye(3) * 1e-3
    vals, vecs = np.linalg.eigh(cov)
    T = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    Y = Xc @ T
    out = np.zeros_like(Y)
    for i in range(3):
        out[:, i] = norm01(Y[:, i])
    return (out.reshape(img_rgb.shape) * 255).astype(np.uint8)


def flatten_bg(scalar_map, sigma=51):
    """背景压平:曲面有平滑的亮度/色彩梯度,绝对值没意义。
    用大核高斯当背景估计再相减,只保留局部偏离。"""
    bg = cv2.GaussianBlur(scalar_map.astype(np.float32), (0, 0), sigmaX=sigma)
    return scalar_map.astype(np.float32) - bg


def clahe_on(scalar_map, clip=3.0, grid=8):
    """CLAHE 局部对比增强(拉开几灰阶的微弱差异;会放大噪声)。"""
    u8 = to_u8(scalar_map)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return cl.apply(u8).astype(np.float32)


def ridge_filter(scalar_map, sig_max=7):
    """线状结构检测(Frangi 脊线滤波):专门响应细长的'线'。
    黄条单像素颜色弱,但成线 —— 用几何信息把它从斑块/噪声里捞出来。"""
    if not HAVE_FRANGI:
        return None
    x = norm01(scalar_map)
    return frangi(x, sigmas=range(1, sig_max), black_ridges=False)


# ----------------------------- 主流程 -----------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    img_path = sys.argv[1]
    info_path = sys.argv[2] if len(sys.argv) > 2 else None

    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"读不到图像: {img_path}")
        sys.exit(1)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    print(f"图像: {img_path}  ({W}x{H})")

    boxes, yellow_boxes, defect_mask, clean_msk = [], [], None, None
    if info_path and os.path.exists(info_path):
        boxes = load_info_boxes(info_path)
        yellow_boxes = [b for b in boxes if is_yellow(b)]
        print(f".info: {len(boxes)} 个框,其中黄斑 {len(yellow_boxes)} 个")
        if yellow_boxes:
            defect_mask = np.zeros((H, W), bool)
            for b in yellow_boxes:
                defect_mask |= poly_mask((H, W), b["pts"])
            clean_msk = clean_mask((H, W), boxes, dilate=40)
    else:
        print(".info 未提供 —— 跳过可分性打分与数据驱动黄轴(只做无监督处理)")

    # --- 构造各方法的图 ---
    # (中文名, 英文名, 图, 类型)
    methods = []

    methods.append(("原图 RGB", "Original RGB", img, "rgb"))

    b = lab_b(img)
    methods.append(("LAB b 通道(固定黄轴)", "LAB b (fixed)", b, "scalar"))

    if defect_mask is not None:
        proj, w = chroma_projection(img, defect_mask, clean_msk)
        print(f"数据驱动黄轴方向(LAB a,b): [{w[0]:+.3f}, {w[1]:+.3f}]")
        methods.append(("数据驱动黄轴投影", "Data-driven axis", proj, "scalar"))
        base = proj
    else:
        base = b  # 无监督时以固定 b 通道为基底

    methods.append(("去相关拉伸 DStretch", "DStretch", dstretch(img), "rgb"))

    flat = flatten_bg(base, sigma=51)
    methods.append(("背景压平(局部偏离)", "BG-flattened", flat, "scalar"))

    methods.append(("CLAHE 局部对比", "CLAHE", clahe_on(flat), "scalar"))

    ridges = ridge_filter(flat)
    if ridges is not None:
        methods.append(("线状结构 Frangi", "Frangi ridges", ridges, "scalar"))
        combined = norm01(flat) * norm01(ridges)
        methods.append(("颜色 × 线状", "Color x Line", combined, "scalar"))
    else:
        print("未装 scikit-image,跳过 Frangi 线检测(pip install scikit-image)")

    def label(cn, en):
        return cn if CJK_OK else en

    # --- 可分性打分 ---
    if defect_mask is not None:
        print("\n=== 可分性排名(d′,越大越好分) ===")
        scored = []
        for cn, en, m, kind in methods:
            if kind == "scalar":
                scored.append((cn, dprime(m, defect_mask, clean_msk)))
        for name, s in sorted(scored, key=lambda t: (np.isnan(t[1]), -t[1])):
            print(f"  {s:7.3f}   {name}")
        print()

    # --- 输出 ---
    outdir = "reveal_out"
    os.makedirs(outdir, exist_ok=True)

    # 原图叠框
    overlay = img.copy()
    for bx in boxes:
        color = (0, 220, 0) if is_yellow(bx) else (0, 200, 220)
        cv2.polylines(overlay, [bx["pts"]], True, color, 3)
        if bx["prob"] is not None:
            p = bx["pts"][0]
            cv2.putText(overlay, f"{bx['cn']} {bx['prob']:.2f}",
                        (int(p[0]), int(p[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(os.path.join(outdir, "00_original_with_boxes.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 每种方法全分辨率(标量图上 magma 色,信号更跳)
    def render_full(m, kind):
        if kind == "rgb":
            return cv2.cvtColor(m, cv2.COLOR_RGB2BGR)
        return cv2.applyColorMap(to_u8(m), cv2.COLORMAP_MAGMA)

    for i, (cn, en, m, kind) in enumerate(methods):
        safe = en.replace(" ", "_").replace("/", "_")
        cv2.imwrite(os.path.join(outdir, f"1x_{i}_{safe}.png"), render_full(m, kind))

    # 整图并排
    n = len(methods)
    cols = 4
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 4, rows * 3.4))
    for i, (cn, en, m, kind) in enumerate(methods):
        ax = plt.subplot(rows, cols, i + 1)
        if kind == "rgb":
            ax.imshow(m)
        else:
            ax.imshow(norm01(m), cmap="magma")
        if defect_mask is not None:
            for bx in yellow_boxes:
                ax.add_patch(plt.Polygon(bx["pts"], fill=False,
                                         edgecolor="lime", linewidth=1.2))
        ax.set_title(label(cn, en), fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_overview.png"), dpi=110)
    plt.close()

    # 已知黄条附近放大并排
    if yellow_boxes:
        all_pts = np.vstack([b["pts"] for b in yellow_boxes])
        x0, y0 = all_pts.min(0)
        x1, y1 = all_pts.max(0)
        pad = int(0.4 * max(x1 - x0, y1 - y0))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(W, x1 + pad), min(H, y1 + pad)

        plt.figure(figsize=(cols * 4, rows * 3.4))
        for i, (cn, en, m, kind) in enumerate(methods):
            ax = plt.subplot(rows, cols, i + 1)
            if kind == "rgb":
                ax.imshow(m[y0:y1, x0:x1])
            else:
                ax.imshow(norm01(m)[y0:y1, x0:x1], cmap="magma")
            for bx in yellow_boxes:
                shifted = bx["pts"] - np.array([x0, y0])
                ax.add_patch(plt.Polygon(shifted, fill=False,
                                         edgecolor="lime", linewidth=1.2))
            ax.set_title(label(cn, en), fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "comparison_zoom.png"), dpi=110)
        plt.close()

    print(f"完成 → ./{outdir}/")
    print("先看 comparison_zoom.png(已知黄条处哪种方法最显),"
          "再看 comparison_overview.png(隐形黄条是否在别处一并显现)。")


if __name__ == "__main__":
    main()

```
