#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成黄斑增强图（用于标注辅助）

对每张输入图输出一种增强版：
  - LAB_b:    LAB b 通道，magma 色板（客观可分性最强）

输出目录：./enhanced/
  enhanced/LAB_b/<原文件名>

用法：
  python enhance_for_labeling.py huangban/
  python enhance_for_labeling.py huangban/1.png huangban/2.png ...
"""

import sys
import os
import glob
from pathlib import Path

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[2]


def norm01(x, p_low=1, p_high=99):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def lab_b_image(img_rgb):
    """LAB b 通道 + magma 色板，偏黄区域显亮。"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = lab[:, :, 2]
    u8 = (norm01(b) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)  # BGR


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def process(img_path, rel_path, out_base):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  跳过（读不到）: {img_path}")
        return
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    lab_path = Path(out_base) / "LAB_b" / rel_path
    lab_dir = lab_path.parent
    os.makedirs(lab_dir, exist_ok=True)

    cv2.imwrite(str(lab_path), lab_b_image(img_rgb))
    print(f"  ✓  {rel_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    jobs = []
    for arg in sys.argv[1:]:
        input_path = Path(arg)
        if not input_path.is_absolute():
            input_path = ROOT / input_path
        if input_path.is_dir():
            for p in sorted(input_path.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    jobs.append((str(p), p.relative_to(input_path)))
        elif input_path.is_file():
            jobs.append((str(input_path), Path(input_path.name)))
        else:
            print(f"找不到: {arg}")

    if not jobs:
        print("没有找到图片文件。")
        sys.exit(1)

    out_base = str(ROOT / "enhanced")
    print(f"共 {len(jobs)} 张图 → 输出到 {out_base}/")
    for p, rel_path in jobs:
        process(p, rel_path, out_base)
    print(f"\n完成。")
    print(f"  enhanced/LAB_b/     ← 用这个标注，信号最稳")


if __name__ == "__main__":
    main()
