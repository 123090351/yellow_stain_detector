#!/usr/bin/env python3
"""
增强版显影，专门针对弥散型（B型）黄条
输出到 enhanced_v2/CLAHE/

用法：python enhance_v2.py data/defect/
"""

import sys, os, glob
from pathlib import Path

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def clahe_b(img_bgr):
    """CLAHE 局部增强 LAB b 通道——对弥散型黄更敏感。"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    lab[:, :, 2] = clahe.apply(lab[:, :, 2])
    b_enhanced = lab[:, :, 2]
    return cv2.applyColorMap(b_enhanced, cv2.COLORMAP_MAGMA)


def deviation_map(img_bgr):
    """每个像素的 b 值偏离全图中位数的程度——突出局部异常。"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = lab[:, :, 2]
    median = np.median(b)
    dev = np.clip(b - median, 0, None)  # 只看偏黄方向
    dev_u8 = (dev / (dev.max() + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(dev_u8, cv2.COLORMAP_HOT)


def process(img_path, rel_path, out_base):
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"  跳过: {img_path}")
        return
    for subdir, fn in [("CLAHE", clahe_b), ("Deviation", deviation_map)]:
        out_path = Path(out_base) / subdir / rel_path
        os.makedirs(out_path.parent, exist_ok=True)
        cv2.imwrite(str(out_path), fn(bgr))
    print(f"  ✓  {rel_path}")


def main():
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
    if not jobs:
        print("用法：python enhance_v2.py data/defect/")
        return
    out_base = str(ROOT / "enhanced_v2")
    print(f"共 {len(jobs)} 张 → {out_base}/")
    for p, rel_path in jobs:
        process(p, rel_path, out_base)
    print(f"\n完成。")
    print(f"  enhanced_v2/CLAHE/      ← 局部增强，对弥散型更敏感")
    print(f"  enhanced_v2/Deviation/  ← 偏离中位数程度，纯看「比周围黄多少」")


if __name__ == "__main__":
    main()
