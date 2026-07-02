#!/usr/bin/env python3
"""
颜色阈值实验：比较黄条图和OK图的 LAB b 通道分布
看两组数值能否用一个阈值切开

用法：python threshold_experiment.py
"""

import os
import glob
from pathlib import Path

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[2]


def get_b_stats(folder):
    """读取文件夹里所有图，返回每张图 LAB b 通道的统计数据。"""
    folder = str(folder)
    paths = sorted(glob.glob(os.path.join(folder, "*.png")) +
                   glob.glob(os.path.join(folder, "*.jpg")))
    results = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        b = lab[:, :, 2]
        results.append({
            "file": os.path.basename(p),
            "mean": b.mean(),
            "max": b.max(),
            "p95": np.percentile(b, 95),  # 95th 百分位（忽略极端高值）
            "p99": np.percentile(b, 99),
        })
    return results


def print_group(label, stats):
    print(f"\n{'='*50}")
    print(f"  {label}  ({len(stats)} 张图)")
    print(f"{'='*50}")
    print(f"{'文件':<35} {'均值':>6} {'最大':>6} {'p95':>6} {'p99':>6}")
    print("-" * 60)
    for s in stats:
        print(f"{s['file']:<35} {s['mean']:>6.1f} {s['max']:>6.1f} {s['p95']:>6.1f} {s['p99']:>6.1f}")

    means = [s["mean"] for s in stats]
    p95s  = [s["p95"]  for s in stats]
    print("-" * 60)
    print(f"{'组平均':<35} {np.mean(means):>6.1f} {'':>6} {np.mean(p95s):>6.1f}")


huangban_stats = (get_b_stats(ROOT / "huangban") +
                  get_b_stats(ROOT / "pic" / "已识别_不含框") +
                  get_b_stats(ROOT / "pic" / "huangban-未识别"))
ok_stats       = get_b_stats(ROOT / "OK")

print_group("黄条图（残次品）", huangban_stats)
print_group("OK 图（合格品）",  ok_stats)

# 简单判断两组能否分开
hb_p95s = [s["p95"] for s in huangban_stats]
ok_p95s = [s["p95"] for s in ok_stats]

hb_min = min(hb_p95s)
ok_max = max(ok_p95s)

print(f"\n{'='*50}")
print(f"  关键对比（p95 指标）")
print(f"{'='*50}")
print(f"黄条图 p95 最低值：{hb_min:.1f}")
print(f"OK 图  p95 最高值：{ok_max:.1f}")

if hb_min > ok_max:
    print(f"\n✅ 两组完全分开！阈值设在 {(hb_min + ok_max) / 2:.1f} 附近即可区分")
    print("   → 纯颜色阈值很可能就够用")
else:
    overlap = ok_max - hb_min
    print(f"\n⚠️  两组有重叠（重叠量：{overlap:.1f}）")
    print("   → 纯颜色阈值会误判，需要 YOLO 模型学位置和形状")
