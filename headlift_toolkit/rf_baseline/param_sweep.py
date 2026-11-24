# -*- coding: utf-8 -*-
"""
参数敏感性扫描（只用规则法，无需人工标注，可直接产出论文素材）
==============================================================

对动态阈值的两个关键参数做单因素扫描：
  1. 阈值系数 k：   1.0 → 2.5（步长 0.25），最短持续固定 0.15s
  2. 最短持续时长： 0.05 → 0.5s，k 固定 1.5

回答审稿人问题："你们的方法是不是对手调参数敏感？"
实测结论（4 只鼠，动态阈值+5 帧滤波）：
  · 最短持续 0.05–0.15s 完全稳定（平台期），默认 0.15s 位于稳定区内；
    0.3s 起计数下降——真正短于 0.3s 的抬头被过滤，属预期截断；
  · 阈值系数 k 无平台：计数随 k 单调下降（阈值本身即行为定义，属预期）。
    组间排序在 k∈[1.0, 2.5] 全区间保持（1635 CNO 始终最高），
    论文写法应为"绝对计数随定义变化，但组间对比在该区间稳健"，勿写"存在稳定平台"。

输出 → ../results/：
  参数敏感性_阈值系数k.csv / 参数敏感性_最短持续.csv
  参数敏感性_扫描图.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import headlift as hl  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.normpath(os.path.join(os.path.dirname(BASE), "..", "video"))
OUT = os.path.normpath(os.path.join(os.path.dirname(BASE), "results"))

VIDEOS = {
    "1631": ("PBS", "1631 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv"),
    "1632": ("PBS", "1632 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv"),
    "1635": ("CNO", "1635 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv"),
    "1637": ("CNO", "1637 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv"),
}

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"1631": "#94A3B8", "1632": "#64748B", "1635": "#E8590C", "1637": "#0D9488"}


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for mouse, (group, fname) in VIDEOS.items():
        csv_path = os.path.join(VIDEO_DIR, fname)
        df = hl.load_dlc_csv(csv_path)
        fps = hl.read_fps(csv_path.replace(".csv", ".WMV"))
        data[mouse] = (group, df, fps, hl.head_angle(df, smooth_win=5))

    # ---------- 扫描 1：阈值系数 k ----------
    ks = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    rows = []
    for mouse, (group, df, fps, ang) in data.items():
        for k in ks:
            ev, _ = hl.detect_lifts(ang, fps, mode="dynamic", k=k, min_dur_s=0.15)
            rows.append({"小鼠": mouse, "组别": group, "k": k,
                         "抬头次数": len(ev),
                         "次每分钟": round(len(ev) / (len(ang) / fps) * 60, 2)})
    k_tab = pd.DataFrame(rows)
    k_tab.to_csv(os.path.join(OUT, "参数敏感性_阈值系数k.csv"), index=False, encoding="utf-8-sig")

    # ---------- 扫描 2：最短持续 ----------
    durs = [0.05, 0.1, 0.15, 0.3, 0.5]
    rows = []
    for mouse, (group, df, fps, ang) in data.items():
        for d in durs:
            ev, _ = hl.detect_lifts(ang, fps, mode="dynamic", k=1.5, min_dur_s=d)
            rows.append({"小鼠": mouse, "组别": group, "最短持续_s": d,
                         "抬头次数": len(ev),
                         "次每分钟": round(len(ev) / (len(ang) / fps) * 60, 2)})
    d_tab = pd.DataFrame(rows)
    d_tab.to_csv(os.path.join(OUT, "参数敏感性_最短持续.csv"), index=False, encoding="utf-8-sig")

    # ---------- 图：两面板扫描曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    for mouse, (group, _, _, _) in data.items():
        sub = k_tab[k_tab.小鼠 == mouse]
        axes[0].plot(sub.k, sub.次每分钟, "o-", color=COLORS[mouse],
                     label=f"{mouse} ({group})", ms=4)
        sub = d_tab[d_tab.小鼠 == mouse]
        axes[1].plot(sub.最短持续_s, sub.次每分钟, "o-", color=COLORS[mouse],
                     label=f"{mouse} ({group})", ms=4)
    axes[0].set_xlabel("动态阈值系数 k（阈值 = 中位数 − k·σ）")
    axes[0].set_title("阈值系数敏感性")
    axes[1].set_xlabel("最短持续时长（秒）")
    axes[1].set_title("最短持续敏感性")
    for ax in axes:
        ax.set_ylabel("抬头频率（次/分钟）")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(1.5 if ax is axes[0] else 0.15, color="#CBD5E1", ls="--", lw=1)
    fig.suptitle("规则法关键参数敏感性扫描（虚线 = 论文默认取值）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "参数敏感性_扫描图.png"))
    plt.close(fig)

    # 控制台摘要：k 在 1.25–2.0 内的相对波动
    print("=== k 扫描（次/分钟） ===")
    print(k_tab.pivot(index="k", columns="小鼠", values="次每分钟").to_string())
    stable = k_tab[(k_tab.k >= 1.25) & (k_tab.k <= 2.0)]
    spread = (stable.groupby("小鼠").次每分钟.agg(["max", "min"]))
    spread["相对波动%"] = ((spread["max"] - spread["min"]) / spread["min"] * 100).round(1)
    print("\nk∈[1.25, 2.0] 区间内各鼠最大相对波动：")
    print(spread["相对波动%"].to_string())
    print(f"\n结果已输出 → {OUT}")


if __name__ == "__main__":
    main()
