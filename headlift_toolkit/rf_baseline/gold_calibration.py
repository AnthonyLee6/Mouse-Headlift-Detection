# -*- coding: utf-8 -*-
"""
参数校准分析：系统参数网格 × 人工金标准一致性
==============================================

金标准验证发现的两个机制性问题：
  1. 连续两次抬头被"回中性区才再计次"的状态机合并（1635 漏检主因）
  2. 轻微抬头被系统计入而人工刻意排除（1631 多余检出主因）

本脚本在 k × neutral_band × min_dur 网格上重跑检测，以事件级 P/R/F1（对人工金标准）
考察哪个参数组合最接近人工判据，并检查论文默认参数 (k=1.5, neutral=10°, 0.15s)
距最优有多远。

※ 论文表述注意：这是"系统与人工判据的校准分析"（calibration analysis），
  说明性能对参数的依赖方向，不能写成"在金标准上调参后的最优性能"（循环论证）。

输出 → ../results/金标准验证/：
  参数校准_网格.csv / 参数校准_热力图.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.features import build_features  # noqa: E402
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS  # noqa: E402
from rf_baseline.validate_gold import match_events, WINDOW  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(BASE, "..", "results", "金标准验证"))

KS = [1.0, 1.25, 1.5, 1.75, 2.0]
NEUTRALS = [5.0, 10.0, 15.0, 20.0]
MINDURS = [0.15, 0.3]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs(OUT, exist_ok=True)
    gold = pd.read_csv(os.path.join(BASE, "gold_抬头事件.csv"))
    gold = gold[gold.note.isna()]

    data = {}
    for mouse in gold.mouse.unique():
        _, csv_name, video_name = VIDEOS[str(mouse)]
        csv_path = os.path.join(VIDEO_DIR, csv_name)
        F, meta = build_features(csv_path, os.path.join(VIDEO_DIR, video_name))
        g = gold[gold.mouse == mouse]
        gold_ivs = [(r.onset_s, r.offset_s + 1.0) for r in g.itertuples()]
        data[mouse] = (meta, gold_ivs)

    rows, all_gold = [], sum([d[1] for d in data.values()], [])
    for k in KS:
        for nb in NEUTRALS:
            for md in MINDURS:
                all_det = []
                per_mouse = {}
                for mouse, (meta, gold_ivs) in data.items():
                    fps = meta["fps"]
                    ev, _ = __import__("headlift").detect_lifts(
                        meta["angle"], fps, mode="dynamic", k=k,
                        neutral_band=nb, min_dur_s=md)
                    det = [(r.onset_frame / fps, (r.offset_frame + 1) / fps)
                           for _, r in ev.iterrows() if r.onset_frame / fps < WINDOW]
                    _, tp, fn, fp = match_events(gold_ivs, det)
                    p = tp / (tp + fp) if tp + fp else 0.0
                    r_ = tp / (tp + fn) if tp + fn else 0.0
                    per_mouse[mouse] = (tp, fp, fn, p, r_)
                    all_det += det
                _, tp, fn, fp = match_events(all_gold, all_det)
                p = tp / (tp + fp) if tp + fp else 0.0
                r_ = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
                rows.append({"k": k, "neutral_deg": nb, "min_dur_s": md,
                             "TP": tp, "FP": fp, "FN": fn,
                             "precision": round(p, 3), "recall": round(r_, 3),
                             "F1": round(f1, 3),
                             **{f"R_{m}": round(v[4], 2) for m, v in per_mouse.items()}})
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(OUT, "参数校准_网格.csv"), index=False, encoding="utf-8-sig")

    d15 = grid[grid.min_dur_s == 0.15]
    heat = d15.pivot(index="k", columns="neutral_deg", values="F1")
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    im = ax.imshow(heat.values, cmap="YlOrRd", vmin=heat.values.min(), vmax=heat.values.max(),
                   origin="lower")
    ax.set_xticks(range(len(heat.columns)), [f"{c:.0f}°" for c in heat.columns])
    ax.set_yticks(range(len(heat.index)), [f"{i:.2f}" for i in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    # 标记论文默认参数位置 (k=1.5, neutral=10)
    ax.plot([1], [1], "s", ms=14, mfc="none", mec="#0D9488", mew=2)
    ax.annotate("默认参数", (1, 1), textcoords="offset points", xytext=(8, 8),
                color="#0D9488", fontsize=9)
    ax.set_xlabel("中性区半宽（度）")
    ax.set_ylabel("阈值系数 k")
    ax.set_title("事件级 F1（对人工金标准）随参数变化（最短持续 0.15s）")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "参数校准_热力图.png"))
    plt.close(fig)

    best = grid.loc[grid.F1.idxmax()]
    default = grid[(grid.k == 1.5) & (grid.neutral_deg == 10) & (grid.min_dur_s == 0.15)].iloc[0]
    print("=== 网格（前 8 名，按 F1） ===")
    print(grid.sort_values("F1", ascending=False).head(8).to_string(index=False))
    print(f"\n默认参数 (k=1.5, neutral=10°, 0.15s): P {default.precision} "
          f"R {default.recall} F1 {default.F1}")
    print(f"网格最优 ({best.k}, {best.neutral_deg}°, {best.min_dur_s}s): "
          f"P {best.precision} R {best.recall} F1 {best.F1}")
    print(f"\n结果已输出 → {OUT}")


if __name__ == "__main__":
    main()
