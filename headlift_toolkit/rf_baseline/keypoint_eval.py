# -*- coding: utf-8 -*-
"""
姿态模型评估：人工点击真值 vs DeepLabCut 预测
================================================

用 keypoint_gui.py 采集的人工关键点，评估 DLC 模型逐 bodypart 的定位误差：
  - RMSE / 中位误差（px），全帧 与 置信度≥0.6 帧分别报告
  - 低置信度帧占比（模型自己"心虚"的比例）
  - 与 DLC 自带评估（20 帧训练集、95/5 划分）的对照说明

输出 → ../results/关键点评估/
  关键点误差_逐视频.csv / 关键点误差_汇总.csv / 图_关键点误差.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS  # noqa: E402
from headlift import load_dlc_csv  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(BASE, "..", "results", "关键点评估"))
PCUTOFF = 0.6
PARTS = [("nose", "鼻尖"), ("head", "头顶"),
         ("left_ear", "左耳"), ("right_ear", "右耳")]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs(OUT, exist_ok=True)
    files = [f for f in os.listdir(os.path.join(BASE, "keypoint_labels"))
             if f.startswith("keypoint_labels_") and f.endswith(".csv")]
    lab = pd.concat([pd.read_csv(os.path.join(BASE, "keypoint_labels", f))
                     for f in files], ignore_index=True)
    lab = lab.drop_duplicates(["mouse", "frame_idx"], keep="last")
    if "stratum" not in lab.columns:
        man = pd.read_csv(os.path.join(BASE, "keypoint_manifest.csv"))
        man["mouse"] = man.mouse.astype(str)
        lab["mouse"] = lab.mouse.astype(str)
        lab = lab.merge(man[["mouse", "frame_idx", "stratum"]],
                        on=["mouse", "frame_idx"], how="left")
        lab["stratum"] = lab["stratum"].fillna("unknown")
    print(f"载入 {len(files)} 份标注，共 {len(lab)} 帧")

    kps = {m: load_dlc_csv(os.path.join(VIDEO_DIR, name))
           for m, (_, name, _) in VIDEOS.items()}

    rows, per_video = [], []
    for _, r in lab.iterrows():
        k = kps[str(r.mouse)]
        f = int(r.frame_idx)
        rec = {"mouse": r.mouse, "frame_idx": f, "stratum": r.stratum}
        for short, _cn in PARTS:
            hx, hy = r[f"{short}_x"], r[f"{short}_y"]
            px, py, plh = k[f"{short}_x"].iloc[f], k[f"{short}_y"].iloc[f], k[f"{short}_lh"].iloc[f]
            d = float(np.hypot(px - hx, py - hy))
            rec[f"err_{short}"] = d
            rec[f"lh_{short}"] = plh
        rows.append(rec)
        for short, _cn in PARTS:
            per_video.append({"mouse": r.mouse, "bodypart": short,
                              "err": float(np.hypot(
                                  kps[str(r.mouse)][f"{short}_x"].iloc[f] - r[f"{short}_x"],
                                  kps[str(r.mouse)][f"{short}_y"].iloc[f] - r[f"{short}_y"])),
                              "lh": float(kps[str(r.mouse)][f"{short}_lh"].iloc[f])})

    df = pd.DataFrame(rows)
    pv = pd.DataFrame(per_video)
    pv["置信段"] = np.where(pv.lh >= PCUTOFF, "置信度≥0.6", "置信度<0.6")

    tab = (pv.groupby(["bodypart", "置信段"]).err
           .agg(RMSE_px=lambda s: np.sqrt((s ** 2).mean()),
                中位误差px="median", 帧数="count")
           .round(2).reset_index())
    tab.to_csv(os.path.join(OUT, "关键点误差_逐bodypart.csv"),
               index=False, encoding="utf-8-sig")

    summ = (pv.groupby("bodypart").err
            .agg(RMSE_px=lambda s: np.sqrt((s ** 2).mean()),
                 中位误差px="median", P95误差px=lambda s: s.quantile(0.95))
            .round(2).reset_index())
    low = pv[pv.lh < PCUTOFF].groupby("bodypart").size() / pv.groupby("bodypart").size()
    summ["低置信度帧占比"] = low.reindex(summ.bodypart).round(3).values
    summ.to_csv(os.path.join(OUT, "关键点误差_汇总.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 逐 bodypart 汇总（px） ===")
    print(summ.to_string(index=False))

    per_m = (pv.groupby(["mouse", "bodypart"]).err
             .agg(RMSE_px=lambda s: np.sqrt((s ** 2).mean())).round(2).reset_index())
    per_m.to_csv(os.path.join(OUT, "关键点误差_逐视频.csv"), index=False, encoding="utf-8-sig")

    # 图：分组柱状（置信段）+ 中位误差标注
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200)
    parts = [p for p, _ in PARTS]
    x = np.arange(len(parts))
    for j, (seg, c) in enumerate([("置信度≥0.6", "#0D9488"), ("置信度<0.6", "#CBD5E1")]):
        vals = [tab[(tab.bodypart == p) & (tab.置信段 == seg)].RMSE_px
                for p in parts]
        vals = [v.iloc[0] if len(v) else 0 for v in vals]
        bars = ax.bar(x + (j - 0.5) * 0.32, vals, 0.32, label=seg, color=c)
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([cn for _, cn in PARTS])
    ax.set_ylabel("RMSE（像素）")
    ax.set_title("DLC 关键点定位误差 vs 人工点击真值（120 帧分层抽样）")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "图_关键点误差.png"))
    plt.close(fig)
    print(f"\n结果已输出 → {OUT}")


if __name__ == "__main__":
    main()
