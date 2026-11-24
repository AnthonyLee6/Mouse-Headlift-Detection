# -*- coding: utf-8 -*-
"""可视化：全长人工 v2 vs 系统检出（时间轴 + 俯仰角）。"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import headlift as hl
from rf_baseline.features import build_features
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(BASE, "gold_抬头事件_v2.csv")
OUT = os.path.normpath(os.path.join(BASE, "..", "results", "融合优化"))
ALL_MICE = ["1631", "1632", "1635", "1637", "2166", "2171"]


def main():
    gold_df = pd.read_csv(V2)
    fig, axes = plt.subplots(len(ALL_MICE), 1, figsize=(12, 1.9 * len(ALL_MICE)), dpi=160)
    if len(ALL_MICE) == 1:
        axes = [axes]
    for ax, mouse in zip(axes, ALL_MICE):
        _, csv, vid = VIDEOS[mouse]
        F, meta = build_features(os.path.join(VIDEO_DIR, csv), os.path.join(VIDEO_DIR, vid))
        fps = meta["fps"]
        ang = np.asarray(meta["angle"], dtype=float)
        t = np.arange(len(ang)) / fps
        g = gold_df[gold_df.mouse.astype(str) == mouse] if len(gold_df) else pd.DataFrame()
        ev, _ = hl.detect_lifts(ang, fps, mode="dynamic", k=1.25, min_amp_deg=30.0)
        sys_ivs = []
        if len(ev):
            for _, r in ev.iterrows():
                sys_ivs.append((float(r.onset_s), float(r.offset_s)))
        ax.plot(t, ang, lw=0.5, color="#64748B", label="俯仰角")
        med = float(np.nanmedian(ang))
        ax.axhline(med, color="#0D9488", ls="--", lw=0.8, alpha=0.7)
        # 人工
        for i, (_, r) in enumerate(g.iterrows()) if len(g) else []:
            ax.axvspan(r.onset_s, r.offset_s, ymin=0.72, ymax=0.95,
                       color="#0D9488", alpha=0.85, lw=0)
        # 系统
        for s, e in sys_ivs:
            # 是否命中人工
            hit = False
            if len(g):
                for _, r in g.iterrows():
                    if min(e, r.offset_s) - max(s, r.onset_s) > -0.75:
                        hit = True
                        break
            ax.axvspan(s, e, ymin=0.05, ymax=0.28,
                       color="#E8590C" if hit else "#CBD5E1", alpha=0.9, lw=0)
        ax.set_xlim(0, t[-1])
        ax.set_ylabel(mouse, rotation=0, ha="right", va="center", fontsize=10)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        n_h = len(g)
        n_s = len(sys_ivs)
        ax.set_title(f"{mouse}  人工{n_h} / 系统{n_s}", fontsize=9, loc="left")
    axes[-1].set_xlabel("时间（秒，全长约5分钟）")
    fig.suptitle("人工金标准（上·青） vs 系统 k=1.25+30°（下：橙=匹配，灰=多余检出）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, "全长_人工vs系统_时间轴.png")
    fig.savefig(path)
    plt.close(fig)
    print("->", path)

    # 单独放大 1637 / 2171
    for mouse in ["1637", "2171"]:
        _, csv, vid = VIDEOS[mouse]
        F, meta = build_features(os.path.join(VIDEO_DIR, csv), os.path.join(VIDEO_DIR, vid))
        fps = meta["fps"]
        ang = np.asarray(meta["angle"], dtype=float)
        t = np.arange(len(ang)) / fps
        ev, _ = hl.detect_lifts(ang, fps, mode="dynamic", k=1.25, min_amp_deg=30.0)
        fig, ax = plt.subplots(figsize=(12, 3.2), dpi=160)
        ax.plot(t, ang, lw=0.7, color="#334155")
        med = float(np.nanmedian(ang))
        thr = med - 1.25 * float(np.nanstd(ang))
        ax.axhline(med, color="#0D9488", ls="--", lw=1, label=f"中位 {med:.0f}°")
        ax.axhline(thr, color="#E8590C", ls="--", lw=1, label=f"阈值 {thr:.0f}°")
        if len(ev):
            for _, r in ev.iterrows():
                ax.axvspan(r.onset_s, r.offset_s, color="#E8590C", alpha=0.25)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("秒")
        ax.set_ylabel("俯仰角")
        ax.set_title(f"{mouse}：人工=0 次抬头，系统仍检出 {len(ev)} 次（橙色段）", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fp = os.path.join(OUT, f"误报诊断_{mouse}.png")
        fig.savefig(fp)
        plt.close(fig)
        print("->", fp)


if __name__ == "__main__":
    main()
