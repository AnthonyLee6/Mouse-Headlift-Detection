# -*- coding: utf-8 -*-
"""清理 v2 重复事件，并对已完成小鼠做全长金标准 vs 系统对比。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import headlift as hl
from rf_baseline.features import build_features
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS
from rf_baseline.validate_gold import match_events

BASE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(BASE, "gold_抬头事件_v2.csv")
OUT = os.path.normpath(os.path.join(BASE, "..", "results", "融合优化"))


def load_clean():
    df = pd.read_csv(V2)
    df = df.sort_values(["mouse", "onset_frame", "offset_frame"])
    # 近重复合并：同鼠、起点差 <3 帧
    keep = []
    for mouse, g in df.groupby("mouse", sort=False):
        prev = None
        for _, r in g.iterrows():
            if prev is None:
                keep.append(r)
                prev = r
                continue
            if abs(int(r.onset_frame) - int(prev.onset_frame)) <= 3:
                # 保留更长的
                if (r.offset_frame - r.onset_frame) > (prev.offset_frame - prev.onset_frame):
                    keep[-1] = r
                    prev = r
                continue
            keep.append(r)
            prev = r
    clean = pd.DataFrame(keep)
    clean.to_csv(V2, index=False, encoding="utf-8-sig")
    return clean


def eval_vs_system(clean):
    rows = []
    for mouse, g in clean.groupby("mouse"):
        mouse = str(mouse)
        _, csv, vid = VIDEOS[mouse]
        F, meta = build_features(os.path.join(VIDEO_DIR, csv), os.path.join(VIDEO_DIR, vid))
        fps = meta["fps"]
        # 人工全长事件（v2）
        gold_ivs = [(float(r.onset_s), float(r.offset_s)) for _, r in g.iterrows()]
        # 系统规则 k=1.25 全长
        ev, _ = hl.detect_lifts(meta["angle"], fps, mode="dynamic", k=1.25, min_amp_deg=30.0)
        sys_ivs = []
        if len(ev):
            for _, r in ev.iterrows():
                sys_ivs.append((float(r.onset_s), float(r.offset_s) + 1.0 / fps))
        pairs, tp, fn, fp = match_events(gold_ivs, sys_ivs)
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        on = [abs(ds - gs) for gs, ge, ds, de in pairs]
        rows.append({
            "mouse": mouse,
            "人工事件_全长": len(gold_ivs),
            "系统事件_全长": len(sys_ivs),
            "TP": tp, "FP": fp, "FN": fn,
            "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3),
            "onset_mae_s": round(float(np.mean(on)), 3) if on else np.nan,
            "人工总时长s": round(sum(e - s for s, e in gold_ivs), 1),
            "系统总时长s": round(sum(e - s for s, e in sys_ivs), 1),
        })
        print(f"[{mouse}] 人工 {len(gold_ivs)} 系统 {len(sys_ivs)} "
              f"P={p:.2f} R={r_:.2f} F1={f1:.2f}")
    df = pd.DataFrame(rows)
    return df


def main():
    clean = load_clean()
    print("清理后条数:")
    print(clean.groupby("mouse").size())
    print("source:")
    print(clean.groupby(["mouse", "source"]).size().unstack(fill_value=0))
    df = eval_vs_system(clean)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, "全长金标准_v2_已完成鼠_对比.csv"),
              index=False, encoding="utf-8-sig")
    # 汇总
    all_g, all_s = [], []
    for mouse, g in clean.groupby("mouse"):
        mouse = str(mouse)
        _, csv, vid = VIDEOS[mouse]
        F, meta = build_features(os.path.join(VIDEO_DIR, csv), os.path.join(VIDEO_DIR, vid))
        fps = meta["fps"]
        all_g.extend([(float(r.onset_s), float(r.offset_s)) for _, r in g.iterrows()])
        ev, _ = hl.detect_lifts(meta["angle"], fps, mode="dynamic", k=1.25, min_amp_deg=30.0)
        if len(ev):
            for _, r in ev.iterrows():
                all_s.append((float(r.onset_s), float(r.offset_s) + 1.0 / fps))
    pairs, tp, fn, fp = match_events(all_g, all_s)
    p = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0
    print(f"\n汇总(仅已完成鼠, 全长): 人工{len(all_g)} 系统{len(all_s)} "
          f"P={p:.3f} R={r_:.3f} F1={f1:.3f}")
    print("输出 ->", os.path.join(OUT, "全长金标准_v2_已完成鼠_对比.csv"))


if __name__ == "__main__":
    main()
