# -*- coding: utf-8 -*-
"""绝对幅度门控检测：只认明确大抬头（峰值抬升 ≥ amp_deg，持续 ≥ min_dur）。"""
from __future__ import annotations

import itertools
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
ALL_MICE = ["1631", "1632", "1635", "1637", "2166", "2171"]


def detect_abs(ang, fps, amp_deg=50.0, min_dur_s=0.25, smooth_win=5, neutral_band=15.0):
    """相对中位角的绝对抬升幅度检测 + 中性区状态机。"""
    x = np.asarray(ang, dtype=float)
    if smooth_win and smooth_win > 1:
        x = pd.Series(x).rolling(smooth_win, center=True, min_periods=1).median().to_numpy()
    med = float(np.nanmedian(x))
    lift = med - x  # 抬头时为正
    min_frames = max(1, int(round(min_dur_s * fps)))
    events = []
    hold, start, in_lift, armed = 0, 0, False, True
    for i, a in enumerate(x):
        if np.isnan(a):
            if in_lift and hold >= min_frames:
                events.append((start, i - 1))
            hold, in_lift = 0, False
            continue
        if not in_lift and abs(a - med) <= neutral_band:
            armed = True
        is_lift = (lift[i] >= amp_deg) and armed
        if is_lift:
            hold += 1
            if hold >= min_frames and not in_lift:
                in_lift, start = True, i - hold + 1
        else:
            if in_lift:
                events.append((start, i - 1))
                armed = False
            hold, in_lift = 0, False
    if in_lift:
        events.append((start, len(x) - 1))
    return [(s / fps, (e + 1) / fps) for s, e in events]


def main():
    dfv = pd.read_csv(V2)
    gold = {m: [] for m in ALL_MICE}
    if len(dfv):
        for m, g in dfv.groupby(dfv.mouse.astype(str)):
            gold[m] = [(float(r.onset_s), float(r.offset_s)) for _, r in g.iterrows()]
    pack = {}
    for m in ALL_MICE:
        _, csv, vid = VIDEOS[m]
        F, meta = build_features(os.path.join(VIDEO_DIR, csv), os.path.join(VIDEO_DIR, vid))
        pack[m] = meta

    rows = []
    for amp, mind, band, smooth in itertools.product(
            [45.0, 50.0, 55.0, 60.0, 70.0],
            [0.15, 0.25, 0.35],
            [10.0, 15.0, 20.0],
            [5, 9]):
        all_g, all_s = [], []
        per = []
        for m in ALL_MICE:
            meta = pack[m]
            ivs = detect_abs(meta["angle"], meta["fps"], amp, mind, smooth, band)
            g = gold[m]
            pairs, tp, fn, fp = match_events(g, ivs)
            p = tp / (tp + fp) if tp + fp else 0.0
            r_ = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
            per.append((m, len(g), len(ivs), tp, fp, fn, f1))
            all_g.extend(g)
            all_s.extend(ivs)
        pairs, tp, fn, fp = match_events(all_g, all_s)
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        rows.append({
            "method": f"abs amp={amp} min={mind} band={band} sm={smooth}",
            "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3),
            "TP": tp, "FP": fp, "FN": fn, "n_det": len(all_s),
            "per": per,
        })
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "per"} for r in rows])
    df = df.sort_values("F1", ascending=False)
    path = os.path.join(OUT, "严格大抬头_绝对幅度扫描.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(df.head(20).to_string(index=False))

    # top 3 per-mouse
    for r in sorted(rows, key=lambda x: -x["F1"])[:3]:
        print("\n", r["method"], "F1", r["F1"], "P", r["P"], "R", r["R"], "n", r["n_det"])
        for t in r["per"]:
            print("  ", t)
    print("->", path)


if __name__ == "__main__":
    main()
