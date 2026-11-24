# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import headlift as hl
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS
from rf_baseline.validate_gold import match_events

v2 = pd.read_csv(os.path.join(_HERE, "gold_抬头事件_v2.csv"))
v2 = v2.sort_values(["mouse", "onset_frame", "offset_frame"])
keep = []
for m, g in v2.groupby("mouse", sort=False):
    prev = None
    for _, r in g.iterrows():
        if prev is None or abs(int(r.onset_frame) - int(prev.onset_frame)) > 3:
            keep.append(r)
            prev = r
        elif (r.offset_frame - r.onset_frame) > (prev.offset_frame - prev.onset_frame):
            keep[-1] = r
            prev = r
v2 = pd.DataFrame(keep)
mice = ["1631", "1632", "1635", "1637", "2166", "2171"]
gold = {m: [(float(r.onset_s), float(r.offset_s))
            for _, r in v2[v2.mouse.astype(str) == m].iterrows()] for m in mice}


def evaluate(margin, mind, sm=5):
    rows = []
    all_g, all_s = [], []
    for m in mice:
        _, csv, vid = VIDEOS[m]
        df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv))
        fps = hl.read_fps(os.path.join(VIDEO_DIR, vid))
        ev, _ = hl.detect_lifts_yrule_df(df, fps=fps, margin_px=margin,
                                         min_dur_s=mind, smooth_win=sm)
        sys_ivs = [(float(r.onset_s), float(r.offset_s) + 1 / fps)
                   for _, r in ev.iterrows()] if len(ev) else []
        g = gold[m]
        pairs, tp, fn, fp = match_events(g, sys_ivs)
        p = tp / (tp + fp) if tp + fp else 0
        r_ = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0
        rows.append({"mouse": m, "人工": len(g), "系统": len(sys_ivs),
                     "TP": tp, "FP": fp, "FN": fn,
                     "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3)})
        all_g.extend(g)
        all_s.extend(sys_ivs)
    pairs, tp, fn, fp = match_events(all_g, all_s)
    p = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0
    return rows, {"P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3),
                  "TP": tp, "FP": fp, "FN": fn, "n_det": len(all_s), "n_gold": len(all_g)}


print("配置对比：是否把小幅度当抬头")
print(f"{'配置':20s} {'P':>6s} {'R':>6s} {'F1':>6s} {'检出':>6s} {'多报FP':>8s}")
for margin, mind, sm, label in [
    (2, 0.2, 5, "生产 2px/0.2s"),
    (0, 0.12, 5, "松 0px/0.12s"),
    (0, 0.15, 5, "0px/0.15s"),
    (-5, 0.15, 5, "很松 -5px/0.15s"),
    (-10, 0.2, 5, "极松 -10px/0.2s"),
]:
    rows, s = evaluate(margin, mind, sm)
    print(f"{label:20s} {s['P']:6.3f} {s['R']:6.3f} {s['F1']:6.3f} {s['n_det']:6d} {s['FP']:8d}")

print()
print("生产默认逐鼠:")
rows, _ = evaluate(2, 0.2, 5)
for r in rows:
    print(r)
