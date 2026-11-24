# -*- coding: utf-8 -*-
"""用 v2 全长人工金标准（含 0 事件小鼠）评估系统。"""
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
ALL_MICE = ["1631", "1632", "1635", "1637", "2166", "2171"]


def load_clean():
    df = pd.read_csv(V2)
    if len(df) == 0:
        return df
    df = df.sort_values(["mouse", "onset_frame", "offset_frame"])
    keep = []
    for mouse, g in df.groupby("mouse", sort=False):
        prev = None
        for _, r in g.iterrows():
            if prev is None:
                keep.append(r)
                prev = r
                continue
            if abs(int(r.onset_frame) - int(prev.onset_frame)) <= 3:
                if (r.offset_frame - r.onset_frame) > (prev.offset_frame - prev.onset_frame):
                    keep[-1] = r
                    prev = r
                continue
            keep.append(r)
            prev = r
    clean = pd.DataFrame(keep)
    clean.to_csv(V2, index=False, encoding="utf-8-sig")
    return clean


def main():
    clean = load_clean()
    print("v2 人工事件数（0=该鼠全长无抬头）:")
    for m in ALL_MICE:
        n = int((clean.mouse.astype(str) == m).sum()) if len(clean) else 0
        print(f"  {m}: {n}")

    rows = []
    all_g, all_s = [], []
    for mouse in ALL_MICE:
        _, csv, vid = VIDEOS[mouse]
        F, meta = build_features(os.path.join(VIDEO_DIR, csv), os.path.join(VIDEO_DIR, vid))
        fps = meta["fps"]
        g = clean[clean.mouse.astype(str) == mouse] if len(clean) else pd.DataFrame()
        gold_ivs = [(float(r.onset_s), float(r.offset_s)) for _, r in g.iterrows()] if len(g) else []
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
        })
        print(f"[{mouse}] 人工{len(gold_ivs)} 系统{len(sys_ivs)} "
              f"TP{tp} FP{fp} FN{fn} F1={f1:.3f}")
        all_g.extend(gold_ivs)
        all_s.extend(sys_ivs)

    df = pd.DataFrame(rows)
    pairs, tp, fn, fp = match_events(all_g, all_s)
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
    on = [abs(ds - gs) for gs, ge, ds, de in pairs]
    summ = {
        "mouse": "汇总", "人工事件_全长": len(all_g), "系统事件_全长": len(all_s),
        "TP": tp, "FP": fp, "FN": fn,
        "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3),
        "onset_mae_s": round(float(np.mean(on)), 3) if on else np.nan,
    }
    df = pd.concat([df, pd.DataFrame([summ])], ignore_index=True)
    path = os.path.join(OUT, "全长金标准_v2_6鼠_对比.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("\n", df.to_string(index=False))
    print("\n->", path)


if __name__ == "__main__":
    main()
