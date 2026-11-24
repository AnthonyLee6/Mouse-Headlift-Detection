# -*- coding: utf-8 -*-
"""
按「只算明确大抬头」的全长 v2 金标准，收紧规则法参数。
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import headlift as hl
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS
from rf_baseline.validate_gold import match_events

BASE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(BASE, "gold_抬头事件_v2.csv")
OUT = os.path.normpath(os.path.join(BASE, "..", "results", "融合优化"))
ALL_MICE = ["1631", "1632", "1635", "1637", "2166", "2171"]


def gold_map():
    df = pd.read_csv(V2)
    out = {m: [] for m in ALL_MICE}
    if len(df):
        for m, g in df.groupby(df.mouse.astype(str)):
            out[m] = [(float(r.onset_s), float(r.offset_s)) for _, r in g.iterrows()]
    return out


def detect(meta, k, band, mind, min_amp, require_neutral=True, smooth=5):
    ang = hl.head_angle.__wrapped__ if False else None
    # recompute with smooth from DLC via meta angle already smooth=5
    ang = meta["angle"]
    fps = meta["fps"]
    ev, _ = hl.detect_lifts(
        ang, fps, mode="dynamic", k=k, neutral_band=band, min_dur_s=mind,
        min_amp_deg=min_amp, require_neutral=require_neutral,
    )
    ivs = []
    if len(ev):
        for _, r in ev.iterrows():
            ivs.append((float(r.onset_s), float(r.offset_s) + 1.0 / fps))
    return ivs


def eval_all(pack, gold, det_fn):
    all_g, all_s, rows = [], [], []
    for m in ALL_MICE:
        ivs = det_fn(m, pack[m])
        g = gold[m]
        pairs, tp, fn, fp = match_events(g, ivs)
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        rows.append({"mouse": m, "人工": len(g), "系统": len(ivs),
                     "TP": tp, "FP": fp, "FN": fn,
                     "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3)})
        all_g.extend(g)
        all_s.extend(ivs)
    pairs, tp, fn, fp = match_events(all_g, all_s)
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
    return {
        "n_gold": len(all_g), "n_det": len(all_s),
        "TP": tp, "FP": fp, "FN": fn,
        "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3),
        "per": rows,
    }


def main():
    gold = gold_map()
    print("gold", {m: len(gold[m]) for m in ALL_MICE}, "total", sum(len(v) for v in gold.values()))
    pack = {}
    for m in ALL_MICE:
        _, csv, vid = VIDEOS[m]
        from rf_baseline.features import build_features
        F, meta = build_features(os.path.join(VIDEO_DIR, csv), os.path.join(VIDEO_DIR, vid))
        pack[m] = meta

    rows = []
    for k, band, mind, amp in itertools.product(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            [10.0, 15.0, 20.0],
            [0.2, 0.3, 0.45],
            [30.0, 40.0, 50.0, 60.0]):
        ev = eval_all(pack, gold, lambda m, meta, k=k, band=band, mind=mind, amp=amp:
                      detect(meta, k, band, mind, amp))
        rows.append({
            "method": f"k={k} band={band} min={mind} amp={amp}",
            "P": ev["P"], "R": ev["R"], "F1": ev["F1"],
            "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"], "n_det": ev["n_det"],
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    path = os.path.join(OUT, "严格大抬头_参数扫描.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(df.head(25).to_string(index=False))
    print("->", path)

    # 最优逐鼠
    top = df.iloc[0]
    print("\nTOP", top.method, "F1", top.F1)
    # parse back is hard; just rerun a few candidates
    for cand in [
        dict(k=2.5, band=15.0, mind=0.3, amp=40.0),
        dict(k=3.0, band=15.0, mind=0.3, amp=50.0),
        dict(k=2.0, band=10.0, mind=0.3, amp=40.0),
        dict(k=3.5, band=20.0, mind=0.45, amp=50.0),
    ]:
        ev = eval_all(pack, gold, lambda m, meta, c=cand:
                      detect(meta, c["k"], c["band"], c["mind"], c["amp"]))
        print(cand, "F1", ev["F1"], "P", ev["P"], "R", ev["R"], "det", ev["n_det"])
        for r in ev["per"]:
            print("   ", r)


if __name__ == "__main__":
    main()
