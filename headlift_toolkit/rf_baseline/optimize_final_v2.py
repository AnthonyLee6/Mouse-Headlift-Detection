# -*- coding: utf-8 -*-
"""最终 v2 金标准上的 y 规则参数扫描 + 混合策略。"""
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
MICE = ["1631", "1632", "1635", "1637", "2166", "2171"]


def load_gold_clean():
    df = pd.read_csv(V2)
    df = df.sort_values(["mouse", "onset_frame", "offset_frame"])
    keep = []
    for m, g in df.groupby("mouse", sort=False):
        prev = None
        for _, r in g.iterrows():
            if prev is None or abs(int(r.onset_frame) - int(prev.onset_frame)) > 3:
                keep.append(r)
                prev = r
            elif (r.offset_frame - r.onset_frame) > (prev.offset_frame - prev.onset_frame):
                keep[-1] = r
                prev = r
    clean = pd.DataFrame(keep)
    clean.to_csv(os.path.join(OUT, "gold_v2_final_clean.csv"), index=False, encoding="utf-8-sig")
    return clean


def ivs_from_ev(ev, fps):
    if not len(ev):
        return []
    return [(float(r.onset_s), float(r.offset_s) + 1.0 / fps) for _, r in ev.iterrows()]


def eval_pack(pack, gold, det_fn):
    all_g, all_s, rows = [], [], []
    for m in MICE:
        sys_ivs = det_fn(m, pack[m])
        g = gold[m]
        pairs, tp, fn, fp = match_events(g, sys_ivs)
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        rows.append({"mouse": m, "人工": len(g), "系统": len(sys_ivs),
                     "TP": tp, "FP": fp, "FN": fn,
                     "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3)})
        all_g.extend(g)
        all_s.extend(sys_ivs)
    pairs, tp, fn, fp = match_events(all_g, all_s)
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
    return {"n_gold": len(all_g), "n_det": len(all_s), "TP": tp, "FP": fp, "FN": fn,
            "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3), "per": rows}


def main():
    clean = load_gold_clean()
    print("清理后人工事件:")
    print(clean.groupby(clean.mouse.astype(str)).size())
    gold = {m: [(float(r.onset_s), float(r.offset_s))
                for _, r in clean[clean.mouse.astype(str) == m].iterrows()]
            for m in MICE}

    pack = {}
    for m in MICE:
        _, csv, vid = VIDEOS[m]
        df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv))
        fps = hl.read_fps(os.path.join(VIDEO_DIR, vid))
        ang = hl.head_angle(df, smooth_win=5).to_numpy()
        pack[m] = {"df": df, "fps": fps, "ang": ang,
                   "med": float(np.nanmedian(ang)),
                   "std": float(np.nanstd(ang))}

    rows = []

    # A. y 规则网格
    for margin, mind, sm, pc in itertools.product(
            [0, 2, 5, 8, 10, 15],
            [0.12, 0.15, 0.2, 0.25, 0.35],
            [5, 9],
            [0.5, 0.6]):
        def det(m, d, margin=margin, mind=mind, sm=sm, pc=pc):
            ev, _ = hl.detect_lifts_yrule(
                d["df"]["nose_y"], d["df"]["head_y"], fps=d["fps"],
                nose_lh=d["df"]["nose_lh"], head_lh=d["df"]["head_lh"],
                pcutoff=pc, margin_px=margin, min_dur_s=mind, smooth_win=sm)
            return ivs_from_ev(ev, d["fps"])
        ev = eval_pack(pack, gold, det)
        rows.append({"method": f"y m={margin} min={mind} sm={sm} pc={pc}",
                     "family": "y", "P": ev["P"], "R": ev["R"], "F1": ev["F1"],
                     "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"], "n_det": ev["n_det"]})

    # B. y 规则 + 俯仰角幅度门控（双条件）
    for margin, mind, sm, amp in itertools.product(
            [0, 5], [0.12, 0.2], [9], [0, 20, 30, 40]):
        def det(m, d, margin=margin, mind=mind, sm=sm, amp=amp):
            ev, _ = hl.detect_lifts_yrule(
                d["df"]["nose_y"], d["df"]["head_y"], fps=d["fps"],
                nose_lh=d["df"]["nose_lh"], head_lh=d["df"]["head_lh"],
                pcutoff=0.6, margin_px=margin, min_dur_s=mind, smooth_win=sm)
            if not len(ev) or amp <= 0:
                return ivs_from_ev(ev, d["fps"])
            ang, med, fps = d["ang"], d["med"], d["fps"]
            out = []
            for _, r in ev.iterrows():
                s, e = int(r.onset_frame), int(r.offset_frame)
                peak = med - float(np.nanmin(ang[s:e + 1]))
                if peak >= amp:
                    out.append((float(r.onset_s), float(r.offset_s) + 1.0 / fps))
            return out
        ev = eval_pack(pack, gold, det)
        rows.append({"method": f"y+amp m={margin} min={mind} amp={amp}",
                     "family": "yamp", "P": ev["P"], "R": ev["R"], "F1": ev["F1"],
                     "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"], "n_det": ev["n_det"]})

    # C. y OR 低阈值俯仰角（提召回）
    for margin, mind, k, band in itertools.product(
            [5, 8], [0.2, 0.25], [1.0, 1.25], [10.0]):
        def det(m, d, margin=margin, mind=mind, k=k, band=band):
            ev_y, _ = hl.detect_lifts_yrule(
                d["df"]["nose_y"], d["df"]["head_y"], fps=d["fps"],
                nose_lh=d["df"]["nose_lh"], head_lh=d["df"]["head_lh"],
                pcutoff=0.6, margin_px=margin, min_dur_s=mind, smooth_win=9)
            ev_a, _ = hl.detect_lifts(
                d["ang"], d["fps"], mode="dynamic", k=k, neutral_band=band,
                min_dur_s=0.15, min_amp_deg=30.0)
            ivs = ivs_from_ev(ev_y, d["fps"]) + ivs_from_ev(ev_a, d["fps"])
            if not ivs:
                return []
            ivs = sorted(ivs)
            mrg = [list(ivs[0])]
            for s, e in ivs[1:]:
                if s <= mrg[-1][1]:
                    mrg[-1][1] = max(mrg[-1][1], e)
                else:
                    mrg.append([s, e])
            return [(s, e) for s, e in mrg]
        ev = eval_pack(pack, gold, det)
        rows.append({"method": f"UNION y{margin}/{mind}+ang k={k}",
                     "family": "union", "P": ev["P"], "R": ev["R"], "F1": ev["F1"],
                     "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"], "n_det": ev["n_det"]})

    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    path = os.path.join(OUT, "最终v2_算法扫描.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("\n========== TOP 25 ==========")
    print(df.head(25).to_string(index=False))
    print("\nfamily max:")
    print(df.groupby("family").F1.max())
    print("->", path)

    # 最优逐鼠
    top = df.iloc[0]
    print("\nTOP:", top.method, "F1", top.F1, "P", top.P, "R", top.R)


if __name__ == "__main__":
    main()
