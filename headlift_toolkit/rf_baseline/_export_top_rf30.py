# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import headlift as hl
from rf_baseline.features import build_features
from rf_baseline.optimize_round2 import MIN_AMP_DEG, OUT, eval_ivs, rf_proba_all, robust_angle
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS
from rf_baseline.validate_gold import WINDOW, load_gold

gold = load_gold()
pack, gold_ivs = {}, {}
for mouse in gold.mouse.unique():
    _, csv_name, video_name = VIDEOS[str(mouse)]
    F, meta = build_features(
        os.path.join(VIDEO_DIR, csv_name),
        os.path.join(VIDEO_DIR, video_name),
    )
    df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv_name))
    pack[mouse] = {"F": F, "meta": meta, "ang_rob": robust_angle(df),
                   "gold": gold[gold.mouse == mouse]}
    g = gold[gold.mouse == mouse]
    gold_ivs[mouse] = [(float(r.onset_s), float(r.offset_s) + 1.0) for r in g.itertuples()]

proba = rf_proba_all(pack)
THR, GAP, MIND = 0.5, 0.6, 0.25


def rf30_events(d, p):
    fps, n = d["meta"]["fps"], d["meta"]["n_frames"]
    t = np.arange(n) / fps
    pos = (p >= THR) & (t < WINDOW)
    if not pos.any():
        return []
    dpos = np.diff(pos.astype(int))
    starts = list(np.where(dpos == 1)[0] + 1)
    ends = list(np.where(dpos == -1)[0] + 1)
    if pos[0]:
        starts = [0] + starts
    if pos[-1]:
        ends = ends + [n]
    segs = []
    for s, e in zip(starts, ends):
        if segs and (s - segs[-1][1]) / fps <= GAP:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    ang = d["ang_rob"]
    med = float(np.nanmedian(ang))
    out = []
    for s, e in segs:
        if (e - s) / fps < MIND:
            continue
        if med - float(np.nanmin(ang[s:e])) < MIN_AMP_DEG:
            continue
        out.append((s / fps, e / fps))
    return out


det = {m: rf30_events(pack[m], proba[m]) for m in pack}
ev = eval_ivs(gold_ivs, det)
print("RF30", {k: ev[k] for k in ev if k != "per"})
for row in ev["per"]:
    print(row)

rule_det = {}
for mouse, d in pack.items():
    fps = d["meta"]["fps"]
    evr, _ = hl.detect_lifts(
        d["ang_rob"], fps, mode="dynamic", k=1.25, neutral_band=10.0,
        min_dur_s=0.15, require_neutral=True, min_amp_deg=MIN_AMP_DEG,
    )
    ivs = []
    if len(evr):
        for _, r in evr.iterrows():
            s, e = float(r.onset_s), float(r.offset_s) + 1.0 / fps
            if s < WINDOW:
                ivs.append((s, e))
    rule_det[mouse] = ivs
rev = eval_ivs(gold_ivs, rule_det)
print("RULE", {k: rev[k] for k in rev if k != "per"})
for row in rev["per"]:
    print(row)

pd.DataFrame(ev["per"]).to_csv(
    os.path.join(OUT, "第二轮最优_RF30逐鼠.csv"), index=False, encoding="utf-8-sig"
)
