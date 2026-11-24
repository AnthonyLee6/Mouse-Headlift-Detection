# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import headlift as hl
from rf_baseline.features import build_features
from rf_baseline.optimize_round2 import MIN_AMP_DEG, OUT, eval_ivs, rf_proba_all, robust_angle
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS
from rf_baseline.validate_gold import WINDOW, load_gold

gold = load_gold()
pack, gold_ivs = {}, {}
for mouse in gold.mouse.unique():
    _, csv, vid = VIDEOS[str(mouse)]
    F, meta = build_features(os.path.join(VIDEO_DIR, csv), os.path.join(VIDEO_DIR, vid))
    df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv))
    pack[mouse] = {"F": F, "meta": meta, "ang_rob": robust_angle(df),
                   "gold": gold[gold.mouse == mouse]}
    g = gold[gold.mouse == mouse]
    gold_ivs[mouse] = [(float(r.onset_s), float(r.offset_s) + 1.0) for r in g.itertuples()]

proba = rf_proba_all(pack)


def rule_ivs(d, k=1.25, band=10.0):
    fps = d["meta"]["fps"]
    ev, _ = hl.detect_lifts(
        d["ang_rob"], fps, mode="dynamic", k=k, neutral_band=band,
        min_dur_s=0.15, require_neutral=True, min_amp_deg=MIN_AMP_DEG,
    )
    out = []
    if len(ev):
        for _, r in ev.iterrows():
            s, e = float(r.onset_s), float(r.offset_s) + 1.0 / fps
            if s < WINDOW:
                out.append((s, e))
    return out


def rf_ivs(d, p, thr=0.5, gap=0.6, mind=0.25):
    fps, n = d["meta"]["fps"], d["meta"]["n_frames"]
    t = np.arange(n) / fps
    pos = (p >= thr) & (t < WINDOW)
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
        if segs and (s - segs[-1][1]) / fps <= gap:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    ang = d["ang_rob"]
    med = float(np.nanmedian(ang))
    out = []
    for s, e in segs:
        if (e - s) / fps < mind:
            continue
        if med - float(np.nanmin(ang[s:e])) < MIN_AMP_DEG:
            continue
        out.append((s / fps, e / fps))
    return out


def merge(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    m = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return [(s, e) for s, e in m]


def run(name, fn):
    det = {m: fn(pack[m], proba[m]) for m in pack}
    ev = eval_ivs(gold_ivs, det)
    print(f"{name}: P={ev['P']} R={ev['R']} F1={ev['F1']} FP={ev['FP']} FN={ev['FN']} n={ev['n_det']}")
    for row in ev["per"]:
        print("   ", row)
    return ev


# RF only
run("RF30", lambda d, p: rf_ivs(d, p, 0.5, 0.6, 0.25))
# rule only
run("RULE", lambda d, p: rule_ivs(d))
# union
run("UNION", lambda d, p: merge(rule_ivs(d) + rf_ivs(d, p, 0.5, 0.6, 0.25)))
# rule + RF high-conf only (thr 0.65)
run("RULE+RF65", lambda d, p: merge(rule_ivs(d) + rf_ivs(d, p, 0.65, 0.6, 0.25)))
# RF primary + rule rescue when RF has no event near gold... can't use gold at test
# RF thr 0.45 higher recall
run("RF45", lambda d, p: rf_ivs(d, p, 0.45, 0.6, 0.2))
# vote: frame in event if rule OR rf@0.55, then segment
def vote(d, p, rthr=0.55):
    fps, n = d["meta"]["fps"], d["meta"]["n_frames"]
    t = np.arange(n) / fps
    rflag = np.zeros(n, bool)
    for s, e in rule_ivs(d):
        rflag[int(round(s * fps)):int(round(e * fps))] = True
    rf = (p >= rthr) & (t < WINDOW)
    pos = rflag | rf
    pos &= t < WINDOW
    if not pos.any():
        return []
    dpos = np.diff(pos.astype(int))
    starts = list(np.where(dpos == 1)[0] + 1)
    ends = list(np.where(dpos == -1)[0] + 1)
    if pos[0]:
        starts = [0] + starts
    if pos[-1]:
        ends = ends + [n]
    ang = d["ang_rob"]
    med = float(np.nanmedian(ang))
    out = []
    for s, e in zip(starts, ends):
        if (e - s) / fps < 0.15:
            continue
        if med - float(np.nanmin(ang[s:e])) < MIN_AMP_DEG:
            continue
        out.append((s / fps, e / fps))
    return merge(out)

run("VOTE r|rf0.55", vote)
run("VOTE r|rf0.45", lambda d, p: vote(d, p, 0.45))
