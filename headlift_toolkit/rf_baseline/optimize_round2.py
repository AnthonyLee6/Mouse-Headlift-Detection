# -*- coding: utf-8 -*-
"""
算法第二轮优化：稳健俯仰角 + 30° 门控 + 事件后处理网格
======================================================

在 k=1.25 / min_amp_deg=30 已定的基础上继续压事件级 F1：
  1. 角度预处理：中值滤波 + 极端值 winsor / 相对中位基线
  2. 规则网格：k × neutral × min_dur × require_neutral（均带 30°）
  3. 后处理：短事件丢弃、长事件按谷拆分、近邻合并
  4. RF 概率：LOMO 全视频预测 → 事件化 → 与规则并/交/补

评估口径与 validate_gold.py 一致。
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import headlift as hl  # noqa: E402
from rf_baseline.features import build_features  # noqa: E402
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS, rf_model  # noqa: E402
from rf_baseline.validate_gold import BASE, WINDOW, load_gold, match_events  # noqa: E402

OUT = os.path.normpath(os.path.join(BASE, "..", "results", "融合优化"))
os.makedirs(OUT, exist_ok=True)
MIN_AMP_DEG = 30.0


def eval_ivs(gold_by_mouse, det_by_mouse):
    all_g, all_d = [], []
    per = []
    for m, g in gold_by_mouse.items():
        d = det_by_mouse.get(m, [])
        pairs, tp, fn, fp = match_events(g, d)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        on = [abs(ds - gs) for gs, ge, ds, de in pairs]
        per.append({
            "mouse": m, "TP": tp, "FP": fp, "FN": fn,
            "P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
            "onset_mae": round(float(np.mean(on)), 3) if on else np.nan,
        })
        all_g.extend(g)
        all_d.extend(d)
    _, tp, fn, fp = match_events(all_g, all_d)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    on = []
    pairs, _, _, _ = match_events(all_g, all_d)
    on = [abs(ds - gs) for gs, ge, ds, de in pairs]
    return {
        "n_gold": len(all_g), "n_det": len(all_d),
        "TP": tp, "FP": fp, "FN": fn,
        "P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
        "onset_mae": round(float(np.mean(on)), 3) if on else np.nan,
        "per": per,
    }


def robust_angle(df, smooth=5, clip_pct=2.0):
    """俯仰角 + winsor 极端值，减轻 DLC 跳点/角度回绕对 std 的污染。"""
    ang = hl.head_angle(df, pcutoff=0.6, smooth_win=smooth).to_numpy()
    lo = np.nanpercentile(ang, clip_pct)
    hi = np.nanpercentile(ang, 100 - clip_pct)
    return np.clip(ang, lo, hi)


def postprocess(ivs, fps, ang, min_dur_s=0.15, max_dur_s=8.0,
                split_gap_s=0.6, merge_gap_s=0.35):
    """事件后处理：时长过滤 + 过长按谷拆 + 近邻合并。"""
    if not ivs:
        return []
    med = float(np.nanmedian(ang))
    thr = med - 1.25 * float(np.nanstd(ang))
    out = []
    for s, e in ivs:
        i0, i1 = int(round(s * fps)), int(round(e * fps))
        i1 = min(i1, len(ang) - 1)
        if i1 <= i0:
            continue
        dur = (i1 - i0 + 1) / fps
        if dur < min_dur_s:
            continue
        if dur <= max_dur_s:
            out.append((i0 / fps, (i1 + 1) / fps))
            continue
        # 过长：在内部找高于 thr 的连续谷并拆分
        seg = ang[i0:i1 + 1]
        above = seg > thr
        cuts = []
        run = 0
        min_sep = max(1, int(split_gap_s * fps))
        for j in range(int(0.15 * len(seg)), int(0.85 * len(seg))):
            if above[j]:
                run += 1
                if run >= min_sep:
                    cuts.append(j - run // 2)
                    run = 0
            else:
                run = 0
        bounds = [i0] + [i0 + c for c in cuts] + [i1 + 1]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if (b - a) / fps >= min_dur_s:
                out.append((a / fps, b / fps))
    # merge
    out = sorted(out)
    merged = [list(out[0])]
    for s, e in out[1:]:
        if s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def rf_proba_all(pack):
    Xs, ys, ms = [], [], []
    meta = {}
    for mouse, d in pack.items():
        F, m, g = d["F"], d["meta"], d["gold"]
        n, fps = m["n_frames"], m["fps"]
        t = np.arange(n) / fps
        pos = np.zeros(n, bool)
        for r in g.itertuples():
            pos |= (t >= r.onset_s) & (t < r.offset_s + 1.0)
        win = t < WINDOW
        Xs.append(F.to_numpy()[win])
        ys.append(pos[win].astype(int))
        ms.append(np.full(int(win.sum()), mouse))
        meta[mouse] = F
    X, y, mice = np.vstack(Xs), np.concatenate(ys), np.concatenate(ms)
    proba = {}
    for m in np.unique(mice):
        clf = rf_model().fit(X[mice != m], y[mice != m])
        proba[m] = clf.predict_proba(meta[m].to_numpy())[:, 1]
    return proba


def main():
    gold = load_gold()
    pack = {}
    gold_ivs = {}
    for mouse in gold.mouse.unique():
        _, csv_name, video_name = VIDEOS[str(mouse)]
        F, meta = build_features(
            os.path.join(VIDEO_DIR, csv_name),
            os.path.join(VIDEO_DIR, video_name),
        )
        pack[mouse] = {"F": F, "meta": meta, "gold": gold[gold.mouse == mouse]}
        g = gold[gold.mouse == mouse]
        gold_ivs[mouse] = [(float(r.onset_s), float(r.offset_s) + 1.0) for r in g.itertuples()]
        print(f"[data] {mouse} n={meta['n_frames']} gold={len(gold_ivs[mouse])}")

    # 预计算稳健角
    for mouse, d in pack.items():
        _, csv_name, _ = VIDEOS[str(mouse)]
        df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv_name))
        d["ang_rob"] = robust_angle(df, smooth=5, clip_pct=2.0)
        d["ang_raw"] = d["meta"]["angle"]

    print("[rf] training LOMO proba...")
    proba = rf_proba_all(pack)

    rows = []

    def add(name, family, det_by_mouse):
        ev = eval_ivs(gold_ivs, det_by_mouse)
        rows.append({"method": name, "family": family,
                     **{k: v for k, v in ev.items() if k != "per"}})
        return ev

    # ---------- A. 稳健角 + 规则网格（30°） ----------
    for k, band, req, mind in itertools.product(
            [1.0, 1.15, 1.25, 1.4, 1.6],
            [5.0, 8.0, 10.0, 15.0],
            [True, False],
            [0.12, 0.15, 0.20]):
        det = {}
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            ev, _ = hl.detect_lifts(
                d["ang_rob"], fps, mode="dynamic", k=k, neutral_band=band,
                min_dur_s=mind, require_neutral=req, min_amp_deg=MIN_AMP_DEG,
            )
            ivs = []
            if len(ev):
                for _, r in ev.iterrows():
                    ivs.append((float(r.onset_s), float(r.offset_s) + 1.0 / fps))
            det[mouse] = [(s, e) for s, e in ivs if s < WINDOW]
        add(f"稳健角 k={k} band={band} req={req} min={mind}", "rule", det)

    # ---------- B. 后处理扫描（固定较好规则配置） ----------
    base_cfgs = [
        dict(k=1.25, band=10.0, req=True),
        dict(k=1.15, band=8.0, req=True),
        dict(k=1.0, band=10.0, req=True),
        dict(k=1.25, band=5.0, req=False),
    ]
    for cfg, mind, maxd, mgap in itertools.product(
            base_cfgs, [0.15], [4.0, 6.0, 10.0], [0.3, 0.5]):
        det = {}
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            ev, _ = hl.detect_lifts(
                d["ang_rob"], fps, mode="dynamic", k=cfg["k"],
                neutral_band=cfg["band"], min_dur_s=mind,
                require_neutral=cfg["req"], min_amp_deg=MIN_AMP_DEG,
            )
            ivs = []
            if len(ev):
                for _, r in ev.iterrows():
                    ivs.append((float(r.onset_s), float(r.offset_s) + 1.0 / fps))
            ivs = postprocess(ivs, fps, d["ang_rob"], min_dur_s=mind,
                              max_dur_s=maxd, merge_gap_s=mgap)
            det[mouse] = [(s, e) for s, e in ivs if s < WINDOW]
        add(f"后处理 {cfg} maxd={maxd} mgap={mgap}", "post", det)

    # ---------- C. RF 事件化（全视频概率，30° 峰值过滤） ----------
    for thr, gap, mind in itertools.product(
            [0.4, 0.5, 0.6], [0.4, 0.6, 0.9], [0.15, 0.25]):
        det = {}
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            n = d["meta"]["n_frames"]
            t = np.arange(n) / fps
            p = proba[mouse]
            pos = (p >= thr) & (t < WINDOW)
            # segments
            if not pos.any():
                det[mouse] = []
                continue
            df = np.diff(pos.astype(int))
            starts = list(np.where(df == 1)[0] + 1)
            ends = list(np.where(df == -1)[0] + 1)
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
            ivs = []
            for s, e in segs:
                if (e - s) / fps < mind:
                    continue
                peak = med - float(np.nanmin(ang[s:e]))
                if peak < MIN_AMP_DEG:
                    continue
                ivs.append((s / fps, e / fps))
            det[mouse] = ivs
        add(f"RF30 thr={thr} gap={gap} min={mind}", "rf", det)

    # ---------- D. 规则 ∪/∩ RF ----------
    for k, band, thr, mode in itertools.product(
            [1.15, 1.25], [8.0, 10.0], [0.45, 0.55], ["union", "add", "inter"]):
        det = {}
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            n = d["meta"]["n_frames"]
            t = np.arange(n) / fps
            ev, _ = hl.detect_lifts(
                d["ang_rob"], fps, mode="dynamic", k=k, neutral_band=band,
                min_dur_s=0.15, require_neutral=True, min_amp_deg=MIN_AMP_DEG,
            )
            rule = []
            if len(ev):
                for _, r in ev.iterrows():
                    s, e = float(r.onset_s), float(r.offset_s) + 1.0 / fps
                    if s < WINDOW:
                        rule.append((s, e))
            p = proba[mouse]
            rf_pos = (p >= thr) & (t < WINDOW)
            # rf segments simple
            ang = d["ang_rob"]
            med = float(np.nanmedian(ang))
            dpos = np.diff(rf_pos.astype(int))
            starts = list(np.where(dpos == 1)[0] + 1)
            ends = list(np.where(dpos == -1)[0] + 1)
            if rf_pos[0]:
                starts = [0] + starts
            if rf_pos[-1]:
                ends = ends + [n]
            rf_ivs = []
            for s, e in zip(starts, ends):
                if (e - s) / fps < 0.2:
                    continue
                if med - float(np.nanmin(ang[s:e])) < MIN_AMP_DEG:
                    continue
                rf_ivs.append((s / fps, e / fps))
            if mode == "union":
                ivs = sorted(rule + rf_ivs)
                if not ivs:
                    det[mouse] = []
                    continue
                mrg = [list(ivs[0])]
                for s, e in ivs[1:]:
                    if s <= mrg[-1][1]:
                        mrg[-1][1] = max(mrg[-1][1], e)
                    else:
                        mrg.append([s, e])
                det[mouse] = [(s, e) for s, e in mrg]
            elif mode == "add":
                out = list(rule)
                for s, e in rf_ivs:
                    ov = max((max(0.0, min(e, b) - max(s, a)) for a, b in rule), default=0.0)
                    if ov < 0.15:
                        out.append((s, e))
                det[mouse] = sorted(out)
            else:  # inter: keep rule only if overlaps RF
                det[mouse] = [
                    (s, e) for s, e in rule
                    if any(min(e, b) - max(s, a) > 0.1 for a, b in rf_ivs)
                ]
        add(f"混合{k}/{band}/thr{thr}/{mode}", "mix", det)

    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    path = os.path.join(OUT, "第二轮优化_汇总.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("\n========== TOP 20 ==========")
    cols = ["method", "family", "P", "R", "F1", "TP", "FP", "FN", "onset_mae", "n_det"]
    print(df[cols].head(20).to_string(index=False))
    print("\nfamily max:")
    print(df.groupby("family").F1.max())
    print("->", path)

    # 最优逐鼠
    top = df.iloc[0]
    print("\nTOP method:", top.method, "F1", top.F1)


if __name__ == "__main__":
    main()
