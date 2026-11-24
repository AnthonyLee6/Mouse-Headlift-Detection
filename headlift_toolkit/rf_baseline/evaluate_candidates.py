# -*- coding: utf-8 -*-
"""
生产候选配置评估：逐鼠明细 + 更稳的融合
========================================

在 optimize_fusion.py 基础上，对少数候选做逐鼠评估，并实现
“规则法骨干 + 高置信 RF 独有段”的保守融合（控制 FP）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.optimize_fusion import (  # noqa: E402
    OUT,
    eval_events,
    fusion_add_rf_only,
    gold_ivs_for,
    load_all,
    merge_intervals,
    rf_only_events,
    rule_events,
    train_rf_proba,
    interval_overlap,
)


def conservative_fusion(rule_ivs, rf_ivs, proba, fps, min_peak=0.7,
                        min_dur_s=0.25, min_overlap_s=0.2) -> list[tuple[float, float]]:
    """只把“高峰值概率 + 与规则不重叠”的 RF 段加进来。"""
    p = np.asarray(proba)
    out = list(rule_ivs)
    for s, e in rf_ivs:
        i0 = max(0, int(round(s * fps)))
        i1 = min(len(p) - 1, int(round(e * fps)))
        if i1 <= i0:
            continue
        if float(np.max(p[i0:i1 + 1])) < min_peak:
            continue
        if (e - s) < min_dur_s:
            continue
        if rule_ivs and max(interval_overlap((s, e), iv) for iv in rule_ivs) >= min_overlap_s:
            continue
        out.append((s, e))
    return merge_intervals(out)


def score_fusion(rule_ivs, proba, fps, ang, thr_score=0.45, min_dur_s=0.2,
                 amp_frac=0.15) -> list[tuple[float, float]]:
    """逐帧分数 = 0.55*规则标志 + 0.45*RF概率，再阈值；规则事件强制保留。

    同时要求新增段满足最小抬升幅度（相对中位数）。
    """
    p = np.asarray(proba, dtype=float)
    n = len(p)
    med = float(np.nanmedian(ang))
    p05 = float(np.nanpercentile(ang, 5))
    denom = max(med - p05, 1e-6)

    rule_flag = np.zeros(n, bool)
    for s, e in rule_ivs:
        rule_flag[int(round(s * fps)):int(round(e * fps)) + 1] = True

    score = 0.55 * rule_flag.astype(float) + 0.45 * p
    cand = score >= thr_score
    # 新增帧还须有物理幅度
    lift_amp = (med - np.asarray(ang)) / denom
    cand = cand & ((rule_flag) | (lift_amp >= amp_frac))
    rf_ivs = []
    # 用 cand 切事件
    d = np.diff(cand.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if cand[0]:
        starts = [0] + starts
    if cand[-1]:
        ends = ends + [n]
    segs = []
    for s, e in zip(starts, ends):
        if (e - s) / fps >= min_dur_s:
            segs.append((s / fps, e / fps))
    # 规则事件并入，合并重叠
    return merge_intervals(list(rule_ivs) + segs)


def per_mouse_eval(pack, proba, detectors: dict) -> pd.DataFrame:
    rows = []
    for name, fn in detectors.items():
        all_g, all_d = [], []
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            ivs = [(s, e) for s, e in fn(mouse, d) if s < 120.0]
            g = gold_ivs_for(d["gold"])
            m = eval_events(g, ivs)
            m.update({"method": name, "mouse": mouse})
            rows.append(m)
            all_g.extend(g)
            all_d.extend(ivs)
        ov = eval_events(all_g, all_d)
        ov.update({"method": name, "mouse": "汇总"})
        rows.append(ov)
    return pd.DataFrame(rows)


def main():
    pack = load_all()
    proba = train_rf_proba(pack)

    detectors = {
        "基线规则 k=1.5": lambda m, d: rule_events(d["meta"], k=1.5, require_neutral=True,
                                                   neutral_band=10.0),
        "推荐规则 k=1.25": lambda m, d: rule_events(d["meta"], k=1.25, require_neutral=True,
                                                    neutral_band=10.0),
        "RF-only thr0.5 gap0.8 min0.25": lambda m, d: rf_only_events(
            proba[m], d["meta"]["fps"], thr=0.5, gap_s=0.8, min_dur_s=0.25,
            window=np.arange(d["meta"]["n_frames"]) / d["meta"]["fps"] < 120.0),
        "RF-only thr0.6 gap0.8 min0.25": lambda m, d: rf_only_events(
            proba[m], d["meta"]["fps"], thr=0.6, gap_s=0.8, min_dur_s=0.25,
            window=np.arange(d["meta"]["n_frames"]) / d["meta"]["fps"] < 120.0),
        "保守融合 k1.25+RF峰0.75": lambda m, d: conservative_fusion(
            rule_events(d["meta"], k=1.25, require_neutral=True, neutral_band=10.0),
            rf_only_events(proba[m], d["meta"]["fps"], thr=0.55, gap_s=0.8,
                           min_dur_s=0.25,
                           window=np.arange(d["meta"]["n_frames"]) / d["meta"]["fps"] < 120.0),
            proba[m], d["meta"]["fps"], min_peak=0.75, min_dur_s=0.25,
            min_overlap_s=0.25),
        "保守融合 k1.25+RF峰0.85": lambda m, d: conservative_fusion(
            rule_events(d["meta"], k=1.25, require_neutral=True, neutral_band=10.0),
            rf_only_events(proba[m], d["meta"]["fps"], thr=0.6, gap_s=0.8,
                           min_dur_s=0.3,
                           window=np.arange(d["meta"]["n_frames"]) / d["meta"]["fps"] < 120.0),
            proba[m], d["meta"]["fps"], min_peak=0.85, min_dur_s=0.3,
            min_overlap_s=0.25),
        "分数融合 score0.55": lambda m, d: score_fusion(
            rule_events(d["meta"], k=1.25, require_neutral=True, neutral_band=10.0),
            proba[m], d["meta"]["fps"], d["meta"]["angle"], thr_score=0.55),
        "分数融合 score0.60": lambda m, d: score_fusion(
            rule_events(d["meta"], k=1.25, require_neutral=True, neutral_band=10.0),
            proba[m], d["meta"]["fps"], d["meta"]["angle"], thr_score=0.60),
    }

    df = per_mouse_eval(pack, proba, detectors)
    path = os.path.join(OUT, "生产候选_逐鼠对比.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")

    summ = df[df.mouse == "汇总"].sort_values("F1", ascending=False)
    print("\n========== 汇总 ==========")
    cols = ["method", "precision", "recall", "F1", "TP", "FP", "FN", "onset_mae_s"]
    print(summ[cols].to_string(index=False))

    print("\n========== 逐鼠 F1 ==========")
    piv = df.pivot(index="method", columns="mouse", values="F1")
    # 只保留真实小鼠列
    mouse_cols = [c for c in piv.columns if c != "汇总"]
    piv = piv[mouse_cols + ["汇总"]] if "汇总" in piv.columns else piv[mouse_cols]
    piv["min鼠"] = piv[mouse_cols].min(axis=1)
    piv["std"] = piv[mouse_cols].std(axis=1).round(3)
    print(piv.sort_values("汇总", ascending=False).to_string())
    piv.to_csv(os.path.join(OUT, "生产候选_逐鼠F1矩阵.csv"), encoding="utf-8-sig")
    print(f"\n已输出 → {path}")


if __name__ == "__main__":
    main()
