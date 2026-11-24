# -*- coding: utf-8 -*-
"""
事件级 RF：把 RF 帧级预测切分成事件，对人工金标准做事件级对比
================================================================

背景：金标准验证显示 RF 帧级 F1 0.64 显著优于规则法 0.39（定义差：RF 学到的是
人工的 episode 边界）。本脚本把 RF 逐帧预测切分成事件后做事件级匹配，回答
"RF 在事件/计数层面表现如何"——如果事件级也不差，则"计数用规则法、时长用 RF"
的混合式叙事成立；如果事件级差，则说明 RF 只适合做帧级分割。

切分规则：RF 正类连续段合并间隔 ≤ gap_s 的段，丢弃短于 min_dur_s 的段。

输出 → ../results/金标准验证/事件级_RF分段vs人工.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.features import build_features  # noqa: E402
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS, metrics, rf_model  # noqa: E402
from rf_baseline.validate_gold import (BASE, WINDOW, load_gold, match_events)  # noqa: E402

GAP_S = 0.5     # 相邻正类段间隔 ≤ 0.5s 视为同一事件
MIN_DUR = 0.15  # 事件最短持续（秒），与规则法一致


def segments_to_events(pos, fps):
    """布尔帧序列 → [(onset_s, offset_s)]，合并短间隔、丢弃过短段。"""
    d = np.diff(pos.astype(int))
    starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
    if pos[0]:
        starts = [0] + starts
    if pos[-1]:
        ends = ends + [len(pos)]
    segs = list(zip(starts, ends))
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) / fps <= GAP_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s / fps, e / fps) for s, e in merged if (e - s) / fps >= MIN_DUR]


def main():
    OUTD = os.path.normpath(os.path.join(BASE, "..", "results", "金标准验证"))
    gold = load_gold()
    X_all, y_all, mice_all = [], [], []
    meta_by_mouse, mask_by_mouse = {}, {}
    for mouse in gold.mouse.unique():
        _, csv_name, video_name = VIDEOS[str(mouse)]
        csv_path = os.path.join(VIDEO_DIR, csv_name)
        F, meta = build_features(csv_path, os.path.join(VIDEO_DIR, video_name))
        meta_by_mouse[mouse] = (F, meta)
        n = meta["n_frames"]
        fps = meta["fps"]
        t = np.arange(n) / fps
        g = gold[gold.mouse == mouse]
        pos = np.zeros(n, bool)
        band = np.zeros(n, bool)
        for r in g.itertuples():
            s, e = r.onset_s, r.offset_s + 1.0
            pos |= (t >= s) & (t < e)
            band |= (np.abs(t - s) < 0.5) | (np.abs(t - e) < 0.5)
        evaluable = (t < WINDOW) & ~band
        mask_by_mouse[mouse] = evaluable
        X_all.append(F.iloc[np.where(evaluable)[0]].to_numpy())
        y_all.append(pos[evaluable].astype(int))
        mice_all.append(np.full(int(evaluable.sum()), mouse))

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    mice = np.concatenate(mice_all)

    # 留一鼠预测帧级标签 → 按视频切事件
    pred_by_mouse = {}
    for m in np.unique(mice):
        clf = rf_model().fit(X[mice != m], y[mice != m])
        pred_by_mouse[m] = clf.predict(X[mice == m])

    rows = []
    for m in np.unique(mice):
        F, meta = meta_by_mouse[m]
        fps = meta["fps"]
        g = gold[gold.mouse == m]
        gold_ivs = [(r.onset_s, r.offset_s + 1.0) for r in g.itertuples()]
        full_pos = np.zeros(meta["n_frames"], bool)
        full_pos[mask_by_mouse[m]] = pred_by_mouse[m].astype(bool)
        rf_ivs = segments_to_events(full_pos, fps)
        _, tp, fn, fp = match_events(gold_ivs, rf_ivs)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"mouse": m, "人工事件数": len(gold_ivs), "RF事件数": len(rf_ivs),
                     "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(p, 3), "recall": round(r, 3),
                     "F1": round(2 * p * r / (p + r) if p + r else 0, 3)})

    ev = pd.read_csv(os.path.join(OUTD, "事件级_规则法vs人工.csv"))
    rf_ev = pd.DataFrame(rows)
    rf_ev["方法"] = "RF分段(留一鼠)"
    ev["方法"] = "规则法"
    cols = ["方法", "mouse", "人工事件数", "TP", "FP", "FN",
            "precision", "recall", "F1"]
    out = pd.concat([rf_ev, ev[ev.mouse != "汇总"][cols]], ignore_index=True)[cols]
    out.to_csv(os.path.join(OUTD, "事件级_RF分段vs人工.csv"),
               index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n已输出 → {os.path.join(OUTD, '事件级_RF分段vs人工.csv')}")


if __name__ == "__main__":
    main()
