# -*- coding: utf-8 -*-
"""
金标准驱动的检测优化实验
========================

目的：找出"特征 × 自适应阈值 × 时序参数 × 后处理"的最优组合。

特征候选（均以视频内中位数为基线，正值 = 抬头方向）：
  pitch     鼻尖相对头顶仰角（arcsin 竖直分量，与朝向无关）
  nose_h    鼻尖绝对高度变化（基线 - nose_y，抬头时鼻尖整体上移）
  dy        头顶-鼻子竖直距离变化（head_y - nose_y）

自适应阈值：enter = 基线 + k·MAD（MAD 对占比 <10% 的抬头事件稳健）；
后处理：相邻检出间隔 ≤ merge_gap 视为同一次抬头（合并分裂事件）。

评估：前 120s，帧级 P/R/F1/κ + 事件级 P/R/F1（中心±1.5s 匹配）。
"""
import itertools
import os

import numpy as np
import pandas as pd

import headlift_core as hl
from evaluate_gt import (CSV_FILES, EVAL_WINDOW_S, GROUND_TRUTH, VIDEO_DIR,
                         event_metrics, frame_metrics, gt_frames)

MAD_K = 1.4826  # MAD → σ 等价系数


def feature_series(df, feat, pcutoff=0.6, smooth=5):
    nose_y, head_y = df["nose_y"], df["head_y"]
    ok = (df["nose_lh"] >= pcutoff) & (df["head_lh"] >= pcutoff)
    if feat == "pitch":
        return hl.head_pitch(df, pcutoff=pcutoff, smooth_win=smooth)
    if feat == "nose_h":
        v = -(nose_y - nose_y.median())
    elif feat == "dy":
        v = head_y - nose_y
    elif feat == "head_h":
        v = -(head_y - head_y.median())
    else:
        raise ValueError(feat)
    v = v.where(ok)
    v = v.interpolate(limit_direction="both")
    if smooth and smooth > 1:
        v = v.rolling(smooth, center=True, min_periods=1).median()
    return v


def detect_adaptive(x, fps, k=3.0, min_dur_s=0.4, merge_gap_s=0.5):
    """抬升期状态机：事件 = 连续高于退出阈值的区段，峰值触及进入阈值即确认。

    事件跨度覆盖整个"抬头可见期"（人工标注习惯），而非只有超严格阈值的峰值。
    """
    x = np.asarray(x, dtype=float)
    base = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - base))) * MAD_K
    enter = base + max(k * mad, 12.0)
    exit_to = base + max(0.35 * k * mad, 5.0)
    min_frames = max(1, int(round(min_dur_s * fps)))

    elevated = np.where(np.isnan(x), False, x >= exit_to)
    runs, s = [], None
    for i, e in enumerate(elevated):
        if e and s is None:
            s = i
        elif not e and s is not None:
            runs.append((s, i - 1)); s = None
    if s is not None:
        runs.append((s, len(elevated) - 1))

    events = [(s, e) for s, e in runs
              if (e - s + 1) >= min_frames
              and float(np.nanmax(x[s:e + 1])) >= enter]

    gap_f = int(round(merge_gap_s * fps))
    merged = []
    for st, e in events:
        if merged and st - merged[-1][1] <= gap_f:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((st, e))

    flags = np.zeros(len(x), dtype=bool)
    for st, e in merged:
        flags[st:e + 1] = True
    ev = pd.DataFrame(merged, columns=["onset_frame", "offset_frame"])
    if len(ev):
        ev["onset_s"] = (ev["onset_frame"] / fps).round(2)
        ev["offset_s"] = (ev["offset_frame"] / fps).round(2)
        ev["duration_s"] = ((ev["offset_frame"] - ev["onset_frame"] + 1) / fps).round(2)
        ev["max_lift_deg"] = [round(float(np.nanmax(x[st:e + 1])) - base, 1)
                              for st, e in merged]
    return ev, flags


def load_all():
    data = {}
    for mouse, csv in CSV_FILES.items():
        df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv))
        fps = hl.read_fps(os.path.join(
            VIDEO_DIR, csv.split("DLC_resnet50")[0].strip() + ".WMV"))
        data[mouse] = (df, fps, len(df))
    return data


def evaluate_cfg(data, feat, k, min_dur, merge_gap, smooth):
    fm_all, ev_all = [], []
    per_mouse = {}
    for mouse, (df, fps, n) in data.items():
        x = feature_series(df, feat, smooth=smooth)
        ev, flags = detect_adaptive(x, fps, k=k, min_dur_s=min_dur,
                                    merge_gap_s=merge_gap)
        w = int(EVAL_WINDOW_S * fps)
        gt = gt_frames(mouse, fps, n)[:w]
        fm = frame_metrics(gt, flags[:w])
        em, _, _ = event_metrics(GROUND_TRUTH[mouse], ev, fps)
        fm_all.append(fm)
        ev_all.append(em)
        per_mouse[mouse] = (len(GROUND_TRUTH[mouse]), len(ev))
    avg = {kk: round(float(np.mean([r[kk] for r in fm_all])), 3) for kk in fm_all[0]}
    evg = {kk: round(float(np.mean([r[kk] for r in ev_all])), 3) for kk in ev_all[0]}
    return {**avg, **evg}, per_mouse


def main():
    data = load_all()
    rows = []
    best = None
    for feat, k, mind, gap, sm in itertools.product(
            ["pitch", "nose_h", "dy"], [3.0, 4.0, 5.0, 6.0],
            [0.3, 0.4, 0.5], [0.3, 0.5, 0.8], [3, 5]):
        m, per_mouse = evaluate_cfg(data, feat, k, mind, gap, sm)
        row = {"特征": feat, "k": k, "min_dur": mind, "merge": gap,
               "smooth": sm, **m}
        rows.append(row)
        key = (m["事件F1"], m["F1"])
        if best is None or key > best[0]:
            best = (key, row.copy(), per_mouse)
    df = pd.DataFrame(rows).sort_values(["事件F1", "F1"], ascending=False)
    print(df.head(20).to_string(index=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "金标准评估_优化扫描.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("已保存 →", out)
    print("\n最优:", best[1])
    print("各鼠 人工/检出:", best[2])


if __name__ == "__main__":
    main()


def detect_combined(df, fps, k_nh=3.0, k_p=2.0, min_dur_s=0.4,
                    merge_gap_s=0.5, smooth=5, pcutoff=0.6):
    """双通道组合检测（人工语义对齐）：
    通道1 nose_h：身体/鼻尖整体抬升的连续区段 → 候选事件（高召回）
    通道2 pitch ：区段内鼻尖仰角须抬过自身基线 + k_p·MAD → 确认"头抬起来看"
    （过滤"身体抬起但低头嗅边"的假阳性）。事件跨度 = nose_h 抬升区段。
    """
    nose_h = feature_series(df, "nose_h", pcutoff=pcutoff, smooth=smooth)
    pitch = hl.head_pitch(df, pcutoff=pcutoff, smooth_win=smooth)
    nh = np.asarray(nose_h, dtype=float)
    p = np.asarray(pitch, dtype=float)

    base_nh = float(np.nanmedian(nh))
    mad_nh = float(np.nanmedian(np.abs(nh - base_nh))) * MAD_K
    enter_nh = base_nh + max(k_nh * mad_nh, 10.0)
    exit_nh = base_nh + max(0.3 * k_nh * mad_nh, 4.0)
    base_p = float(np.nanmedian(p))
    mad_p = float(np.nanmedian(np.abs(p - base_p))) * MAD_K
    enter_p = base_p + max(k_p * mad_p, 8.0)
    min_frames = max(1, int(round(min_dur_s * fps)))

    elevated = np.where(np.isnan(nh), False, nh >= exit_nh)
    runs, s = [], None
    for i, e in enumerate(elevated):
        if e and s is None:
            s = i
        elif not e and s is not None:
            runs.append((s, i - 1)); s = None
    if s is not None:
        runs.append((s, len(elevated) - 1))

    events = []
    for st, e in runs:
        if float(np.nanmax(p[st:e + 1])) >= enter_p:  # 区段内头确实抬过
            events.append((st, e))
    events = [(s, e) for s, e in events if e - s + 1 >= min_frames]

    gap_f = int(round(merge_gap_s * fps))
    merged = []
    for st, e in events:
        if merged and st - merged[-1][1] <= gap_f:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((st, e))
    merged = [(s, e) for s, e in merged if e - s + 1 >= min_frames]

    flags = np.zeros(len(nh), dtype=bool)
    for st, e in merged:
        flags[st:e + 1] = True
    ev = pd.DataFrame(merged, columns=["onset_frame", "offset_frame"])
    if len(ev):
        ev["onset_s"] = (ev["onset_frame"] / fps).round(2)
        ev["offset_s"] = (ev["offset_frame"] / fps).round(2)
        ev["duration_s"] = ((ev["offset_frame"] - ev["onset_frame"] + 1) / fps).round(2)
    return ev, flags


def evaluate_combined(data, **kw):
    fm_all, ev_all, per_mouse = [], [], {}
    for mouse, (df, fps, n) in data.items():
        ev, flags = detect_combined(df, fps, **kw)
        w = int(EVAL_WINDOW_S * fps)
        gt = gt_frames(mouse, fps, n)[:w]
        fm = frame_metrics(gt, flags[:w])
        em, _, _ = event_metrics(GROUND_TRUTH[mouse], ev, fps)
        fm_all.append(fm); ev_all.append(em)
        per_mouse[mouse] = (len(GROUND_TRUTH[mouse]), len(ev))
    avg = {kk: round(float(np.mean([r[kk] for r in fm_all])), 3) for kk in fm_all[0]}
    evg = {kk: round(float(np.mean([r[kk] for r in ev_all])), 3) for kk in ev_all[0]}
    return {**avg, **evg}, per_mouse
