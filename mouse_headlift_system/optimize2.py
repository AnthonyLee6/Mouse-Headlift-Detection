# -*- coding: utf-8 -*-
"""
金标准驱动的检测优化 · 第二轮：滚动局部基线
============================================

针对诊断图暴露的问题：
  1. 姿态变化导致仰角基线漂移（-61° → -80°），全局中位数阈值失效；
  2. 人工标注的抬头事件峰值差异巨大（局部基线以上 +15° ~ +60°），
     单一绝对阈值无法同时覆盖 → 漏检与误检并存。

方案：拖尾滚动中位数作为局部基线（跟随漂移、对事件鲁棒），
      相对抬升量 e = pitch − base，阈值 = k·MAD(e)（全局稳健尺度），
      迟滞状态机（进入 e≥thr 持续 min_dur，退出 e≤exit_frac·thr），
      相邻事件 gap ≤ merge_gap 合并。
变体：单通道（pitch 仰角 / nose_h 鼻尖高度）与双通道（and=同时超阈、or=任一超阈）。

用法：python optimize2.py            # 全量网格扫描
      python optimize2.py --best    # 只输出最优配置的逐鼠明细与诊断图
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

import headlift_core as hl
from evaluate_gt import (CSV_FILES, EVAL_WINDOW_S, GROUND_TRUTH, VIDEO_DIR,
                         event_metrics, frame_metrics, gt_frames)

BASE = os.path.dirname(os.path.abspath(__file__))
MAD_K = 1.4826


def feat_series(df, feat, pcutoff=0.6, smooth=5):
    """仰角 / 鼻尖高度特征（含置信度插值与中值滤波）。"""
    if feat == "pitch":
        return hl.head_pitch(df, pcutoff=pcutoff, smooth_win=smooth)
    if feat == "nose_h":
        return hl.nose_height(df, pcutoff=pcutoff, smooth_win=smooth)
    raise ValueError(feat)


def detect_local(x, fps, win_s=10.0, k=3.0, min_dur_s=0.3,
                 exit_frac=0.4, merge_gap_s=0.5, grow_frac=0.25):
    """滚动局部基线抬头检测。返回 (事件表, 逐帧布尔数组, 抬升量序列)。

    事件定义：核心 = e ≥ k·MAD 持续 ≥ min_dur；随后向两侧生长到
    e < grow_frac·thr 为止（对齐人工"整段动作"的标注习惯）。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    win = max(7, int(win_s * fps))
    s = pd.Series(x)
    base = s.rolling(win, min_periods=max(5, win // 4)).median().bfill()
    e = (s - base).to_numpy()
    med_e = np.nanmedian(e)
    mad = MAD_K * float(np.nanmedian(np.abs(e - med_e)))
    thr = k * mad if mad > 1e-6 else np.inf
    min_frames = max(1, int(round(min_dur_s * fps)))
    exit_thr = exit_frac * thr
    grow_thr = grow_frac * thr

    events, flags = [], np.zeros(n, dtype=bool)
    hold, start = 0, 0
    in_lift = False
    for i, v in enumerate(e):
        if np.isnan(v):
            if in_lift and hold >= min_frames:
                events.append((start, i - 1))
            hold, in_lift = 0, False
            continue
        if not in_lift and v >= thr:
            hold += 1
            if hold >= min_frames:
                in_lift, start = True, i - hold + 1
        elif in_lift and v >= exit_thr:
            hold = max(hold, 1)  # 迟滞区内维持事件
        else:
            if in_lift:
                events.append((start, i - 1))
            hold, in_lift = 0, False
    if in_lift:
        events.append((start, n - 1))

    # 合并间隔 ≤ merge_gap 的事件
    if merge_gap_s and len(events) > 1:
        gap = int(round(merge_gap_s * fps))
        merged = [events[0]]
        for s0, e0 in events[1:]:
            if s0 - merged[-1][1] <= gap:
                merged[-1] = (merged[-1][0], e0)
            else:
                merged.append((s0, e0))
        events = merged

    # 边界生长：核心之外，抬升量仍 ≥ grow_thr 的部分并入事件
    if grow_frac is not None:
        grown = []
        for s0, e0 in events:
            while s0 > 0 and e[s0 - 1] >= grow_thr:
                s0 -= 1
            while e0 < n - 1 and e[e0 + 1] >= grow_thr:
                e0 += 1
            grown.append((s0, e0))
        events = grown

    for s0, e0 in events:
        flags[s0:e0 + 1] = True
    ev = pd.DataFrame(events, columns=["onset_frame", "offset_frame"])
    if len(ev):
        ev["onset_s"] = (ev["onset_frame"] / fps).round(2)
        ev["offset_s"] = (ev["offset_frame"] / fps).round(2)
        ev["duration_s"] = ((ev["offset_frame"] - ev["onset_frame"] + 1) / fps).round(2)
    return ev, flags, e


def run_channels(data, cfg):
    """对 4 段视频计算单/双通道检测结果。cfg: feat/win/k/mind/exit/gap/smooth。"""
    feat = cfg["feat"]
    out = {}
    for mouse, (df, fps, n) in data.items():
        p = feat_series(df, "pitch", smooth=cfg["smooth"]).to_numpy()
        h = feat_series(df, "nose_h", smooth=cfg["smooth"]).to_numpy()
        ev_p, fl_p, e_p = detect_local(p, fps, win_s=cfg["win"], k=cfg["k"],
                                       min_dur_s=cfg["mind"],
                                       exit_frac=cfg["exit"], merge_gap_s=cfg["gap"])
        if feat == "pitch":
            out[mouse] = (ev_p, fl_p, fps, n)
            continue
        ev_h, fl_h, e_h = detect_local(h, fps, win_s=cfg["win"], k=cfg["k"],
                                       min_dur_s=cfg["mind"],
                                       exit_frac=cfg["exit"], merge_gap_s=cfg["gap"])
        if feat == "nose_h":
            out[mouse] = (ev_h, fl_h, fps, n)
        elif feat == "and":
            flags = fl_p & fl_h
            out[mouse] = (_flags_to_events(flags), flags, fps, n)
        elif feat == "or":
            flags = fl_p | fl_h
            out[mouse] = (_flags_to_events(flags), flags, fps, n)
        else:
            raise ValueError(feat)
    return out


def _flags_to_events(flags):
    idx = np.flatnonzero(np.diff(np.r_[0, flags.astype(int), 0]))
    ev = pd.DataFrame({"onset_frame": idx[::2], "offset_frame": idx[1::2] - 1})
    if len(ev):
        ev["onset_s"] = ev["onset_frame"].round(2)
        ev["offset_s"] = ev["offset_frame"].round(2)
        ev["duration_s"] = (ev["offset_frame"] - ev["onset_frame"] + 1).round(2)
    return ev


def evaluate_cfg(data, res, detail=False):
    fm_all, ev_all, rows = [], [], []
    for mouse, (ev, flags, fps, n) in res.items():
        w = int(EVAL_WINDOW_S * fps)
        gt = gt_frames(mouse, fps, n)[:w]
        pred = flags[:w]
        fm = frame_metrics(gt, pred)
        fm_all.append(fm)
        em, _, _ = event_metrics(GROUND_TRUTH[mouse], ev, fps)
        ev_all.append(em)
        if detail:
            fm.update({"小鼠": mouse, "人工事件数": len(GROUND_TRUTH[mouse]),
                       "检出事件数": len(ev)})
            rows.append(fm)
    avg = {k: round(float(np.mean([r[k] for r in fm_all])), 3) for k in fm_all[0]}
    avg_ev = {k: round(float(np.mean([r[k] for r in ev_all])), 3) for k in ev_all[0]}
    return {**avg, **avg_ev}, rows


def load_data():
    data = {}
    for mouse, csv in CSV_FILES.items():
        df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv))
        fps = hl.read_fps(os.path.join(
            VIDEO_DIR, csv.split("DLC_resnet50")[0].strip() + ".WMV"))
        data[mouse] = (df, fps, len(df))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", action="store_true")
    args = ap.parse_args()
    data = load_data()
    print(f"评估窗口: 前 {EVAL_WINDOW_S:.0f}s | 人工事件总数:",
          sum(len(v) for v in GROUND_TRUTH.values()))

    if args.best:
        cfg = dict(feat="and", win=10, k=2.0, mind=0.3, exit=0.4, gap=0.5, smooth=5)
        res = run_channels(data, cfg)
        m, rows = evaluate_cfg(data, res, detail=True)
        print(pd.DataFrame(rows).to_string(index=False))
        print({k: v for k, v in m.items()})
        return

    rows = []
    grid = list(itertools.product(
        ["pitch", "nose_h", "and", "or"],   # 通道
        [6, 10, 15, 20],                    # 局部基线窗口(s)
        [1.5, 2.0, 2.5, 3.0, 3.5],          # k·MAD
        [0.2, 0.3, 0.4, 0.5],               # 最短持续
        [0.3, 0.5]))                        # 迟滞退出
    print("网格规模:", len(grid))
    for feat, win, k, mind, exitf in grid:
        cfg = dict(feat=feat, win=win, k=k, mind=mind, exit=exitf,
                   gap=0.5, smooth=5)
        res = run_channels(data, cfg)
        m, _ = evaluate_cfg(data, res)
        rows.append({"通道": feat, "win": win, "k": k, "mind": mind,
                     "exit": exitf, **m})
    df = pd.DataFrame(rows).sort_values("事件F1", ascending=False)
    print(df.head(20).to_string(index=False))
    out = os.path.join(BASE, "金标准评估_局部基线扫描.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("已保存 →", out)


if __name__ == "__main__":
    main()
