# -*- coding: utf-8 -*-
"""
特征工程（MARS / SimBA 风格的可解释姿态特征）
=============================================

从 DeepLabCut 关键点 CSV 构建逐帧特征矩阵，供随机森林基线使用。
设计原则：
  1. 全部特征可解释（角度/高度/速度/置信度），对应 SimBA "explainable ML" 的定位；
  2. 预处理与规则法一致（pcutoff=0.6 置信度置 NaN 后线性插值），保证 RF 与规则法
     在同一份数据上比较；
  3. 所有滑窗统计在"单段视频内部"计算，不跨视频泄漏。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import headlift as hl  # noqa: E402

PCUTOFF = 0.6
FRAME_H_DEFAULT = 288  # config.yaml video_sets crop: 0,704,0,288
DEFAULT_K = 1.25  # 金标准校准后的生产推荐动态阈值系数


def _interp(arr, ok):
    """置信度不足的帧置 NaN 后线性插值（与 headlift.head_angle 同一预处理）。"""
    out = np.asarray(arr, dtype=float).copy()
    out[~ok] = np.nan
    return pd.Series(out).interpolate(limit_direction="both").to_numpy()


def _roll(s, win, stat):
    r = pd.Series(s).rolling(win, center=True, min_periods=1)
    return (r.mean() if stat == "mean" else r.std()).to_numpy()


def build_features(csv_path, video_path=None, frame_h=FRAME_H_DEFAULT):
    """返回 (特征 DataFrame, 关键中间量 dict)。

    关键中间量供抽样/绘图复用：angle(平滑后主信号)、angle_raw、thr(动态阈值)、
    med(中位数)、fps、rule_flags(规则法逐帧判定)。
    """
    df = hl.load_dlc_csv(csv_path)
    fps = hl.read_fps(video_path) if video_path else 20.0
    n = len(df)
    H = frame_h

    ok_nose = df["nose_lh"].to_numpy() >= PCUTOFF
    ok_head = df["head_lh"].to_numpy() >= PCUTOFF
    ok_le = df["left_ear_lh"].to_numpy() >= PCUTOFF
    ok_re = df["right_ear_lh"].to_numpy() >= PCUTOFF

    # ---- 核心信号：俯仰角（头顶→鼻子，抬头时下降；与规则法完全一致） ----
    ang_raw = hl.head_angle(df, pcutoff=PCUTOFF, smooth_win=0).to_numpy()
    ang = hl.head_angle(df, pcutoff=PCUTOFF, smooth_win=5).to_numpy()
    med = float(np.nanmedian(ang))
    thr = med - DEFAULT_K * float(np.nanstd(ang))

    # ---- 坐标类特征（低置信度插值） ----
    nose_x = _interp(df["nose_x"].to_numpy(), ok_nose)
    nose_y = _interp(df["nose_y"].to_numpy(), ok_nose)
    head_x = _interp(df["head_x"].to_numpy(), ok_head)
    head_y = _interp(df["head_y"].to_numpy(), ok_head)
    lex = _interp(df["left_ear_x"].to_numpy(), ok_le)
    ley = _interp(df["left_ear_y"].to_numpy(), ok_le)
    rex = _interp(df["right_ear_x"].to_numpy(), ok_re)
    rey = _interp(df["right_ear_y"].to_numpy(), ok_re)

    # ---- 特征表 ----
    F = pd.DataFrame(index=df["frame"].astype(int))
    F["ang"] = ang                                  # 主信号（平滑）
    F["ang_rel"] = ang - med                        # 相对视频中位数（自适应基线）
    F["ang_raw"] = ang_raw                          # 未平滑原始角
    F["vel_deg_s"] = np.gradient(np.nan_to_num(ang, nan=med)) * fps  # 角速度

    for w in (5, 15, 31):
        F[f"ang_mean{w}"] = _roll(ang, w, "mean")
        F[f"ang_std{w}"] = _roll(ang, w, "std")
    F["vel_mean5"] = _roll(F["vel_deg_s"].to_numpy(), 5, "mean")
    F["vel_std5"] = _roll(F["vel_deg_s"].to_numpy(), 5, "std")

    # 高度类：图像 y 向下为正 → 抬头时 nose_y 变小
    F["nose_y_rel"] = (nose_y - np.nanmedian(nose_y)) / H
    F["head_y_rel"] = (head_y - np.nanmedian(head_y)) / H
    F["dy_nose_head"] = (head_y - nose_y) / H       # 正 = 头顶低于鼻子（y 规则信号）
    F["dist_nose_head"] = np.hypot(head_x - nose_x, head_y - nose_y) / H

    # 耳朵类：滚转角 + 双耳中点高度 + 耳距（辅助区分抬头 vs 侧身/歪头）
    F["ear_roll_deg"] = np.degrees(np.arctan2(ley - rey, lex - rex))
    F["ear_mid_y_rel"] = ((ley + rey) / 2 - np.nanmedian((ley + rey) / 2)) / H
    F["ear_dist"] = np.hypot(lex - rex, ley - rey) / H

    # 置信度类：让模型学会"不信任低质量帧"
    F["lh_min"] = df[["nose_lh", "head_lh", "left_ear_lh", "right_ear_lh"]].min(axis=1)
    F["lh_nose"] = df["nose_lh"]
    F["lh_head"] = df["head_lh"]

    rule_ev, rule_flags = hl.detect_lifts(ang, fps, mode="dynamic", k=DEFAULT_K)
    meta = {
        "fps": fps,
        "med": med,
        "thr": thr,
        "angle": ang,
        "angle_raw": ang_raw,
        "rule_flags": rule_flags,
        "rule_events": rule_ev,
        "n_frames": n,
    }
    return F, meta
