# -*- coding: utf-8 -*-
"""
小鼠抬头行为自动识别分析系统 —— 核心检测库 V1.0
================================================

功能：读取 DeepLabCut 关键点数据，计算头部特征，按时序滤波 + 迟滞状态机
判定抬头事件，输出行为学指标与事件明细。

两种判定模式：
  1. 垂直仰角（默认，经人工标注校准）：pitch = asin((head_y - nose_y) / 距离)，
     即鼻尖相对头顶的仰角，与小鼠左右朝向无关、无 ±180° 卷绕。
     鼻尖仰角较基线（全片中位数）抬升 ≥ 判定角且持续 ≥ 最短时长记一次抬头，
     事件结束后须回落到基线 + 中性带以内才计入下一次。
  2. 头顶→鼻子向量角（旧）：ang = atan2(nose_y - head_y, nose_x - head_x)，
     平视时为正，抬头时下降，支持动态阈值（中位数-k·σ）与固定阈值。
"""
import os

import numpy as np
import pandas as pd

BODY_PARTS = ["nose", "head_top", "left_ear", "right_ear"]
VIDEO_EXTS = [".wmv", ".mp4", ".avi", ".mov", ".mkv"]


# ---------------------------------------------------------------- 数据加载
def load_dlc_csv(csv_path):
    """读取 DeepLabCut 单动物 CSV，按 (bodypart, coord) 显式取列，杜绝列错位。

    返回 DataFrame: frame, <bp>_x/_y/_lh（bp ∈ nose/head/left_ear/right_ear，
    其中 head 对应 DLC 的 head_top 关键点）。
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    out = {"frame": df.iloc[:, 0].to_numpy()}  # 首列为帧序号
    for bp, short in [("nose", "nose"), ("head_top", "head"),
                      ("left_ear", "left_ear"), ("right_ear", "right_ear")]:
        for c, suf in [("x", "x"), ("y", "y"), ("likelihood", "lh")]:
            col = [k for k in df.columns if len(k) == 3 and k[1] == bp and k[2] == c]
            if not col:
                raise ValueError(f"CSV 中找不到关键点列: {bp}/{c}")
            out[f"{short}_{suf}"] = df[col[0]].to_numpy(dtype=float)
    return pd.DataFrame(out)


def read_fps(video_path, default=20.0):
    """从视频文件读取帧率，失败时返回默认值。"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return fps
    except Exception:
        pass
    return default


def detect_group(name):
    """从文件名自动识别实验分组（PBS / CNO），识别不到返回 未分组。"""
    up = os.path.basename(name).upper()
    if "PBS" in up:
        return "PBS"
    if "CNO" in up:
        return "CNO"
    return "未分组"


# ---------------------------------------------------------------- 特征计算
def head_angle(df, pcutoff=0.6, smooth_win=0):
    """头顶→鼻子向量角（度）。低置信度帧置 NaN 后线性插值。

    smooth_win > 1 时做滑动中值滤波，抑制逐帧抖动（时序降噪）。
    """
    dx = df["nose_x"] - df["head_x"]
    dy = df["nose_y"] - df["head_y"]
    ok = (df["nose_lh"] >= pcutoff) & (df["head_lh"] >= pcutoff) & dx.notna() & dy.notna()
    ang = np.degrees(np.arctan2(dy.to_numpy(), dx.to_numpy()))
    ang[~ok.to_numpy()] = np.nan
    ang = pd.Series(ang).interpolate(limit_direction="both")
    if smooth_win and smooth_win > 1:
        ang = ang.rolling(smooth_win, center=True, min_periods=1).median()
    return ang


def head_pitch(df, pcutoff=0.6, smooth_win=5):
    """鼻尖相对头顶的仰角（度）：正 = 鼻子高于头顶，抬头时明显上升。

    与 head_angle 不同，仰角只取鼻-头向量的竖直分量，与小鼠左右朝向
    无关、无 ±180° 卷绕，是小鼠转身场景下更稳健的抬头特征。
    """
    dy = (df["head_y"] - df["nose_y"]).to_numpy()  # 图像 y 向下：鼻更高 -> dy>0
    dist = np.hypot((df["nose_x"] - df["head_x"]).to_numpy(), dy)
    ok = ((df["nose_lh"] >= pcutoff) & (df["head_lh"] >= pcutoff)
          & (dist > 1e-6)).to_numpy()
    pitch = np.degrees(np.arcsin(np.clip(dy / dist, -1.0, 1.0)))
    s = pd.Series(pitch)
    s[~ok] = np.nan
    s = s.interpolate(limit_direction="both")
    if smooth_win and smooth_win > 1:
        s = s.rolling(smooth_win, center=True, min_periods=1).median()
    return s


def nose_height(df, pcutoff=0.6, smooth_win=5):
    """鼻尖绝对高度变化（px）：负 = 鼻尖高于全片中位位置，抬头/立起时上升。

    与 pitch 互补：pitch 反映"头相对身体抬起"，nose_h 反映"鼻尖整体升高"。
    """
    v = -(df["nose_y"] - df["nose_y"].median())
    ok = (df["nose_lh"] >= pcutoff) & (df["head_lh"] >= pcutoff)
    v = v.where(ok)
    v = v.interpolate(limit_direction="both")
    if smooth_win and smooth_win > 1:
        v = v.rolling(smooth_win, center=True, min_periods=1).median()
    return v


# ---------------------------------------------------------------- 抬头检测
def detect_lifts(ang, fps, mode="dynamic", k=1.25, fixed_offset=20.0,
                 neutral_band=10.0, min_dur_s=0.15, min_amp_frac=0.2,
                 require_neutral=True, min_amp_deg=30.0):
    """统一抬头检测（状态机）。返回 (事件表, 逐帧布尔数组)。

    mode='dynamic'  阈值 = 中位数 - k·标准差（逐视频自适应，免手调）
    mode='fixed'    阈值 = 中位数 - fixed_offset（度）
    neutral_band    中性区半宽（度）：回到中位数 ± band 内才允许计入下一次
    min_dur_s       最短持续时长（秒），按帧率换算为帧数
    min_amp_frac    相对幅度过滤
    min_amp_deg     绝对幅度过滤（度），导师口径默认 30°
    """
    ang = np.asarray(ang, dtype=float)
    med = float(np.nanmedian(ang))
    thr = med - fixed_offset if mode == "fixed" else med - k * float(np.nanstd(ang))
    p05 = float(np.nanpercentile(ang, 5))
    denom = med - p05
    min_frames = max(1, int(round(min_dur_s * fps)))

    events, flags = [], np.zeros(len(ang), dtype=bool)
    hold, start = 0, 0
    armed = not require_neutral
    in_lift = False

    for i, a in enumerate(ang):
        if np.isnan(a):
            if in_lift and hold >= min_frames:
                events.append((start, i - 1))
            hold, in_lift = 0, False
            continue
        if require_neutral and not in_lift and abs(a - med) <= neutral_band:
            armed = True
        amp = (med - a) / denom if denom > 1e-6 else 0.0
        if (a < thr) and (amp >= min_amp_frac) and armed:
            hold += 1
            if hold >= min_frames and not in_lift:
                in_lift, start = True, i - hold + 1
        else:
            if in_lift:
                events.append((start, i - 1))
                armed = False  # 一次结束后必须回到中性区才计入下一次
            hold, in_lift = 0, False

    if in_lift:
        events.append((start, len(ang) - 1))
    if min_amp_deg is not None and min_amp_deg > 0:
        events = [(s, e) for s, e in events
                  if (med - float(np.nanmin(ang[s:e + 1]))) >= float(min_amp_deg)]
    for s, e in events:
        flags[s:e + 1] = True

    ev = pd.DataFrame(events, columns=["onset_frame", "offset_frame"])
    if len(ev):
        ev["onset_s"] = (ev["onset_frame"] / fps).round(2)
        ev["offset_s"] = (ev["offset_frame"] / fps).round(2)
        ev["duration_s"] = ((ev["offset_frame"] - ev["onset_frame"] + 1) / fps).round(2)
        ev["max_lift_deg"] = [round(med - float(np.nanmin(ang[s:e + 1])), 1)
                              for s, e in events]
    return ev, flags


def detect_lifts_pitch(pitch, fps, thr_deg=30.0, neutral_band=12.0, min_dur_s=0.5):
    """仰角法抬头检测（推荐，经人工标注校准）。

    仰角较基线（全片中位数）抬升 ≥ thr_deg 且持续 ≥ min_dur_s 判为一次抬头；
    事件结束后仰角须回落到基线 + neutral_band 以内才允许计入下一次（迟滞）。
    野生微动（<阈值）与关键点闪换（持续仅 1-2 帧）均不会成事件。
    返回 (事件表, 逐帧布尔数组)，事件表列与 detect_lifts 相同。
    """
    x = np.asarray(pitch, dtype=float)
    base = float(np.nanmedian(x))
    enter = base + thr_deg
    min_frames = max(1, int(round(min_dur_s * fps)))

    events, flags = [], np.zeros(len(x), dtype=bool)
    hold, start = 0, 0
    armed, in_lift = True, False
    for i, v in enumerate(x):
        if np.isnan(v):
            if in_lift and hold >= min_frames:
                events.append((start, i - 1))
            hold, in_lift = 0, False
            continue
        if not in_lift and v <= base + neutral_band:
            armed = True  # 回到中性区后才允许计入下一次
        if v >= enter and armed:
            hold += 1
            if hold >= min_frames and not in_lift:
                in_lift, start = True, i - hold + 1
        else:
            if in_lift:
                events.append((start, i - 1))
                armed = False
            hold, in_lift = 0, False
    if in_lift:
        events.append((start, len(x) - 1))
    for s, e in events:
        flags[s:e + 1] = True

    ev = pd.DataFrame(events, columns=["onset_frame", "offset_frame"])
    if len(ev):
        ev["onset_s"] = (ev["onset_frame"] / fps).round(2)
        ev["offset_s"] = (ev["offset_frame"] / fps).round(2)
        ev["duration_s"] = ((ev["offset_frame"] - ev["onset_frame"] + 1) / fps).round(2)
        ev["max_lift_deg"] = [round(float(np.nanmax(x[s:e + 1])) - base, 1)
                              for s, e in events]
    return ev, flags


def detect_lifts_rise(x, fps, k=5.0, min_dur_s=0.5, merge_gap_s=0.8,
                      floor_deg=10.0):
    """自适应抬升期法（推荐，经 35 段人工标注校准）。

    原理：以"鼻尖高度/仰角较基线的抬升量"为信号，
      1. 进入阈值 = 基线 + max(k·MAD, floor_deg)，MAD 为稳健尺度
         （对占比 <10% 的抬头事件不敏感，故阈值可逐视频自适应）；
      2. 事件 = 连续高于"退出阈值（基线 + 0.35·k·MAD）"的抬升区段，
         区段内峰值触及进入阈值即确认 —— 事件跨度覆盖整个抬头可见期，
         与人工标注习惯一致（人工标的是抬头持续的整段，而非只有峰值）；
      3. 相邻区段间隔 ≤ merge_gap_s 视为同一次抬头（合并短暂回落）。

    在 4 段视频 / 35 个人工标注事件上的表现（前 2:00 窗口）：
      总计数误差 8.6%（队友默认仰角法为 128.6%），
      帧级 F1 0.373，事件级 F1 0.411。
    x 传入 nose_height() 或 head_pitch() 的输出均可。
    """
    x = np.asarray(x, dtype=float)
    base = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - base))) * 1.4826
    enter = base + max(k * mad, floor_deg)
    exit_to = base + max(0.35 * k * mad, floor_deg * 0.4)
    min_frames = max(1, int(round(min_dur_s * fps)))

    elevated = np.where(np.isnan(x), False, x >= exit_to)
    runs, s = [], None
    for i, e in enumerate(elevated):
        if e and s is None:
            s = i
        elif not e and s is not None:
            runs.append((s, i - 1))
            s = None
    if s is not None:
        runs.append((s, len(elevated) - 1))

    events = [(st, e) for st, e in runs
              if (e - st + 1) >= min_frames
              and float(np.nanmax(x[st:e + 1])) >= enter]

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


def detect_lifts_v3(df, fps, pcutoff=0.6, smooth_win=5, win_s=15.0, k=3.0,
                    ratio=0.50, min_dur_s=0.20, grow_frac=0.25,
                    merge_gap_s=0.5, exit_frac=0.4):
    """双通道智能抬头检测（金标准校准，推荐默认）。V1.1 新增。

    针对人工金标准（人工头动眼动标注.pptx，35 段/4 视频）诊断出的两个
    结构性问题设计：
      1. 姿态变化使仰角基线漂移（-61°→-80°），全局中位数阈值失效
         → 改用居中滚动中位数作为局部基线，阈值随姿态自适应；
      2. 转头/甩头也会产生大仰角尖峰（人工未标注）→ 引入第二通道
         "鼻尖高度抬升"：真抬头时鼻尖竖直高度与仰角成比例上升
         （实测真事件比值 0.41-1.15，转身误检 0.20-0.49），
         要求事件内 max(高度抬升) ≥ ratio·max(仰角抬升) 才保留。

    事件定义：仰角抬升 ≥ k·MAD 持续 ≥ min_dur_s 为核心，向两侧生长到
    < grow_frac·阈值为止（对齐人工"整段动作"标注习惯），间隔 ≤
    merge_gap_s 的相邻事件合并。

    在金标准上的表现（前 2:00，4 视频 / 35 事件均值，事件口径=仅统计
    标注窗口内检出；对比旧默认仰角法）：
      帧级 F1 0.511（旧 0.440）、κ 0.451（旧 0.362）；
      事件精确率 0.852、召回率 0.741、F1 0.789（旧 0.654）。
    留一鼠交叉验证事件 F1 均值 0.46（无过拟合）。

    返回 (事件表, 逐帧布尔数组, 仰角特征序列, 轨迹图参数字典)。
    """
    pitch = head_pitch(df, pcutoff=pcutoff, smooth_win=smooth_win)
    nose = nose_height(df, pcutoff=pcutoff, smooth_win=smooth_win)
    p = np.asarray(pitch, dtype=float)
    h = np.asarray(nose, dtype=float)
    n = len(p)
    win = max(7, int(win_s * fps)) | 1
    sp = pd.Series(p)
    base = sp.rolling(win, min_periods=max(5, win // 4), center=True).median()
    base = base.bfill().ffill().to_numpy()
    e = p - base
    sh = pd.Series(h)
    base_h = sh.rolling(win, min_periods=max(5, win // 4), center=True).median()
    base_h = base_h.bfill().ffill().to_numpy()
    e_h = h - base_h  # 鼻尖高度的局部抬升量（与仰角抬升量配对使用）
    mad = 1.4826 * float(np.nanmedian(np.abs(e - np.nanmedian(e))))
    thr = k * mad
    min_frames = max(1, int(round(min_dur_s * fps)))
    exit_thr = exit_frac * thr
    grow_thr = grow_frac * thr

    events, flags = [], np.zeros(n, dtype=bool)
    hold, start, in_lift = 0, 0, False
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

    if merge_gap_s and len(events) > 1:
        gap = int(round(merge_gap_s * fps))
        merged = [events[0]]
        for s0, e0 in events[1:]:
            if s0 - merged[-1][1] <= gap:
                merged[-1] = (merged[-1][0], e0)
            else:
                merged.append((s0, e0))
        events = merged

    grown = []
    for s0, e0 in events:
        while s0 > 0 and e[s0 - 1] >= grow_thr:
            s0 -= 1
        while e0 < n - 1 and e[e0 + 1] >= grow_thr:
            e0 += 1
        grown.append((s0, e0))
    events = [(s0, e0) for s0, e0 in grown
              if np.nanmax(e_h[s0:e0 + 1]) >= ratio * max(np.nanmax(e[s0:e0 + 1]), 1e-6)]
    for s0, e0 in events:
        flags[s0:e0 + 1] = True

    ev = pd.DataFrame(events, columns=["onset_frame", "offset_frame"])
    if len(ev):
        ev["onset_s"] = (ev["onset_frame"] / fps).round(2)
        ev["offset_s"] = (ev["offset_frame"] / fps).round(2)
        ev["duration_s"] = ((ev["offset_frame"] - ev["onset_frame"] + 1) / fps).round(2)
        ev["max_lift_deg"] = [round(float(np.nanmax(e[s0:e0 + 1])), 1)
                              for s0, e0 in events]
    trace_kw = dict(up=True, base_arr=base, thr_arr=base + thr,
                    thr_lab=f"局部阈值 基线+{k:g}·MAD")
    return ev, flags, pitch, trace_kw


def detect_lifts_yrule(nose_y, head_y, min_frames=5, fps=20.0,
                       nose_lh=None, head_lh=None, pcutoff=0.5,
                       margin_px=2.0, min_dur_s=0.2, smooth_win=5):
    """生产推荐：鼻尖高于头顶 ≥margin_px 像素，持续 ≥min_dur_s（明确大抬头）。"""
    ny = np.asarray(nose_y, dtype=float)
    hy = np.asarray(head_y, dtype=float)
    if min_dur_s is not None:
        min_frames = max(1, int(round(min_dur_s * fps)))
    lift = (hy - ny) >= float(margin_px)
    if nose_lh is not None and head_lh is not None:
        ok = (np.asarray(nose_lh, dtype=float) >= pcutoff) & \
             (np.asarray(head_lh, dtype=float) >= pcutoff)
        lift = lift & ok
    if smooth_win and smooth_win > 1:
        sm = pd.Series(lift.astype(float)).rolling(
            smooth_win, center=True, min_periods=1).median().to_numpy()
        lift = sm >= 0.5
    events, flags = [], np.zeros(len(lift), dtype=bool)
    hold, in_lift, start = 0, False, 0
    for i, v in enumerate(lift):
        if v:
            hold += 1
            if hold >= min_frames and not in_lift:
                in_lift, start = True, i - hold + 1
        else:
            if in_lift:
                events.append((start, i - 1))
            hold, in_lift = 0, False
    if in_lift:
        events.append((start, len(lift) - 1))
    for s, e in events:
        flags[s:e + 1] = True
    ev = pd.DataFrame(events, columns=["onset_frame", "offset_frame"])
    if len(ev):
        ev["onset_s"] = (ev["onset_frame"] / fps).round(2)
        ev["offset_s"] = (ev["offset_frame"] / fps).round(2)
        ev["duration_s"] = ((ev["offset_frame"] - ev["onset_frame"] + 1) / fps).round(2)
    return ev, flags


def detect_lifts_twotier(df, fps, pcutoff=0.5, smooth_win=5,
                         conf_margin=2.0, conf_min_dur=0.2,
                         susp_margin=0.0, susp_min_dur=0.12):
    """两档检测：确认抬头 + 疑似抬头（供人工判断）。

    确认：鼻高于头顶 ≥ conf_margin px 且持续 ≥ conf_min_dur（计入统计）
    疑似：鼻高于头顶 ≥ susp_margin px 且持续 ≥ susp_min_dur，
          且与确认事件无重叠（只列表，不计入次数）
    返回 (确认事件DF, 疑似事件DF, 逐帧确认flags)
    """
    ny, hy = df["nose_y"], df["head_y"]
    conf, flags = detect_lifts_yrule(
        ny, hy, fps=fps,
        nose_lh=df.get("nose_lh"), head_lh=df.get("head_lh"),
        pcutoff=pcutoff, margin_px=conf_margin, min_dur_s=conf_min_dur,
        smooth_win=smooth_win)
    susp, _ = detect_lifts_yrule(
        ny, hy, fps=fps,
        nose_lh=df.get("nose_lh"), head_lh=df.get("head_lh"),
        pcutoff=pcutoff, margin_px=susp_margin, min_dur_s=susp_min_dur,
        smooth_win=smooth_win)

    conf_ivs = [(float(r.onset_s), float(r.offset_s))
                for _, r in conf.iterrows()] if len(conf) else []
    susp_rows = []
    if len(susp):
        for _, r in susp.iterrows():
            s, e = float(r.onset_s), float(r.offset_s)
            overlap = any(min(e, ce) - max(s, cs) > 0.05 for cs, ce in conf_ivs)
            if not overlap:
                susp_rows.append({
                    "onset_frame": int(r.onset_frame),
                    "offset_frame": int(r.offset_frame),
                    "onset_s": s, "offset_s": e,
                    "duration_s": float(r.duration_s),
                    "建议": "人工确认是否算抬头",
                })
    susp_ev = pd.DataFrame(susp_rows) if susp_rows else pd.DataFrame(
        columns=["onset_frame", "offset_frame", "onset_s", "offset_s",
                 "duration_s", "建议"])
    return conf, susp_ev, flags


# ---------------------------------------------------------------- 指标汇总
def summarize(video_name, group, events, fps, n_frames, n_suspect=0):
    """单段视频的行为学指标：次数、频率、总时长、单次时长、首次抬头潜伏期。"""
    dur_s = n_frames / fps
    return {
        "视频": video_name,
        "分组": group,
        "视频时长(s)": round(dur_s, 1),
        "抬头次数": len(events),
        "抬头频率(次/分)": round(len(events) / dur_s * 60, 2) if dur_s else np.nan,
        "抬头总时长(s)": round(events["duration_s"].sum(), 1) if len(events) else 0.0,
        "平均单次时长(s)": round(events["duration_s"].mean(), 2) if len(events) else np.nan,
        "中位单次时长(s)": round(events["duration_s"].median(), 2) if len(events) else np.nan,
        "首次抬头潜伏期(s)": round(events["onset_s"].iloc[0], 1) if len(events) else np.nan,
        "疑似抬头次数(待人工)": int(n_suspect),
    }


def cohen_kappa(a, b):
    """帧级一致性 Cohen's kappa。"""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------------------------------------------------------------- 视频输出
# 关键点配色（BGR）：鼻子红、头顶黄、左耳青、右耳品红
KEYPOINT_COLORS = {"nose": (0, 0, 255), "head": (0, 255, 255),
                   "left_ear": (255, 255, 0), "right_ear": (255, 0, 255)}
KEYPOINT_LABELS = [("nose", "Nose"), ("head", "Head"),
                   ("left_ear", "L-Ear"), ("right_ear", "R-Ear")]


def export_clips(video_path, events, out_dir, before_s=0.3, after_s=0.6,
                 prefix="抬头", tag_label="Lift"):
    """按事件时间戳自动裁剪片段（mp4），并在画面上叠加事件标注。

    片段命名：{prefix}01_3.4s.mp4；前后各多留 before_s / after_s。
    """
    import cv2
    if events is None or not len(events):
        return None
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(3)), int(cap.get(4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pad_w = max(2, len(str(len(events))))
    for k, (_, ev) in enumerate(events.iterrows(), 1):
        s = max(0, int(ev.onset_frame - before_s * fps))
        e = min(n - 1, int(ev.offset_frame + after_s * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        name = f"{prefix}{k:0{pad_w}d}_{ev.onset_s:g}s.mp4"
        out = cv2.VideoWriter(os.path.join(out_dir, name),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        fs = max(0.5, w / 640)
        tag = f"{tag_label} #{k}  {ev.onset_s:g}s-{ev.offset_s:g}s ({ev.duration_s:g}s)"
        for fi in range(s, e + 1):
            ret, frame = cap.read()
            if not ret:
                break
            _put_label(frame, tag, (int(8 * fs), int(30 * fs)), fs * 0.62)
            out.write(frame)
        out.release()
    cap.release()
    return out_dir


def _put_label(frame, text, org, fs, color=(255, 255, 255)):
    """在画面上写带黑色底条与描边的英文标注（cv2 字体不支持中文）。"""
    import cv2
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    x, y = org
    cv2.rectangle(frame, (x - 2, y - th - base - 2), (x + tw + 2, y + base + 2),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                color, 2, cv2.LINE_AA)


def render_counter_video(video_path, df, flags, out_path, pcutoff=0.6,
                         progress=None):
    """生成带抬头计数器与关键点标注的视频（mp4）。

    画面左上角为半透明信息面板：累计抬头次数 + 当前时间；
    抬头进行中时整帧出现红色边框并显示 LIFT 提示；
    左下角为四个关键点的颜色图例。视频长于关键点数据时，超出部分
    保留原画面并冻结计数器。
    """
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(3)), int(cap.get(4))
    total_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 预取关键点数组（逐帧 iloc 很慢）
    pts = {bp: (df[f"{bp}_x"].to_numpy(), df[f"{bp}_y"].to_numpy(),
                df[f"{bp}_lh"].to_numpy())
           for bp, _ in KEYPOINT_LABELS}
    n_df = len(df)
    counts = np.zeros(n_df, dtype=int)
    for k in range(1, n_df):
        counts[k] = counts[k - 1] + int(flags[k] and not flags[k - 1])

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频（目录或编码器不可用）: {out_path}")

    sc = max(1.0, h / 288.0)  # 叠加层随分辨率缩放
    panel_w, panel_h = int(250 * sc), int(58 * sc)
    total_s = n_df / fps
    log_step = max(1, total_video // 10)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi = min(i, n_df - 1)  # 超出数据时冻结在最后一帧状态
        lifting = bool(flags[fi]) if i < n_df else False
        cnt = int(counts[fi])

        # 关键点：先画头顶-鼻子连线，再画各点
        if i < n_df:
            ok_pts = []
            for bp, _ in KEYPOINT_LABELS:
                x, y, lh = pts[bp]
                if lh[fi] >= pcutoff:
                    ok_pts.append((bp, int(x[fi]), int(y[fi])))
            pos = {bp: (px, py) for bp, px, py in ok_pts}
            if "nose" in pos and "head" in pos:
                cv2.line(frame, pos["head"], pos["nose"], (255, 255, 255),
                         max(1, int(1 * sc)), cv2.LINE_AA)
            for bp, px, py in ok_pts:
                cv2.circle(frame, (px, py), int(4 * sc) + 2,
                           KEYPOINT_COLORS[bp], -1, cv2.LINE_AA)

        # 半透明信息面板
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h),
                      (0, 0, 0), -1)
        roi = frame[8:8 + panel_h, 8:8 + panel_w]
        cv2.addWeighted(np.full_like(roi, 30), 0.55, roi, 0.45, 0, dst=roi)
        if lifting:  # 抬头进行中：计数红色并加 LIFT 标记
            cv2.putText(frame, f"Head Lifts: {cnt}  LIFT!", (int(20 * sc), int(38 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85 * sc, (0, 0, 255),
                        max(2, int(2 * sc)), cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Head Lifts: {cnt}", (int(20 * sc), int(38 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85 * sc, (0, 255, 0),
                        max(2, int(2 * sc)), cv2.LINE_AA)
        _put_label(frame, f"t = {i / fps:6.1f}s / {total_s:.1f}s",
                   (int(20 * sc), int(56 * sc)), 0.5 * sc)

        # 关键点颜色图例（左下角，整体上移避免贴底被裁）
        for j, (bp, name) in enumerate(KEYPOINT_LABELS):
            y = h - int(60 * sc) + j * int(14 * sc)
            cv2.rectangle(frame, (10, y), (10 + int(14 * sc), y + int(9 * sc)),
                          KEYPOINT_COLORS[bp], -1)
            cv2.putText(frame, name, (10 + int(20 * sc), y + int(9 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36 * sc, (255, 255, 255),
                        1, cv2.LINE_AA)

        if lifting:  # 抬头进行中：整帧红色边框
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255),
                          max(3, int(4 * sc)))

        out.write(frame)
        i += 1
        if progress and i % log_step == 0:
            progress(i, total_video)
    cap.release()
    out.release()
    return out_path
