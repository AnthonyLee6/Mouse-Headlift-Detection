# -*- coding: utf-8 -*-
"""
小鼠抬头行为统一检测流水线（比赛优化版）
=========================================

整合了项目里多套脚本的思路并修复已知 bug：
  - H_L_T3.py            → 中性区状态机（必须从平视抬起才计数）
  - final_analyze.py     → 动态阈值（中位数 ± k·σ）+ 幅度过滤 + 按帧率自适应最短时长
                           ※ 原脚本读列错位（把帧序号/似然值当坐标），本文件已修正
  - generate_custom_video_gui.py → y 坐标规则基线（头顶低于鼻子 = 垂直抬头）

角度定义（与 H_L_T3 一致）：
  ang = atan2(nose_y - head_y, nose_x - head_x)，即"头顶→鼻子"向量角。
  平视时鼻子在头顶下方，ang 为正（约 +90°）；抬头时鼻子升高，ang 下降。
  因此 抬头判定 = ang < 阈值。

用法示例：
  python headlift.py --csv "video/1631 PBS....csv" --video "video/1631 PBS.WMV"
  （仅有关键点 CSV 时 --video 可省略，fps 用 --fps 指定）
"""
import argparse
import os

import numpy as np
import pandas as pd

BODY_PARTS = ["nose", "head_top", "left_ear", "right_ear"]
DEFAULT_K = 1.25  # 金标准校准后的生产推荐动态阈值系数


# ---------------------------------------------------------------- 数据加载
def load_dlc_csv(csv_path):
    """读取 DeepLabCut 单动物 CSV，按 (bodypart, coord) 显式取列，杜绝列错位。

    返回 DataFrame: frame, <bp>_x, <bp>_y, <bp>_lh（bp ∈ nose/head/left_ear/right_ear，
    其中 head 即 DLC 的 head_top）。
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    out = {}
    out["frame"] = df.iloc[:, 0].to_numpy()  # 首列为帧序号
    for bp, short in [("nose", "nose"), ("head_top", "head"),
                      ("left_ear", "left_ear"), ("right_ear", "right_ear")]:
        for c, suf in [("x", "x"), ("y", "y"), ("likelihood", "lh")]:
            col = [k for k in df.columns if len(k) == 3 and k[1] == bp and k[2] == c]
            if not col:
                raise ValueError(f"CSV 中找不到关键点列: {bp}/{c}")
            out[f"{short}_{suf}"] = df[col[0]].to_numpy(dtype=float)
    return pd.DataFrame(out)


def read_fps(video_path, default=20.0):
    """从视频文件读取帧率；失败则返回 default。"""
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


# ---------------------------------------------------------------- 特征计算
def head_angle(df, pcutoff=0.6, smooth_win=0):
    """计算头顶→鼻子向量角（度）。低置信度帧置 NaN 后线性插值。

    smooth_win > 1 时先做滑动中值滤波（抑制逐帧抖动，即 PPT 2.1 提出的时序降噪）。
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


# ---------------------------------------------------------------- 检测算法
def detect_lifts(ang, fps, mode="dynamic", k=DEFAULT_K, fixed_offset=20.0,
                 neutral_band=10.0, min_dur_s=0.15, min_amp_frac=0.2,
                 require_neutral=True, min_amp_deg=30.0):
    """统一抬头检测（状态机）。ang: headlift.head_angle() 的输出。

    参数:
      mode          'dynamic'  → 阈值 = 中位数 - k·标准差（自适应每段视频）
                    'fixed'    → 阈值 = 中位数 - fixed_offset（度）
      neutral_band  中性区半宽（度）：只有回到中位数 ±band 内才允许计入下一次抬头
      min_dur_s     最短持续时长（秒），按帧率换算成帧数，自动适配 20/25/30fps
      min_amp_frac  相对幅度过滤：抬升幅度需达到 (中位数-5分位) 的该比例
      min_amp_deg   绝对幅度过滤（度）：峰值抬升 < 该值的事件丢弃（导师口径默认 30°）
      require_neutral  是否要求"从平视抬起"

    返回 (events DataFrame, 逐帧布尔数组)
    """
    ang = np.asarray(ang, dtype=float)
    med = float(np.nanmedian(ang))
    thr = med - fixed_offset if mode == "fixed" else med - k * float(np.nanstd(ang))
    p05 = float(np.nanpercentile(ang, 5))
    denom = med - p05
    min_frames = max(1, int(round(min_dur_s * fps)))

    events, flags = [], np.zeros(len(ang), dtype=bool)
    hold, start, armed = 0, 0, not require_neutral
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
        is_lift = (a < thr) and (amp >= min_amp_frac) and armed
        if is_lift:
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

    # 绝对幅度门控（导师：不到 30° 不算抬头）
    if min_amp_deg is not None and min_amp_deg > 0:
        events = [(s, e) for s, e in events
                  if (med - float(np.nanmin(ang[s:e + 1]))) >= float(min_amp_deg)]

    for s, e in events:
        flags[s:e + 1] = True

    ev = pd.DataFrame(events, columns=["onset_frame", "offset_frame"])
    if len(ev):
        ev["onset_s"] = ev["onset_frame"] / fps
        ev["offset_s"] = ev["offset_frame"] / fps
        ev["duration_s"] = (ev["offset_frame"] - ev["onset_frame"] + 1) / fps
        ev["max_lift_deg"] = [med - float(np.nanmin(ang[s:e + 1])) for s, e in events]
    return ev, flags


DEFAULT_Y_MARGIN_PX = 2.0    # 鼻尖须比头顶高至少这么多像素
DEFAULT_Y_MIN_DUR_S = 0.2    # 明确大抬头最短持续
DEFAULT_Y_SMOOTH = 5         # 中值滤波窗口（帧）


def detect_lifts_yrule(nose_y, head_y, min_frames=5, fps=20.0,
                       nose_lh=None, head_lh=None, pcutoff=0.6,
                       margin_px=DEFAULT_Y_MARGIN_PX,
                       min_dur_s=None, smooth_win=DEFAULT_Y_SMOOTH):
    """生产推荐：鼻尖高于头顶（图像 y 向下为正）。

    判定：head_y - nose_y >= margin_px（鼻子更高至少 margin_px 像素），
    低置信度帧不计，可选中值滤波；连续满足 min_dur_s 才算一次抬头。

    兼容旧调用：detect_lifts_yrule(nose_y, head_y, min_frames=5, fps=20)
    """
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
        ev["onset_s"] = ev["onset_frame"] / fps
        ev["offset_s"] = ev["offset_frame"] / fps
        ev["duration_s"] = (ev["offset_frame"] - ev["onset_frame"] + 1) / fps
        dy = hy - ny
        ev["max_dy_px"] = [float(np.nanmax(dy[s:e + 1])) for s, e in events]
    return ev, flags


def detect_lifts_yrule_df(df, fps=None, **kwargs):
    """从 DLC DataFrame 用生产 y 规则检出。"""
    if fps is None:
        fps = 20.0
    return detect_lifts_yrule(
        df["nose_y"], df["head_y"], fps=fps,
        nose_lh=df.get("nose_lh"), head_lh=df.get("head_lh"),
        **kwargs,
    )


# ---------------------------------------------------------------- 指标汇总
def summarize(events, fps, n_frames):
    """一段视频的行为学指标：次数、频率、总时长、单次时长、首次抬头潜伏期。"""
    dur_s = n_frames / fps
    out = {
        "video_duration_s": round(dur_s, 1),
        "n_lifts": len(events),
        "lifts_per_min": round(len(events) / dur_s * 60, 2) if dur_s else np.nan,
        "total_lift_s": round(events["duration_s"].sum(), 1) if len(events) else 0.0,
        "mean_dur_s": round(events["duration_s"].mean(), 2) if len(events) else np.nan,
        "median_dur_s": round(events["duration_s"].median(), 2) if len(events) else np.nan,
        "latency_first_s": round(events["onset_s"].iloc[0], 1) if len(events) else np.nan,
    }
    return out


def cohen_kappa(a, b):
    """帧级一致性 Cohen's kappa（手工实现，无需 sklearn）。"""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    po = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------------------------------------------------------------- 可选：片段裁剪
def export_clips(video_path, events, out_dir, before_s=0.3, after_s=0.6):
    """按事件时间戳自动裁剪抬头片段（供成果展示/答辩演示用）。"""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _, ev in events.iterrows():
        s = max(0, int(ev.onset_frame - before_s * fps))
        e = min(n - 1, int(ev.offset_frame + after_s * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        out = cv2.VideoWriter(
            os.path.join(out_dir, f"lift_{int(ev.onset_frame)}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(s, e + 1):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
        out.release()
    cap.release()
    return out_dir


# ---------------------------------------------------------------- 命令行
def main():
    ap = argparse.ArgumentParser(description="小鼠抬头检测统一流水线")
    ap.add_argument("--csv", required=True, help="DLC 关键点 CSV")
    ap.add_argument("--video", help="原始/标注视频（用于读 fps、裁片段）")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--mode", choices=["dynamic", "fixed"], default="dynamic")
    ap.add_argument("--k", type=float, default=1.25, help="动态阈值系数 σ（金标准校准推荐）")
    ap.add_argument("--fixed-offset", type=float, default=20.0)
    ap.add_argument("--smooth", type=int, default=5, help="中值滤波窗口（0/1 关闭）")
    ap.add_argument("--outdir", default="results/single")
    ap.add_argument("--clips", action="store_true", help="同时裁剪抬头片段")
    args = ap.parse_args()

    fps = read_fps(args.video) if args.video else args.fps
    df = load_dlc_csv(args.csv)
    ang = head_angle(df, smooth_win=args.smooth)
    ev, flags = detect_lifts(ang, fps, mode=args.mode, k=args.k,
                             fixed_offset=args.fixed_offset)
    s = summarize(ev, fps, len(df))
    print(f"视频时长 {s['video_duration_s']}s | 抬头 {s['n_lifts']} 次 "
          f"| {s['lifts_per_min']} 次/分 | 总时长 {s['total_lift_s']}s "
          f"| 潜伏期 {s['latency_first_s']}s")

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    ev.to_csv(os.path.join(args.outdir, f"{stem}_events.csv"),
              index=False, encoding="utf-8-sig")
    pd.DataFrame([s]).to_csv(os.path.join(args.outdir, f"{stem}_summary.csv"),
                             index=False, encoding="utf-8-sig")
    if args.clips and args.video:
        export_clips(args.video, ev, os.path.join(args.outdir, "clips"))
        print("片段已裁剪 →", os.path.join(args.outdir, "clips"))


if __name__ == "__main__":
    main()
