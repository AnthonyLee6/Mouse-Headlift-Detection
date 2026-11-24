# -*- coding: utf-8 -*-
"""
预标注帧抽样（A-SOiD 式主动学习思想）+ 帧缓存提取
==================================================

对 4 只鼠的 DLC 关键点做分层抽样，把"人工最该看的帧"挑出来：
  A 事件内帧        规则法判定的抬头事件内（onset/peak/offset/随机内部）
  B 阈值附近帧      |角度-阈值| ≤ 5° 且不在事件内 —— 规则法最不可靠、最值得人工仲裁
  C 高角速度帧      角速度前 10% 且不在事件内 —— 抬起/放下的过渡段
  D 低置信度帧      任一关键点 likelihood < 0.6 —— 关键点不可靠的帧
  E 随机背景帧      其余帧随机补足

同时把选中帧从 WMV 顺序解码导出为 JPEG 缓存（帧号精确，避免 WMV 关键帧 seek 偏移），
供 label_gui.py 直接读图。

用法：
  python sample_frames.py                 # 抽样 + 导帧缓存 + 写 annotation_manifest.csv
  python sample_frames.py --per-video 120 # 每段视频少抽一些
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.features import build_features  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(BASE)
VIDEO_DIR = os.path.normpath(os.path.join(TOOLKIT, "..", "video"))

VIDEOS = {
    # mouse: (组别, DLC CSV, 原始视频)
    "1631": ("PBS", "1631 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1631 PBS.WMV"),
    "1632": ("PBS", "1632 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1632 PBS.WMV"),
    "1635": ("CNO", "1635 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1635 CNO.WMV"),
    "1637": ("CNO", "1637 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1637 CNO.WMV"),
}
STRATA = ["A_事件内", "B_阈值附近", "C_高角速度", "D_低置信度", "E_随机背景"]


def open_video_fallback(path):
    """cv2 打不开非 ASCII 路径时退回 Windows 8.3 短路径。"""
    import cv2
    import ctypes

    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap
    try:
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024):
            cap = cv2.VideoCapture(buf.value)
            if cap.isOpened():
                return cap
    except Exception:
        pass
    raise RuntimeError(f"无法打开视频: {path}")


def extract_frames(video_path, frame_indices, out_dir, quality=92):
    """顺序解码并导出指定帧为 JPEG（帧号与 CSV 对齐）。已存在的帧跳过。

    写盘用 imencode + 文件对象：cv2.imwrite 在 Windows 上对非 ASCII 路径会静默失败。
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    need = sorted({int(i) for i in frame_indices}
                  - {int(f[:6]) for f in os.listdir(out_dir) if f.endswith(".jpg")})
    if not need:
        return 0
    cap = open_video_fallback(video_path)
    need_set, done, i = set(need), 0, 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if i in need_set:
            ret, frame = cap.retrieve()
            if ret:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok:
                    with open(os.path.join(out_dir, f"{i:06d}.jpg"), "wb") as fh:
                        buf.tofile(fh)
                    done += 1
        i += 1
    cap.release()
    if done != len(need):
        raise RuntimeError(f"帧缓存导出不完整：{done}/{len(need)}（视频 {video_path}，"
                           f"总帧数可能少于清单最大帧号）")
    return done


def sample_video(mouse, group, per_video, rng):
    _, csv_name, video_name = VIDEOS[mouse]
    csv_path = os.path.join(VIDEO_DIR, csv_name)
    video_path = os.path.join(VIDEO_DIR, video_name)
    F, meta = build_features(csv_path, video_path)
    fps, ang, thr = meta["fps"], meta["angle"], meta["thr"]
    flags = meta["rule_flags"]
    idx = F.index.to_numpy()

    in_event = flags.astype(bool)
    outside = ~in_event
    ok = ~np.isnan(ang)
    vel = np.abs(np.gradient(np.nan_to_num(ang, nan=meta["med"])))

    parts = []
    # A 事件内：每个事件取 onset/peak/offset + 至多 2 个随机内部帧
    ev_frames = []
    for _, e in meta["rule_events"].iterrows():
        s, t = int(e.onset_frame), int(e.offset_frame)
        peak = s + int(np.nanargmin(ang[s:t + 1]))
        inner = rng.choice(idx[s:t + 1], size=min(2, t - s + 1), replace=False)
        ev_frames.extend([s, t, peak, *inner.tolist()])
    parts.append(("A_事件内", np.array(ev_frames, dtype=int), 40))
    parts.append(("B_阈值附近", idx[(np.abs(ang - thr) <= 5) & outside], 40))
    parts.append(("C_高角速度", idx[(vel >= np.nanpercentile(vel, 90)) & outside], 25))
    parts.append(("D_低置信度", idx[F["lh_min"].to_numpy() < 0.6], 20))

    used = set()

    def pick_from(cand, cap_n):
        cand = np.array(sorted({int(c) for c in cand if 0 <= int(c) < len(idx)}))
        cand = cand[ok[cand] & ~np.isin(cand, list(used))]
        if len(cand) == 0:
            return np.array([], dtype=int)
        return rng.choice(cand, size=min(cap_n, len(cand)), replace=False)

    chosen = []
    for name, cand, cap_n in parts:
        for f in pick_from(cand, cap_n):
            chosen.append((name, int(f)))
            used.add(int(f))

    # E 随机背景补足
    for f in pick_from(idx[outside], max(0, per_video - len(chosen))):
        chosen.append(("E_随机背景", int(f)))
        used.add(int(f))

    rows = []
    for stratum, f in sorted(chosen, key=lambda x: x[1]):
        rows.append({
            "mouse": mouse, "group": group, "frame_idx": int(f),
            "time_s": round(f / fps, 2),
            "ang": round(float(ang[f]), 1), "thr": round(thr, 1),
            "rule_flag": int(bool(flags[f])), "stratum": stratum,
            "suggestion": int(bool(flags[f])),
        })
    return pd.DataFrame(rows), meta, video_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=170)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    manifest, cache_stats = [], []
    for mouse, (group, _, _) in VIDEOS.items():
        rows, meta, video_path = sample_video(mouse, group, args.per_video, rng)
        manifest.append(rows)
        n_cached = extract_frames(video_path, rows["frame_idx"],
                                  os.path.join(BASE, "frame_cache", mouse))
        cache_stats.append((mouse, len(rows), n_cached))
        print(f"[{mouse} {group}] 抽样 {len(rows)} 帧 "
              f"({rows.stratum.value_counts().reindex(STRATA).fillna(0).astype(int).tolist()})"
              f"，本次导出帧缓存 {n_cached} 张，fps={meta['fps']:.2f}")

    out = os.path.join(BASE, "annotation_manifest.csv")
    m = pd.concat(manifest, ignore_index=True)
    m.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n标注清单 → {out}（共 {len(m)} 帧）")
    print("stratum 分布：")
    print(m.stratum.value_counts().reindex(STRATA).to_string())


if __name__ == "__main__":
    main()
