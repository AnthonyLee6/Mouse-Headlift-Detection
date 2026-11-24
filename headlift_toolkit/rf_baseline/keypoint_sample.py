# -*- coding: utf-8 -*-
"""
关键点评估 · 帧抽样 + 缓存导出
================================

为姿态模型评估抽 30 帧/视频 × 4 视频 = 120 帧，分层覆盖：
  随机(8) / 低置信度(6) / 阈值附近(6) / 人工事件内(5) / 高角速度(5)
并从 WMV 顺序解码导出 JPEG 到 keypoint_cache/，供 keypoint_gui.py 点击标注。

用法：python keypoint_sample.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.features import build_features  # noqa: E402
from rf_baseline.sample_frames import extract_frames, VIDEOS, VIDEO_DIR  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "keypoint_cache")
STRATA = ["随机", "低置信度", "阈值附近", "人工事件内", "高角速度"]


def sample_video(mouse, group, rng):
    _, csv_name, video_name = VIDEOS[mouse]
    csv_path = os.path.join(VIDEO_DIR, csv_name)
    F, meta = build_features(csv_path, os.path.join(VIDEO_DIR, video_name))
    fps, ang, thr = meta["fps"], meta["angle"], meta["thr"]
    idx = F.index.to_numpy()
    ok = ~np.isnan(ang)

    gold = pd.read_csv(os.path.join(BASE, "gold_抬头事件.csv"))
    gold = gold[(gold.mouse.astype(str) == mouse) & gold.note.isna()]
    t_all = idx / fps
    in_gold = np.zeros(len(idx), bool)
    for r in gold.itertuples():
        in_gold |= (t_all >= r.onset_s) & (t_all <= r.offset_s + 1)

    vel = np.abs(np.gradient(np.nan_to_num(ang, nan=meta["med"])))
    lh = F["lh_min"].to_numpy()

    used, rows = set(), []

    def pick(name, mask, n):
        cand = np.array([int(i) for i in idx[mask & ok] if int(i) not in used])
        if len(cand) == 0:
            return
        for f in rng.choice(cand, size=min(n, len(cand)), replace=False):
            used.add(int(f))
            rows.append({"mouse": mouse, "group": group, "frame_idx": int(f),
                         "time_s": round(float(f) / fps, 2), "stratum": name})

    pick("随机", np.ones(len(idx), bool), 8)
    pick("低置信度", lh <= np.nanpercentile(lh, 10), 6)
    pick("阈值附近", np.abs(ang - thr) <= 5, 6)
    pick("人工事件内", in_gold, 5)
    pick("高角速度", vel >= np.nanpercentile(vel, 90), 5)
    return rows, video_name


def main():
    rng = np.random.default_rng(42)
    rows_all = []
    for mouse, (group, _, _) in VIDEOS.items():
        rows, video_name = sample_video(mouse, group, rng)
        rows_all += rows
        n_cached = extract_frames(os.path.join(VIDEO_DIR, video_name),
                                  [r["frame_idx"] for r in rows],
                                  os.path.join(CACHE, mouse))
        vc = pd.DataFrame(rows).stratum.value_counts().reindex(STRATA).fillna(0).astype(int)
        print(f"[{mouse} {group}] 抽 {len(rows)} 帧，导出缓存 {n_cached} 张 {vc.tolist()}")

    out = os.path.join(BASE, "keypoint_manifest.csv")
    pd.DataFrame(rows_all).sort_values(["mouse", "frame_idx"]).to_csv(
        out, index=False, encoding="utf-8-sig")
    print(f"\n抽样清单 → {out}（共 {len(rows_all)} 帧）")


if __name__ == "__main__":
    main()
