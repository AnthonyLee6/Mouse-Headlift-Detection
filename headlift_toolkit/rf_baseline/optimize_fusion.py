# -*- coding: utf-8 -*-
"""
金标准驱动的检测优化：规则法变体 + RF 融合
============================================

目标：在与 validate_gold.py 完全相同的事件匹配口径下，提升事件级 F1。

策略：
  1. 规则法变体：require_neutral / neutral_band / k 扫描（直接打“合并漏检”）
  2. RF LOMO：在 0–120s 全窗口帧上训练（含边界），对测试鼠全视频预测概率
  3. 融合：
     - RF-only 事件
     - 规则法 ∪ RF（规则高精确骨干 + RF 补召回）
     - 规则法 + RF 生长（边界）
     - 规则法 + RF 拆分（长事件/疑似合并）

用法：
  python optimize_fusion.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from headlift import detect_lifts, head_angle  # noqa: E402
from rf_baseline.features import build_features  # noqa: E402
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS, rf_model  # noqa: E402
from rf_baseline.validate_gold import (  # noqa: E402
    BASE,
    MATCH_TOL,
    WINDOW,
    load_gold,
    match_events,
)

OUT = os.path.normpath(os.path.join(BASE, "..", "results", "融合优化"))
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ 工具
def gold_ivs_for(g: pd.DataFrame) -> list[tuple[float, float]]:
    return [(float(r.onset_s), float(r.offset_s) + 1.0) for r in g.itertuples()]


def frame_pos_from_gold(n: int, fps: float, g: pd.DataFrame) -> np.ndarray:
    t = np.arange(n) / fps
    pos = np.zeros(n, bool)
    for r in g.itertuples():
        pos |= (t >= r.onset_s) & (t < r.offset_s + 1.0)
    return pos


def segments_from_bool(pos: np.ndarray, fps: float, min_dur_s: float = 0.15,
                       gap_s: float = 0.5) -> list[tuple[float, float]]:
    """布尔帧序列 → 事件区间（秒）。合并 gap，丢弃过短段。"""
    pos = np.asarray(pos, bool)
    if pos.size == 0 or not pos.any():
        return []
    d = np.diff(pos.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if pos[0]:
        starts = [0] + starts
    if pos[-1]:
        ends = ends + [len(pos)]
    segs = list(zip(starts, ends))
    merged: list[list[int]] = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) / fps <= gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if (e - s) / fps >= min_dur_s:
            out.append((s / fps, e / fps))
    return out


def merge_intervals(ivs: list[tuple[float, float]], pad_s: float = 0.0
                    ) -> list[tuple[float, float]]:
    if not ivs:
        return []
    items = sorted((s - pad_s, e + pad_s) for s, e in ivs)
    merged = [list(items[0])]
    for s, e in items[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def eval_events(gold_ivs, det_ivs) -> dict:
    pairs, tp, fn, fp = match_events(gold_ivs, det_ivs)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    on_err = [abs(ds - gs) for gs, ge, ds, de in pairs]
    return {
        "n_gold": len(gold_ivs),
        "n_det": len(det_ivs),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "F1": round(f1, 3),
        "onset_mae_s": round(float(np.mean(on_err)), 3) if on_err else np.nan,
    }


# ------------------------------------------------------------------ 数据
def load_all():
    gold = load_gold()
    pack = {}
    for mouse in gold.mouse.unique():
        _, csv_name, video_name = VIDEOS[str(mouse)]
        csv_path = os.path.join(VIDEO_DIR, csv_name)
        F, meta = build_features(csv_path, os.path.join(VIDEO_DIR, video_name))
        g = gold[gold.mouse == mouse]
        pack[mouse] = {"F": F, "meta": meta, "gold": g}
        print(f"[data] mouse={mouse} frames={meta['n_frames']} fps={meta['fps']:.1f} "
              f"gold_events={len(g)}")
    return pack


def train_rf_proba(pack) -> dict[int, np.ndarray]:
    """LOMO：在测试鼠 0–120s 全窗口帧上训练，输出该鼠全视频 P(抬头)。"""
    X_parts, y_parts, m_parts = [], [], []
    meta = {}
    for mouse, d in pack.items():
        F, m, g = d["F"], d["meta"], d["gold"]
        n, fps = m["n_frames"], m["fps"]
        t = np.arange(n) / fps
        pos = frame_pos_from_gold(n, fps, g)
        win = t < WINDOW  # 只在标注覆盖窗口取训练帧
        X_parts.append(F.to_numpy()[win])
        y_parts.append(pos[win].astype(int))
        m_parts.append(np.full(int(win.sum()), mouse))
        meta[mouse] = (F, m)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    mice = np.concatenate(m_parts)
    print(f"[rf] train pool frames={len(y)} pos={int(y.sum())} "
          f"pos_rate={y.mean():.3f}")

    proba = {}
    for m in np.unique(mice):
        clf = rf_model().fit(X[mice != m], y[mice != m])
        F, _ = meta[m]
        p = clf.predict_proba(F.to_numpy())[:, 1]
        proba[m] = p
        # 窗口内帧级对照
        n, fps = meta[m][1]["n_frames"], meta[m][1]["fps"]
        t = np.arange(n) / fps
        pos = frame_pos_from_gold(n, fps, pack[m]["gold"])
        ev = t < WINDOW
        pred = (p >= 0.5) & ev
        tp = int((pred & pos & ev).sum())
        fp = int((pred & ~pos).sum())
        fn = int((~pred & pos).sum())
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        print(f"[rf] LOMO {m}: window frame P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
    return proba


# ------------------------------------------------------------------ 检测器
def rule_events(meta, k=1.5, require_neutral=True, neutral_band=10.0,
                min_dur_s=0.15, min_amp_frac=0.2):
    ang = meta["angle"]
    fps = meta["fps"]
    ev, _ = detect_lifts(
        ang, fps, mode="dynamic", k=k, neutral_band=neutral_band,
        min_dur_s=min_dur_s, min_amp_frac=min_amp_frac,
        require_neutral=require_neutral,
    )
    if not len(ev):
        return []
    return [(float(r.onset_s), float(r.offset_s) + 1.0 / fps) for _, r in ev.iterrows()]


def rf_only_events(proba, fps, thr=0.5, min_dur_s=0.15, gap_s=0.5,
                   window=None):
    pos = np.asarray(proba) >= thr
    if window is not None:
        pos = pos & window
    return segments_from_bool(pos, fps, min_dur_s=min_dur_s, gap_s=gap_s)


def fusion_union(rule_ivs, rf_ivs) -> list[tuple[float, float]]:
    return merge_intervals(list(rule_ivs) + list(rf_ivs))


def fusion_add_rf_only(rule_ivs, rf_ivs, min_overlap_s=0.15) -> list[tuple[float, float]]:
    """保留全部规则事件；仅加入与规则事件重叠不足的 RF 段。"""
    out = list(rule_ivs)
    for rs, re_ in rf_ivs:
        if not rule_ivs or max(interval_overlap((rs, re_), iv) for iv in rule_ivs) < min_overlap_s:
            out.append((rs, re_))
    return merge_intervals(out)


def fusion_grow(rule_ivs, proba, fps, thr=0.4, max_grow_s=1.0) -> list[tuple[float, float]]:
    """用 RF 概率把规则事件向两侧生长到 thr 以下。"""
    p = np.asarray(proba)
    n = len(p)
    grown = []
    for s, e in rule_ivs:
        i0 = max(0, int(round(s * fps)))
        i1 = min(n - 1, int(round(e * fps)))
        # 向左
        left = i0
        limit = max(0, i0 - int(max_grow_s * fps))
        while left > limit and p[left - 1] >= thr:
            left -= 1
        # 向右
        right = i1
        limit = min(n - 1, i1 + int(max_grow_s * fps))
        while right < limit and p[right + 1] >= thr:
            right += 1
        grown.append((left / fps, (right + 1) / fps))
    return merge_intervals(grown)


def fusion_split(rule_ivs, ang, fps, proba=None, min_sep_s=0.8,
                 dip_frac=0.35) -> list[tuple[float, float]]:
    """在规则事件内部找“角度回弹谷”，把疑似合并事件拆开。

    条件：
      - 事件时长 ≥ 1.2 * min_sep_s
      - 内部存在谷：角度回升到 thr 上方一段，再回落
    """
    ang = np.asarray(ang, dtype=float)
    med = float(np.nanmedian(ang))
    thr = med - 1.5 * float(np.nanstd(ang))
    min_sep = max(1, int(min_sep_s * fps))
    out = []
    for s, e in rule_ivs:
        i0 = max(0, int(round(s * fps)))
        i1 = min(len(ang) - 1, int(round(e * fps)))
        seg = ang[i0:i1 + 1]
        if len(seg) < int(1.2 * min_sep):
            out.append((s, e))
            continue
        # 在事件内部找高于 thr 的连续区间作为“谷/间隔”
        above = seg > thr
        # 只在中间 20%–80% 找分割点，避免在边界切
        lo = int(0.2 * len(seg))
        hi = int(0.8 * len(seg))
        best = None
        run = 0
        for i in range(lo, hi):
            if above[i]:
                run += 1
                if run >= min_sep:
                    # 分割点取这段谷的中点
                    mid = i - run // 2
                    best = mid
                    break
            else:
                run = 0
        if best is None:
            out.append((s, e))
        else:
            cut = (i0 + best) / fps
            left = (s, cut)
            right = (cut, e)
            # 两侧仍须够长
            if (left[1] - left[0]) >= 0.15 and (right[1] - right[0]) >= 0.15:
                out.extend([left, right])
            else:
                out.append((s, e))
    return merge_intervals(out)


# ------------------------------------------------------------------ 主流程
def main():
    pack = load_all()
    proba = train_rf_proba(pack)

    rows = []
    details = []

    # ---- A. 规则法变体扫描 ----
    k_list = [1.0, 1.25, 1.5, 1.75, 2.0]
    band_list = [5.0, 10.0, 15.0, 20.0]
    for k, band, req in itertools.product(k_list, band_list, [True, False]):
        name = f"规则 k={k} band={band} neutral={req}"
        per = []
        all_g, all_d = [], []
        for mouse, d in pack.items():
            ivs = rule_events(d["meta"], k=k, require_neutral=req, neutral_band=band)
            # 窗口内
            ivs = [(s, e) for s, e in ivs if s < WINDOW]
            g = gold_ivs_for(d["gold"])
            m = eval_events(g, ivs)
            per.append(m)
            all_g.extend(g)
            all_d.extend(ivs)
        overall = eval_events(all_g, all_d)
        row = {"method": name, "family": "rule", **overall}
        rows.append(row)

    # ---- B. RF-only ----
    for thr, gap, mind in itertools.product([0.35, 0.4, 0.5, 0.6], [0.3, 0.5, 0.8], [0.15, 0.25]):
        name = f"RF-only thr={thr} gap={gap} min={mind}"
        all_g, all_d = [], []
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            n = d["meta"]["n_frames"]
            t = np.arange(n) / fps
            ivs = rf_only_events(proba[mouse], fps, thr=thr, gap_s=gap,
                                 min_dur_s=mind, window=t < WINDOW)
            g = gold_ivs_for(d["gold"])
            all_g.extend(g)
            all_d.extend(ivs)
        overall = eval_events(all_g, all_d)
        rows.append({"method": name, "family": "rf", **overall})

    # ---- C. 融合：以默认规则法为骨干 ----
    base_rule_cfg = dict(k=1.5, require_neutral=True, neutral_band=10.0)
    for thr, gap, mind in itertools.product([0.3, 0.4, 0.5], [0.4, 0.6], [0.15, 0.2]):
        # C1 取并集
        all_g, all_d = [], []
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            n = d["meta"]["n_frames"]
            t = np.arange(n) / fps
            r = [(s, e) for s, e in rule_events(d["meta"], **base_rule_cfg) if s < WINDOW]
            f = rf_only_events(proba[mouse], fps, thr=thr, gap_s=gap, min_dur_s=mind,
                               window=t < WINDOW)
            ivs = merge_intervals(r + f)
            all_g.extend(gold_ivs_for(d["gold"]))
            all_d.extend(ivs)
        rows.append({"method": f"融合-并集 thr={thr} gap={gap} min={mind}",
                     "family": "fusion", **eval_events(all_g, all_d)})

        # C2 只补 RF 独有段
        all_g, all_d = [], []
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            n = d["meta"]["n_frames"]
            t = np.arange(n) / fps
            r = [(s, e) for s, e in rule_events(d["meta"], **base_rule_cfg) if s < WINDOW]
            f = rf_only_events(proba[mouse], fps, thr=thr, gap_s=gap, min_dur_s=mind,
                               window=t < WINDOW)
            ivs = fusion_add_rf_only(r, f)
            all_g.extend(gold_ivs_for(d["gold"]))
            all_d.extend(ivs)
        rows.append({"method": f"融合-补RF独有 thr={thr} gap={gap} min={mind}",
                     "family": "fusion", **eval_events(all_g, all_d)})

    # C3 RF 生长
    for thr, grow in itertools.product([0.3, 0.4, 0.5], [0.5, 1.0]):
        all_g, all_d = [], []
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            r = [(s, e) for s, e in rule_events(d["meta"], **base_rule_cfg) if s < WINDOW]
            ivs = fusion_grow(r, proba[mouse], fps, thr=thr, max_grow_s=grow)
            ivs = [(s, e) for s, e in ivs if s < WINDOW]
            all_g.extend(gold_ivs_for(d["gold"]))
            all_d.extend(ivs)
        rows.append({"method": f"融合-RF生长 thr={thr} grow={grow}s",
                     "family": "fusion", **eval_events(all_g, all_d)})

    # C4 规则拆分（不依赖 RF）
    for min_sep, req in itertools.product([0.8, 1.2], [True, False]):
        all_g, all_d = [], []
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            r = [(s, e) for s, e in rule_events(d["meta"], k=1.5, require_neutral=req,
                                                neutral_band=10.0) if s < WINDOW]
            ivs = fusion_split(r, d["meta"]["angle"], fps, min_sep_s=min_sep)
            all_g.extend(gold_ivs_for(d["gold"]))
            all_d.extend(ivs)
        rows.append({"method": f"规则+谷拆分 sep={min_sep} neutral={req}",
                     "family": "fusion", **eval_events(all_g, all_d)})

    # C5 最佳规则变体 + RF 补独有
    # 先从 rule 行里挑汇总 F1 最高
    rule_rows = [r for r in rows if r["family"] == "rule"]
    best_rule = max(rule_rows, key=lambda x: x["F1"])
    print(f"\n[best-rule] {best_rule['method']} F1={best_rule['F1']}")
    # 解析 best_rule 的配置（简单从 method 字符串取，避免再扫一遍）
    # 直接重扫一轮“最佳附近”的融合
    for k, band, req, thr, gap in itertools.product(
            [1.0, 1.25, 1.5], [5.0, 10.0, 15.0], [True, False],
            [0.35, 0.45], [0.5]):
        all_g, all_d = [], []
        for mouse, d in pack.items():
            fps = d["meta"]["fps"]
            n = d["meta"]["n_frames"]
            t = np.arange(n) / fps
            r = [(s, e) for s, e in rule_events(d["meta"], k=k, require_neutral=req,
                                                neutral_band=band) if s < WINDOW]
            f = rf_only_events(proba[mouse], fps, thr=thr, gap_s=gap, min_dur_s=0.15,
                               window=t < WINDOW)
            ivs = fusion_add_rf_only(r, f)
            all_g.extend(gold_ivs_for(d["gold"]))
            all_d.extend(ivs)
        rows.append({
            "method": f"融合2 k={k} band={band} neutral={req} +RF thr={thr}",
            "family": "fusion2",
            **eval_events(all_g, all_d),
        })

    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    df.to_csv(os.path.join(OUT, "融合优化_汇总.csv"), index=False, encoding="utf-8-sig")

    # Top 逐鼠明细
    top = df.iloc[0]
    print("\n========== TOP ==========")
    print(top.to_string())
    print("\nTop 15:")
    cols = ["method", "precision", "recall", "F1", "TP", "FP", "FN", "n_det"]
    print(df[cols].head(15).to_string(index=False))

    # 对 top 方法输出逐鼠
    # 重新解析执行 top method 的检测逻辑较繁，改为把所有 fusion2/rule 的 per-mouse
    # 在扫描时已丢弃；这里只补 top 一条的逐鼠明细（按 method 前缀重跑）
    # 简化：把每个方法的 overall 已足够；另存 top10 配置
    df.head(20).to_csv(os.path.join(OUT, "融合优化_Top20.csv"), index=False,
                       encoding="utf-8-sig")
    print(f"\n已输出 → {OUT}")


if __name__ == "__main__":
    main()
