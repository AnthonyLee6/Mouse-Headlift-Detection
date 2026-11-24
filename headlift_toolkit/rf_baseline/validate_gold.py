# -*- coding: utf-8 -*-
"""
人工金标准验证（誊自 mouse_headlift_system/人工头动眼动标注.pptx）
====================================================================

金标准口径（PPT 第 1 页说明）：
  - 每个视频只标注前 2:00 内人工认为是抬头的部分（事件级，粒度 1 秒）；
  - 前 2 min 内未标注的所有行为（含未达阈值的轻微抬头、抽动）均不算抬头。

约定（本脚本实现）：
  - 评估窗口 = 0–120 s（标注覆盖范围）；
  - 人工事件 [onset, offset] 秒（闭区间，1s 粒度）→ 帧级正类窗 [onset, offset+1)；
  - 帧级评估剔除事件边界 ±0.5s 的帧（吸收 1s 标注粒度误差）；
  - 事件级匹配：人工区间向两侧扩 0.75s 后与系统检测求重叠，重叠>0 视为匹配
    （一对一贪心，按重叠最大配对）；匹配对统计 onset 时间误差。

输出 → ../results/金标准验证/：
  事件级_规则法vs人工.csv / 帧级_规则法vs人工.csv / RF留一鼠_帧级.csv
  金标准_事件对齐图.png / 金标准_帧级对比图.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.features import build_features  # noqa: E402
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS, metrics, rf_model  # noqa: E402
from headlift import cohen_kappa  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(BASE, "..", "results", "金标准验证"))

WINDOW = 120.0        # 标注覆盖窗口（秒）
BOUND_TOL = 0.5       # 帧级剔除的事件边界半宽（秒）
MATCH_TOL = 0.75      # 事件匹配时人工区间外扩（秒）

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_gold():
    g = pd.read_csv(os.path.join(BASE, "gold_抬头事件.csv"))
    # 2166/2171 已有 DLC CSV（服务器推理回传），一并纳入
    g = g[g.note.isna() | (g.note.astype(str).str.strip() == "")]
    return g


def match_events(gold_ivs, det_ivs):
    """gold_ivs/det_ivs: [(start_s, end_s)]；返回 (matched pairs, tp, fn, fp)。"""
    used = [False] * len(det_ivs)
    pairs = []
    for gs, ge in gold_ivs:
        es, ee = gs - MATCH_TOL, ge + MATCH_TOL
        best, best_ov = None, 0.0
        for j, (ds, de) in enumerate(det_ivs):
            if used[j]:
                continue
            ov = min(ee, de) - max(es, ds)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None:
            used[best] = True
            pairs.append((gs, ge, *det_ivs[best]))
    tp = len(pairs)
    return pairs, tp, len(gold_ivs) - tp, len(det_ivs) - tp


def main():
    os.makedirs(OUT, exist_ok=True)
    gold = load_gold()
    data = {}
    for mouse in gold.mouse.unique():
        _, csv_name, video_name = VIDEOS[str(mouse)]
        csv_path = os.path.join(VIDEO_DIR, csv_name)
        F, meta = build_features(csv_path, os.path.join(VIDEO_DIR, video_name))
        data[mouse] = (gold[gold.mouse == mouse], F, meta)

    # ---------------- 1. 事件级：规则法 vs 人工 ----------------
    ev_rows, align_rows, timeline, detail = [], [], [], []
    for mouse, (g, F, meta) in data.items():
        fps = meta["fps"]
        gold_ivs = [(r.onset_s, r.offset_s + 1.0) for r in g.itertuples()]
        det_ivs = [(r.onset_frame / fps, (r.offset_frame + 1) / fps)
                   for _, r in meta["rule_events"].iterrows()
                   if r.onset_frame / fps < WINDOW]
        pairs, tp, fn, fp = match_events(gold_ivs, det_ivs)
        for gs, ge, ds, de in pairs:
            detail.append({"mouse": mouse, "类型": "TP", "人工_onset_s": gs,
                           "人工_offset_s": ge - 1.0, "人工时长_s": ge - gs,
                           "系统_onset_s": round(ds, 2), "系统时长_s": round(de - ds, 2),
                           "onset误差_s": round(ds - gs, 2)})
        for gs, ge in gold_ivs:
            if not any(gs == p[0] and ge == p[1] for p in pairs):
                detail.append({"mouse": mouse, "类型": "FN(漏检)", "人工_onset_s": gs,
                               "人工_offset_s": ge - 1.0, "人工时长_s": ge - gs,
                               "系统_onset_s": np.nan, "系统时长_s": np.nan,
                               "onset误差_s": np.nan})
        for j, (ds, de) in enumerate(det_ivs):
            if not any(abs(ds - p[2]) < 0.01 for p in pairs):
                detail.append({"mouse": mouse, "类型": "FP(多余检出)", "人工_onset_s": np.nan,
                               "人工_offset_s": np.nan, "人工时长_s": np.nan,
                               "系统_onset_s": round(ds, 2), "系统时长_s": round(de - ds, 2),
                               "onset误差_s": np.nan})
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        on_err = [ds - gs for gs, ge, ds, de in pairs]
        ev_rows.append({"mouse": mouse, "人工事件数": len(gold_ivs),
                        "系统检出数(窗口内)": len(det_ivs), "TP": tp, "FP": fp, "FN": fn,
                        "precision": round(prec, 3), "recall": round(rec, 3),
                        "F1": round(f1, 3),
                        "onset误差均值_s": round(np.mean(np.abs(on_err)), 2) if on_err else np.nan,
                        "onset误差最大_s": round(np.max(np.abs(on_err)), 2) if on_err else np.nan})
        timeline.append((mouse, gold_ivs, det_ivs, pairs))
        print(f"[{mouse}] 人工 {len(gold_ivs)} | 系统 {len(det_ivs)} | "
              f"P {prec:.2f} R {rec:.2f} F1 {f1:.2f} | onset 误差 "
              f"{np.mean(np.abs(on_err)) if on_err else float('nan'):.2f}s")

    # 汇总（拼接全部区间再匹配一次）
    all_g = sum([t[1] for t in timeline], [])
    all_d = sum([t[2] for t in timeline], [])
    _, tp, fn, fp = match_events(all_g, all_d)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    ev_rows.append({"mouse": "汇总", "人工事件数": len(all_g),
                    "系统检出数(窗口内)": len(all_d), "TP": tp, "FP": fp, "FN": fn,
                    "precision": round(prec, 3), "recall": round(rec, 3),
                    "F1": round(2 * prec * rec / (prec + rec) if prec + rec else 0, 3),
                    "onset误差均值_s": np.nan, "onset误差最大_s": np.nan})
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(os.path.join(OUT, "事件级_规则法vs人工.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(detail).sort_values(["mouse", "人工_onset_s"], na_position="last"
        ).to_csv(os.path.join(OUT, "事件级_逐事件明细.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 2. 帧级：规则法 vs 人工（剔除边界带） ----------------
    fr_rows, X_all, y_all, m_all, r_all = [], [], [], [], []
    for mouse, (g, F, meta) in data.items():
        fps = meta["fps"]
        n = meta["n_frames"]
        t = np.arange(n) / fps
        pos = np.zeros(n, bool)
        band = np.zeros(n, bool)
        for r in g.itertuples():
            s, e = r.onset_s, r.offset_s + 1.0
            pos |= (t >= s) & (t < e)
            band |= (np.abs(t - s) < BOUND_TOL) | (np.abs(t - e) < BOUND_TOL)
        evaluable = (t < WINDOW) & ~band
        y = pos[evaluable].astype(int)
        rule = meta["rule_flags"][evaluable].astype(int)
        m = metrics(y, rule)
        fr_rows.append({"mouse": mouse, "评估帧数": int(evaluable.sum()),
                        "抬头帧数": int(y.sum()), **{k: round(v, 3) for k, v in m.items()}})
        cols = [c for c in F.columns]  # 特征列顺序即 RF 输入顺序
        X_all.append(F.iloc[np.where(evaluable)[0]].to_numpy())
        y_all.append(y)
        m_all.append(np.full(int(evaluable.sum()), mouse))
        r_all.append(rule)
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    mice = np.concatenate(m_all)
    rule = np.concatenate(r_all)
    fr = pd.DataFrame(fr_rows)
    fr.to_csv(os.path.join(OUT, "帧级_规则法vs人工.csv"), index=False, encoding="utf-8-sig")
    print("\n帧级（规则法 vs 人工）：")
    print(fr.to_string(index=False))

    # ---------------- 3. RF 留一鼠（同一评估帧集） ----------------
    rf_rows = []
    all_pred = np.zeros_like(y)
    for m in np.unique(mice):
        tr, te = mice != m, mice == m
        clf = rf_model().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        all_pred[te] = pred
        rf_rows.append({"test_mouse": m, "方法": "RF-留一鼠", **metrics(y[te], pred)})
    rf_rows.append({"test_mouse": "汇总", "方法": "RF-留一鼠", **metrics(y, all_pred)})
    rf_rows.append({"test_mouse": "汇总", "方法": "规则法", **metrics(y, rule)})
    rf = pd.DataFrame(rf_rows)
    for c in ("precision", "recall", "f1", "kappa"):
        rf[c] = rf[c].round(3)
    rf.to_csv(os.path.join(OUT, "RF留一鼠_帧级.csv"), index=False, encoding="utf-8-sig")
    print("\n帧级对比（对人工金标准）：")
    print(rf[rf.test_mouse == "汇总"].to_string(index=False))

    # ---------------- 图 1：事件对齐时间线 ----------------
    fig, axes = plt.subplots(len(timeline), 1, figsize=(11, 1.55 * len(timeline) + 1), dpi=200)
    for ax, (mouse, g_ivs, d_ivs, pairs) in zip(np.atleast_1d(axes), timeline):
        for k, (s, e) in enumerate(g_ivs):
            ax.axvspan(s, e, ymin=0.55, ymax=0.95, color="#0D9488", alpha=0.85, lw=0)
        for k, (s, e) in enumerate(d_ivs):
            hit = any(abs(s - ps) < MATCH_TOL + 0.01 for ps, _, ds, _ in pairs)
            ax.axvspan(s, e, ymin=0.08, ymax=0.48,
                       color="#E8590C" if hit else "#CBD5E1", alpha=0.9, lw=0)
        ax.set_xlim(0, WINDOW)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(mouse, rotation=0, ha="right", va="center", fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[-1].set_xlabel("时间（秒，仅前 2 分钟标注窗口）")
    fig.suptitle("人工金标准（上，青） vs 系统检测（下：橙=匹配，灰=多余检出）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "金标准_事件对齐图.png"))
    plt.close(fig)

    # ---------------- 图 2：帧级 P/R/F1 ----------------
    summ = rf[rf.test_mouse == "汇总"]
    methods = ["规则法", "RF-留一鼠"]
    colors = ["#64748B", "#E8590C"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    for j, (m, c) in enumerate(zip(methods, colors)):
        sub = summ[summ.方法 == m].iloc[0]
        vals = [sub.precision, sub.recall, sub.f1]
        bars = ax.bar(x + (j - 0.5) * 0.3, vals, 0.3, label=m, color=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["精确率", "召回率", "F1"])
    ax.set_ylim(0, 1.08)
    ax.set_title("对人工金标准的帧级性能（前 2 分钟窗口，剔除事件边界 ±0.5s）")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "金标准_帧级对比图.png"))
    plt.close(fig)

    print(f"\n全部结果已输出 → {OUT}")


if __name__ == "__main__":
    main()
