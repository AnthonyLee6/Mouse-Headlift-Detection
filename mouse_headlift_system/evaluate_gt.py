# -*- coding: utf-8 -*-
"""
人工金标准评估脚本 V1.0
========================

金标准来源：人工头动眼动标注.pptx（Word 截图转录）。
规则：每个视频只评估前 2:00；前 2 分钟内被标注的时间段 = 抬头（正样本），
未标注的一切行为（含未达阈值的轻微抬头、抽动）= 非抬头（负样本）。

评估内容：
  1. 帧级：准确率 / 精确率 / 召回率 / F1 / Cohen's kappa
  2. 事件级：检出事件与人工事件匹配（中心点落入人工区间 ±1s 容差）
     → 事件精确率 / 召回率 / F1，并输出漏检与误检明细供诊断

用法：python evaluate_gt.py            # 跑默认配置对比
      python evaluate_gt.py --sweep    # 仰角法参数扫描
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

import headlift_core as hl

BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.normpath(os.path.join(BASE, "..", "video"))
EVAL_WINDOW_S = 120.0  # 人工只标注了前 2 分钟

# ---- 人工金标准（分钟:秒 → 秒），含端点；仅前 2:00 ----
GROUND_TRUTH = {
    "1631 PBS":  [(20, 21), (69, 70), (83, 84), (86, 86)],
    "1632 PBS":  [(5, 9), (11, 11), (18, 19), (36, 37), (51, 53),
                  (74, 74), (115, 115), (116, 117)],
    "1635 CNO":  [(3, 4), (19, 21), (27, 28), (29, 30), (35, 38), (39, 42),
                  (46, 48), (83, 85), (90, 91), (92, 93), (103, 104),
                  (105, 106), (112, 114), (117, 117)],
    "1637 CNO":  [(7, 8), (9, 10), (14, 15), (27, 29), (36, 37), (66, 67),
                  (81, 83), (86, 89), (93, 94)],
}
CSV_FILES = {
    "1631 PBS": "1631 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv",
    "1632 PBS": "1632 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv",
    "1635 CNO": "1635 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv",
    "1637 CNO": "1637 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv",
}


def gt_frames(mouse, fps, n_frames):
    """人工事件 → 逐帧布尔数组（仅评估窗口内）。"""
    flags = np.zeros(n_frames, dtype=bool)
    for s, e in GROUND_TRUTH[mouse]:
        f0 = int(s * fps)
        f1 = min(int((e + 1) * fps) - 1, n_frames - 1)
        flags[f0:f1 + 1] = True
    return flags


def gt_events_s(mouse):
    return GROUND_TRUTH[mouse]


def frame_metrics(gt, pred):
    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(gt)
    kappa = hl.cohen_kappa(gt, pred)
    return dict(精确率=round(prec, 3), 召回率=round(rec, 3), F1=round(f1, 3),
                准确率=round(acc, 3), kappa=round(kappa, 3))


def event_metrics(gt_list, det_ev, fps, tol_s=1.5,
                  window_s=EVAL_WINDOW_S):
    """事件级匹配：检出事件中心落入某人工区间（±tol_s）即 TP；一个人工事件
    只能被匹配一次。仅统计评估窗口内的检出（窗口外人工未标注，不计误检）。"""
    det_centers = []
    for _, e in det_ev.iterrows():
        c = (e.onset_frame + e.offset_frame) / 2 / fps
        if c <= window_s:
            det_centers.append(c)
    used = set()
    tp, matched = 0, []
    for (s, e) in gt_list:
        gt_center = (s + e) / 2
        hit = None
        for j, c in enumerate(det_centers):
            if j in used:
                continue
            if s - tol_s <= c <= e + tol_s:
                hit = j
                break
        if hit is not None:
            used.add(hit)
            tp += 1
            matched.append(hit)
        # 未匹配的人工事件 = 漏检
    fp = len(det_centers) - len(used)
    fn = len(gt_list) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(事件精确率=round(prec, 3), 事件召回率=round(rec, 3),
                事件F1=round(f1, 3)), used, matched


def run_config(data, kind, **kw):
    """按配置对 4 段视频检测。返回 {mouse: (events, flags, fps, n)}。"""
    out = {}
    for mouse, csv in CSV_FILES.items():
        df, fps, n = data[mouse]
        if kind == "pitch":
            pitch = hl.head_pitch(df, pcutoff=kw.get("pcutoff", 0.6),
                                  smooth_win=kw.get("smooth", 5))
            ev, flags = hl.detect_lifts_pitch(pitch, fps, thr_deg=kw.get("thr", 30.0),
                                              neutral_band=kw.get("band", 12.0),
                                              min_dur_s=kw.get("min_dur", 0.5))
        elif kind == "rise":
            feat = hl.nose_height(df, pcutoff=kw.get("pcutoff", 0.6),
                                  smooth_win=kw.get("smooth", 5))
            ev, flags = hl.detect_lifts_rise(feat, fps, k=kw.get("k", 5.0),
                                             min_dur_s=kw.get("mind", 0.3),
                                             merge_gap_s=kw.get("gap", 0.5))
        elif kind == "angle":
            ang = hl.head_angle(df, pcutoff=kw.get("pcutoff", 0.6),
                                smooth_win=kw.get("smooth", 5))
            ev, flags = hl.detect_lifts(ang, fps, mode=kw.get("mode", "dynamic"),
                                        k=kw.get("k", 1.5),
                                        min_dur_s=kw.get("min_dur", 0.15))
        elif kind == "v3":
            ev, flags, _, _ = hl.detect_lifts_v3(
                df, fps, pcutoff=kw.get("pcutoff", 0.6),
                smooth_win=kw.get("smooth", 5), k=kw.get("k", 4.0),
                min_dur_s=kw.get("mind", 0.25))
        elif kind == "yrule":
            ev, flags = hl.detect_lifts_yrule(df["nose_y"], df["head_y"],
                                              min_frames=5, fps=fps)
        else:
            raise ValueError(kind)
        out[mouse] = (ev, flags, fps, n)
    return out


def evaluate(data, kind, detail=False, **kw):
    res = run_config(data, kind, **kw)
    fm_all, ev_all = [], []
    detail_rows = []
    for mouse, (ev, flags, fps, n) in res.items():
        w = int(EVAL_WINDOW_S * fps)
        gt = gt_frames(mouse, fps, n)[:w]
        pred = flags[:w]
        fm = frame_metrics(gt, pred)
        fm_all.append(fm)
        em, used, matched = event_metrics(gt_events_s(mouse), ev, fps)
        ev_all.append(em)
        if detail:
            det_centers = [round((e.onset_frame + e.offset_frame) / 2 / fps, 1)
                           for _, e in ev.iterrows()]
            missed = [(s, e) for k, (s, e) in enumerate(gt_events_s(mouse))
                      if k >= len(matched) or True]
            miss_list = []
            used2 = set(matched)
            for k, (s, e) in enumerate(gt_events_s(mouse)):
                c = (s + e) / 2
                if not any(abs(c - det_centers[j]) <= (e - s) / 2 + 1.5
                           for j in used2):
                    miss_list.append(f"{s}-{e}s")
            spurious = [f"{det_centers[j]}s" for j in range(len(det_centers))
                        if j not in used2 and det_centers[j] <= EVAL_WINDOW_S]
            detail_rows.append({
                "小鼠": mouse, "人工事件数": len(gt_events_s(mouse)),
                "检出事件数(窗口内)": sum(1 for c in det_centers
                                         if c <= EVAL_WINDOW_S),
                "漏检": " ".join(miss_list) or "-",
                "误检": " ".join(spurious) or "-"})
    avg = {k: round(float(np.mean([r[k] for r in fm_all])), 3) for k in fm_all[0]}
    avg_ev = {k: round(float(np.mean([r[k] for r in ev_all])), 3) for k in ev_all[0]}
    return {**avg, **avg_ev}, detail_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    data = {}
    for mouse, csv in CSV_FILES.items():
        df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv))
        fps = hl.read_fps(os.path.join(
            VIDEO_DIR, csv.split("DLC_resnet50")[0].strip() + ".WMV"))
        data[mouse] = (df, fps, len(df))
    print(f"评估窗口: 前 {EVAL_WINDOW_S:.0f}s | 人工事件总数:",
          sum(len(v) for v in GROUND_TRUTH.values()))

    if not args.sweep:
        rows = []
        configs = [
            ("双通道智能 v3(推荐)", "v3", {}),
            ("仰角法(默认 thr=30)", "pitch", {}),
            ("仰角法 thr=35", "pitch", {"thr": 35.0}),
            ("自适应抬升(高召回 k=5)", "rise",
             {"k": 5.0, "mind": 0.3, "gap": 0.5, "smooth": 5}),
            ("角度法 动态k=1.5", "angle", {}),
            ("y规则基线", "yrule", {}),
        ]
        for name, kind, kw in configs:
            m, _ = evaluate(data, kind, **kw)
            rows.append({"配置": name, **m})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        out = os.path.join(BASE, "金标准评估_基线.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print("已保存 →", out)
    else:
        rows = []
        for thr, band, mind, sm in itertools.product(
                [15, 20, 25, 30, 35], [8, 12, 16], [0.3, 0.5, 0.8], [3, 5, 9]):
            m, _ = evaluate(data, "pitch", thr=thr, band=band,
                            min_dur=mind, smooth=sm)
            rows.append({"thr": thr, "band": band, "min_dur": mind,
                         "smooth": sm, **m})
        df = pd.DataFrame(rows).sort_values("事件F1", ascending=False)
        print(df.head(15).to_string(index=False))
        out = os.path.join(BASE, "金标准评估_参数扫描.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print("已保存 →", out)


if __name__ == "__main__":
    main()
