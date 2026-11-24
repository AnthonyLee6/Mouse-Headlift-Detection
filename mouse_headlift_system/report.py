# -*- coding: utf-8 -*-
"""
小鼠抬头行为自动识别分析系统 —— 报表与统计图模块 V1.0
======================================================

功能：批量检测完成后，自动生成
  1. Excel 多工作表报表（检测总览 / 抬头事件明细 / 参数设置 / 行为指标图）
  2. 每段视频的俯仰角轨迹图（含动态阈值线、中性区、检出事件标注）
  3. 组间行为指标对比图（自动按文件名识别 PBS / CNO 分组）
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN, C_ACC, C_BAND = "#334155", "#E8590C", "#0D9488"


def make_trace_figure(feat, fps, events, title, out_path, up=True,
                      thr_deg=None, band=None, base_arr=None, thr_arr=None,
                      thr_lab=None, ylab=None):
    """单段视频的特征轨迹图：曲线 + 基线/阈值/中性区 + 事件高亮。

    up=True  鼻尖仰角（pitch）：抬头时上升，阈值 = 基线 + thr_deg
    up=False 头顶→鼻子向量角（旧动态模式）：抬头时下降，阈值 = 中位数 - 1.5σ
    base_arr/thr_arr 提供时变基线/阈值（需与 feat 等长；标量会自动广播）
    """
    feat = np.asarray(feat, dtype=float)
    n = len(feat)
    base = float(np.nanmedian(feat))
    t = np.arange(n) / fps

    def _as_arr(v, fill):
        if v is None:
            return None
        a = np.asarray(v, dtype=float)
        if a.ndim == 0:
            return np.full(n, float(a))
        if len(a) != n:
            return np.full(n, float(fill))
        return a

    base_arr = _as_arr(base_arr, base)
    thr_arr = _as_arr(thr_arr, base)

    if base_arr is not None and thr_arr is not None:
        # 滚动局部基线 / y 规则阈值模式
        fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
        ax.plot(t, feat, lw=0.6, color=C_MAIN)
        ax.plot(t, base_arr, color=C_BAND, ls="--", lw=1, label="基线")
        ax.plot(t, thr_arr, color=C_ACC, ls="--", lw=1.2,
                label=thr_lab or "判定阈值")
        for _, e in events.iterrows():
            ax.axvspan(e.onset_s, e.offset_s, color=C_ACC, alpha=0.25)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("时间（秒）")
        ax.set_ylabel(ylab or "特征值")
        ax.set_title(f"{title} — 检出 {len(events)} 次抬头", fontsize=11)
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    if up:
        thr = base + float(thr_deg or 30.0)
        band = float(band or 12.0)
        lo, hi = base, thr
        ylab = "鼻尖仰角（度）\n抬头时上升"
        thr_lab = f"判定阈值 基线+{thr_deg or 30:g}°"
    else:
        thr = base - 1.5 * float(np.nanstd(feat))
        band = 10.0
        lo, hi = thr, base
        ylab = "头部俯仰角（度）\n抬头时角度下降"
        thr_lab = f"动态阈值 {thr:.0f}°"

    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
    ax.plot(t, feat, lw=0.6, color=C_MAIN)
    ax.axhline(base, color=C_BAND, ls="--", lw=1, label=f"基线中位数 {base:.0f}°")
    ax.axhline(thr, color=C_ACC, ls="--", lw=1.2, label=thr_lab)
    ax.fill_between(t, lo, hi, color=C_BAND, alpha=0.10, label="判定区间")
    for _, e in events.iterrows():
        ax.axvspan(e.onset_s, e.offset_s, color=C_ACC, alpha=0.25)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel(ylab)
    ax.set_title(f"{title} — 检出 {len(events)} 次抬头", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def make_group_figure(summary_df, out_path):
    """组间/个体间行为指标对比图：抬头频率、总时长、首次抬头潜伏期三联图。"""
    metrics = [("抬头频率(次/分)", "抬头频率（次/分钟）"),
               ("抬头总时长(s)", "抬头总时长（秒）"),
               ("首次抬头潜伏期(s)", "首次抬头潜伏期（秒）")]
    has_group = summary_df["分组"].nunique() > 1
    colors = {"PBS": "#64748B", "CNO": "#E8590C", "未分组": "#0D9488"}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=200)
    n_per = summary_df.groupby("分组").size().max() if has_group else 1
    bw = 0.8 / n_per  # 组内柱宽（留间隙）
    for ax, (col, label) in zip(axes, metrics):
        groups = summary_df["分组"].unique() if has_group else ["未分组"]
        for gi, g in enumerate(groups):
            sub = summary_df[summary_df["分组"] == g].reset_index(drop=True)
            n = len(sub)
            xs = (gi + (np.arange(n) - (n - 1) / 2) * bw) if has_group else np.arange(n)
            ax.bar(xs, sub[col], (bw - 0.08) if has_group else 0.6,
                   color=colors.get(g, "#0D9488"), label=g,
                   edgecolor="white", linewidth=1)
            for x, v, name in zip(xs, sub[col], sub["视频"]):
                ax.text(x, v + ax.get_ylim()[1] * 0.01, f"{v:g}",
                        ha="center", va="bottom", fontsize=7, color="#334155")
                ax.text(x, 0.15, str(name)[:4], ha="center", va="bottom",
                        rotation=90, fontsize=6.5, color="white")
        ax.set_xticks(range(len(groups)) if has_group else range(len(summary_df)))
        ax.set_xticklabels(groups if has_group else summary_df["视频"], fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("各组 / 各个体抬头行为指标对比", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _style_header(ws, ncols, fill="0D9488"):
    cell = ws.cell(row=1, column=1)
    ws.cell(row=1, column=ncols)
    for j in range(1, ncols + 1):
        c = ws.cell(row=1, column=j)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=fill)
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"


def _autosize(ws, df, max_w=30):
    for j, col in enumerate(df.columns, start=1):
        width = max(len(str(col)) * 2, *(len(str(v)) + 2 for v in df[col].head(50)))
        ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = min(max_w, width + 2)


def write_excel(out_path, summary_df, events_df, params, figure_paths,
                suspect_df=None):
    """生成多工作表 Excel 报表，并把统计图嵌入「行为指标图」表。

    suspect_df：疑似抬头明细（不计入次数，供人工判断）
    """
    if suspect_df is None:
        suspect_df = pd.DataFrame()
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        summary_df.to_excel(xw, sheet_name="检测总览", index=False)
        events_df.to_excel(xw, sheet_name="抬头事件明细", index=False)
        if len(suspect_df):
            suspect_df.to_excel(xw, sheet_name="疑似抬头_待人工", index=False)
        else:
            pd.DataFrame({"说明": ["本批无疑似抬头"]}).to_excel(
                xw, sheet_name="疑似抬头_待人工", index=False)
        pd.DataFrame([params]).to_excel(xw, sheet_name="参数设置", index=False)
        wb = xw.book
        ws = wb.create_sheet("行为指标图")
        ws.sheet_view.showGridLines = False

        _style_header(xw.sheets["检测总览"], len(summary_df.columns))
        _autosize(xw.sheets["检测总览"], summary_df)
        _style_header(xw.sheets["抬头事件明细"], len(events_df.columns))
        _autosize(xw.sheets["抬头事件明细"], events_df)
        if len(suspect_df):
            _style_header(xw.sheets["疑似抬头_待人工"], len(suspect_df.columns), fill="C2410C")
            _autosize(xw.sheets["疑似抬头_待人工"], suspect_df)

        ws["A1"] = "行为指标统计图（自动生成）"
        ws["A1"].font = Font(bold=True, size=14)
        row = 3
        for fp in figure_paths:
            img = XLImage(fp)
            scale = 760 / img.width
            img.width, img.height = int(img.width * scale), int(img.height * scale)
            ws.add_image(img, f"A{row}")
            row += int(img.height / 19) + 3
    return out_path
