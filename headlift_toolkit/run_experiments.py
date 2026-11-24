# -*- coding: utf-8 -*-
"""
批量实验脚本：在 4 只鼠（2 PBS + 2 CNO）的 DLC 关键点上运行统一流水线，
产出比赛材料所需的四类结果 → results/

  1. 消融对比表+图：固定阈值 vs 动态阈值 vs 加时序滤波（含 y 规则基线）
  2. 每只鼠的角度轨迹图（中位数/阈值/检出事件标注，可直接放进 PPT 成果页）
  3. CNO vs PBS 组间行为学指标对比（n=2/组，试点性质）
  4. 跨方法一致性：角度法 vs y 坐标法的帧级 Cohen's kappa
"""
import os

import matplotlib
import numpy as np
import pandas as pd

import headlift as hl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.normpath(os.path.join(BASE, "..", "video"))
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

VIDEOS = {
    "1631": ("PBS", "1631 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv"),
    "1632": ("PBS", "1632 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv"),
    "1635": ("CNO", "1635 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv"),
    "1637": ("CNO", "1637 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv"),
}

# 消融配置：名称 → 检测参数（yrule 为 y 坐标规则基线）
METHODS = {
    "固定阈值-20°": dict(mode="fixed", fixed_offset=20.0),
    "固定阈值-30°": dict(mode="fixed", fixed_offset=30.0),
    "动态阈值 k=1.25": dict(mode="dynamic", k=1.25),
    "动态阈值+时序滤波(推荐)": dict(mode="dynamic", k=1.25, smooth=5),
    "y坐标规则(基线)": dict(mode="yrule"),
}


def load(mouse):
    group, fname = VIDEOS[mouse]
    df = hl.load_dlc_csv(os.path.join(VIDEO_DIR, fname))
    fps = hl.read_fps(os.path.join(VIDEO_DIR, fname.replace(".csv", ".WMV")))
    return group, df, fps


def run_config(df, fps, method):
    cfg = METHODS[method]
    if cfg["mode"] == "yrule":
        return hl.detect_lifts_yrule(df["nose_y"], df["head_y"], fps=fps)
    ang = hl.head_angle(df, smooth_win=cfg.get("smooth", 0))
    return hl.detect_lifts(ang, fps, mode=cfg["mode"], k=cfg.get("k", 1.25),
                           fixed_offset=cfg.get("fixed_offset", 20.0))


def main():
    data = {m: load(m) for m in VIDEOS}

    # ---------- 1. 消融对比 ----------
    rows = []
    for mouse, (group, df, fps) in data.items():
        for method in METHODS:
            ev, _ = run_config(df, fps, method)
            r = hl.summarize(ev, fps, len(df))
            rows.append({"小鼠": mouse, "组别": group, "方法": method, **r})
    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(OUT, "消融对比_方法x小鼠.csv"),
               index=False, encoding="utf-8-sig")

    # 稳健性：同一方法在 4 只鼠上抬头频率的变异系数（越小越不依赖手调参数）
    rob = (abl[abl.方法 != "y坐标规则(基线)"]
           .groupby("方法")["lifts_per_min"]
           .agg(均值="mean", 标准差="std")
           .reset_index())
    rob["变异系数CV"] = (rob["标准差"] / rob["均值"]).round(3)
    rob.round(2).to_csv(os.path.join(OUT, "方法稳健性_跨小鼠CV.csv"),
                        index=False, encoding="utf-8-sig")
    print(rob.round(3).to_string(index=False), "\n")

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    mice = list(VIDEOS)
    x = np.arange(len(mice))
    width = 0.16
    colors = ["#94A3B8", "#64748B", "#0D9488", "#E8590C", "#CBD5E1"]
    for j, method in enumerate(METHODS):
        vals = [abl[(abl.小鼠 == m) & (abl.方法 == method)]["lifts_per_min"].iloc[0]
                for m in mice]
        ax.bar(x + (j - 2) * width, vals, width, label=method, color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m} ({VIDEOS[m][0]})" for m in mice])
    ax.set_ylabel("抬头频率（次/分钟）")
    ax.set_title("不同判定方法的抬头频率对比（阈值敏感性分析）")
    ax.legend(fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "图1_方法消融对比.png"))
    plt.close(fig)

    # ---------- 2. 每只鼠角度轨迹图（推荐配置） ----------
    for mouse, (group, df, fps) in data.items():
        ang = hl.head_angle(df, smooth_win=5)
        ev, flags = hl.detect_lifts(ang, fps, mode="dynamic", k=1.25)
        med = float(np.nanmedian(ang))
        thr = med - 1.25 * float(np.nanstd(ang))
        t = np.arange(len(ang)) / fps

        fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
        ax.plot(t, ang, lw=0.6, color="#334155")
        ax.axhline(med, color="#0D9488", ls="--", lw=1, label=f"中位数 {med:.0f}°")
        ax.axhline(thr, color="#E8590C", ls="--", lw=1.2, label=f"动态阈值 {thr:.0f}°")
        ax.fill_between(t, med - 10, med + 10, color="#0D9488", alpha=0.10,
                        label="中性区 ±10°")
        if len(ev):
            for _, e in ev.iterrows():
                ax.axvspan(e.onset_s, e.offset_s, color="#E8590C", alpha=0.25)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("时间（秒）")
        ax.set_ylabel("头部俯仰角（度）\n抬头时角度下降")
        ax.set_title(f"小鼠 {mouse}（{group}）头部俯仰角轨迹与抬头事件检测 "
                     f"— 检出 {len(ev)} 次", fontsize=11)
        ax.legend(fontsize=8, loc="lower right", ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"图2_俯仰角轨迹_{mouse}_{group}.png"))
        plt.close(fig)

    # ---------- 3. 组间对比（推荐配置，n=2/组 试点数据） ----------
    rows = []
    for mouse, (group, df, fps) in data.items():
        ev, _ = hl.detect_lifts(hl.head_angle(df, smooth_win=5), fps,
                                mode="dynamic", k=1.25)
        rows.append({"小鼠": mouse, "组别": group,
                     **hl.summarize(ev, fps, len(df))})
    grp = pd.DataFrame(rows)
    grp.to_csv(os.path.join(OUT, "组间行为指标_PBS_vs_CNO.csv"),
               index=False, encoding="utf-8-sig")

    metrics = [("lifts_per_min", "抬头频率（次/分钟）"),
               ("total_lift_s", "抬头总时长（秒）"),
               ("latency_first_s", "首次抬头潜伏期（秒）")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=200)
    n_per = grp.groupby("组别").size().max()
    bw = 0.8 / n_per  # 组内每根柱宽（留出组内间隙，防止缩小时误读分组）
    for ax, (col, label) in zip(axes, metrics):
        for gi, (g, c) in enumerate([("PBS", "#64748B"), ("CNO", "#E8590C")]):
            sub = grp[grp.组别 == g].reset_index(drop=True)
            xs = gi + (np.arange(len(sub)) - (len(sub) - 1) / 2) * bw
            ax.bar(xs, sub[col], bw - 0.08, color=c, label=g,
                   edgecolor="white", linewidth=1)
            for x, v, mid in zip(xs, sub[col], sub["小鼠"]):
                ax.text(x, v + ax.get_ylim()[1] * 0.01, f"{v:g}",
                        ha="center", va="bottom", fontsize=7, color="#334155")
                ax.text(x, 0.15, str(mid), ha="center", va="bottom",
                        rotation=90, fontsize=6.5, color="white")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["PBS", "CNO"])
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, title="n=2/组", title_fontsize=8, loc="upper right")
    fig.suptitle("PBS 对照组 vs CNO 组抬头行为对比（统一流水线输出，试点数据）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "图3_组间对比_PBS_vs_CNO.png"))
    plt.close(fig)

    # ---------- 4. 跨方法一致性（角度法 vs y 规则） ----------
    rows = []
    for mouse, (group, df, fps) in data.items():
        _, flags_ang = hl.detect_lifts(hl.head_angle(df, smooth_win=5), fps,
                                       mode="dynamic", k=1.25)
        _, flags_y = hl.detect_lifts_yrule(df["nose_y"], df["head_y"])
        rows.append({"小鼠": mouse, "组别": group,
                     "角度法抬头帧数": int(flags_ang.sum()),
                     "y规则抬头帧数": int(flags_y.sum()),
                     "帧级一致率": round(float((flags_ang == flags_y).mean()), 3),
                     "Cohen_kappa": round(hl.cohen_kappa(flags_ang, flags_y), 3)})
    agr = pd.DataFrame(rows)
    agr.to_csv(os.path.join(OUT, "跨方法一致性_角度法vs坐标法.csv"),
               index=False, encoding="utf-8-sig")

    print(abl.to_string(index=False))
    print()
    print(grp.to_string(index=False))
    print()
    print(agr.to_string(index=False))
    print("\n全部结果已输出 →", OUT)


if __name__ == "__main__":
    main()
