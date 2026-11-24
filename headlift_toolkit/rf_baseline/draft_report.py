# -*- coding: utf-8 -*-
"""
预标注统计报告：统计"规则法 + y坐标法"双草稿意见的一致性，生成 预标注报告.md
==========================================================================

标注前花 1 分钟看完即可了解工作量分布：哪些帧两法一致（Enter 快进即可）、
哪些帧两法不一致（人工价值最高）。报告同时提醒"只标一部分时的评估陷阱"。

用法：
  python draft_report.py
"""
import os

import pandas as pd

from label_gui import STRATA_RANK, add_second_opinion

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    manifest = pd.read_csv(os.path.join(BASE, "annotation_manifest.csv"))
    m = add_second_opinion(manifest)
    n = len(m)
    lines = []
    lines.append("# 预标注统计报告（draft_report.py 自动生成）\n")
    lines.append(f"- 抽样帧总数：**{n}**（每鼠 170 帧 × 4）")
    lines.append(f"- 建议A 规则法阳性：{int(m.suggestion.sum())} 帧"
                 f"（{m.suggestion.mean() * 100:.1f}%）")
    lines.append(f"- 建议B y坐标法阳性：{int(m.yrule.sum())} 帧"
                 f"（{m.yrule.mean() * 100:.1f}%）")
    lines.append(f"- **两法不一致：{int(m.disagree.sum())} 帧"
                 f"（{m.disagree.mean() * 100:.1f}%）**——这些帧人工价值最高，GUI 默认优先推送\n")
    lines.append("## 每鼠分布\n")
    lines.append("| 小鼠 | 组别 | 帧数 | 规则法+ | y法+ | 两法不一致 |")
    lines.append("|---|---|---|---|---|---|")
    for (mouse, grp), g in m.groupby(["mouse", "group"]):
        lines.append(f"| {mouse} | {grp} | {len(g)} | {int(g.suggestion.sum())} | "
                     f"{int(g.yrule.sum())} | {int(g.disagree.sum())} |")
    lines.append("\n## 分层 × 两法一致性\n")
    lines.append("| 分层 | 帧数 | 不一致帧 | 不一致率 |")
    lines.append("|---|---|---|---|")
    strata = list(STRATA_RANK) + [s for s in m.stratum.unique() if s not in STRATA_RANK]
    for s in strata:
        g = m[m.stratum == s]
        if not len(g):
            continue
        lines.append(f"| {s} | {len(g)} | {int(g.disagree.sum())} | "
                     f"{g.disagree.mean() * 100:.0f}% |")
    lines.append("\n## 标注顺序与工作量\n")
    lines.append("- 默认优先级队列：先标两法不一致帧，再按分层 B>A>C>D>E；"
                 "两法一致的帧核对无误后按 `Enter` 快进。")
    lines.append("- 全部 680 帧预计 2–4 小时，建议分 2–3 次完成（防疲劳漂移），进度实时保存。")
    lines.append("- **若只标一部分**：做“规则法 vs 人工”评估必须用完整 680 帧或按分层随机子集，"
                 "不能只用已标帧——优先级队列偏向困难帧，直接用会高估方法误差。")
    lines.append("- 需要评分者间 κ：第二个人用自己名字再标一遍"
                 "（`python label_gui.py --annotator 姓名`），至少覆盖 2 段视频。")
    out = os.path.join(BASE, "预标注报告.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
