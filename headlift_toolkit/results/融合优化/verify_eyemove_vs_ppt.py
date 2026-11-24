# -*- coding: utf-8 -*-
"""眼动金标准：PPT 截图识读 vs gold_眼动事件.csv 核对。"""
import pandas as pd

PPT = {
    ("A14", "垂直", "下向上"): [
        "1:09-1:12", "1:12-1:15", "1:15-1:38", "1:38-1:43",
        "1:43-1:51", "1:55-2:01", "2:12-2:17",
    ],
    ("A14", "垂直", "上向下"): [
        "2:25-2:35", "2:35-2:48", "2:53-2:57",
    ],
    ("A6", "垂直", "下向上"): [
        "1:09-1:21", "1:57-2:04", "2:05-2:17",
    ],
    ("A6", "垂直", "上向下"): [
        "2:23-2:36",
    ],
    ("A7", "水平", "左向右"): [
        "1:10-1:14", "1:19-1:23", "1:23-1:28", "1:28-1:29",
        "1:29-1:31", "1:31-1:32", "1:35-1:39", "1:39-1:42",
        "1:43-1:48", "1:49-1:51", "1:51-1:57", "1:57-2:02",
        "2:02-2:07", "2:08-2:09", "2:09-2:17", "2:18-2:23",
        "2:23-2:28", "2:28-2:36", "2:36-2:42", "2:42-2:48",
        "2:48-2:53", "2:53-2:58",
    ],
    ("A13", "水平", "左向右"): [
        "1:13-1:17", "1:17-1:24", "1:27-1:33", "1:33-1:42",
        "1:42-1:50", "1:51-2:04", "2:04-2:14", "2:14-2:16",
        "2:19-2:22", "2:22-2:29", "2:29-2:35", "2:35-2:36",
        "2:36-2:39",
    ],
    ("A13", "水平", "右向左"): [
        "2:42-2:44", "2:44-2:45", "2:46-2:48", "2:49-2:51",
        "2:51-2:52", "2:52-2:55", "2:55-2:58", "2:58-3:01",
        "3:01-3:04", "3:04-3:05", "3:05-3:08", "3:08-3:10",
        "3:10-3:14", "3:14-3:17", "3:17-3:25", "3:25-3:28",
        "3:28-3:32", "3:32-3:34", "3:34-3:36",
    ],
}

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_HERE, "..", "..", "rf_baseline", "gold_眼动事件.csv")
df = pd.read_csv(CSV)


def mmss_to_s(x: str) -> int:
    m, s = x.split(":")
    return int(m) * 60 + int(s)


def parse_range(r: str):
    a, b = r.split("-")
    return mmss_to_s(a), mmss_to_s(b)


rows = []
issues = []
for key, spans in PPT.items():
    animal, typ, direction = key
    sub = df[(df.animal == animal) & (df["类型"] == typ) & (df["方向"] == direction)]
    ppt_parsed = [parse_range(s) for s in spans]
    csv_parsed = list(zip(sub.onset_s.astype(int), sub.offset_s.astype(int)))
    n = max(len(ppt_parsed), len(csv_parsed))
    for i in range(n):
        p = ppt_parsed[i] if i < len(ppt_parsed) else None
        c = csv_parsed[i] if i < len(csv_parsed) else None
        match = p == c
        # 额外检查 onset>offset
        bad_order = c is not None and c[0] > c[1]
        status = "OK" if match and not bad_order else ("OFFSET_LT_ONSET" if bad_order else "MISMATCH")
        if status != "OK":
            issues.append((key, i + 1, p, c, status))
        rows.append({
            "animal": animal, "类型": typ, "方向": direction, "idx": i + 1,
            "ppt": f"{p[0]}-{p[1]}" if p else "",
            "csv": f"{c[0]}-{c[1]}" if c else "",
            "csv_原文": sub.iloc[i]["原文"] if i < len(sub) and "原文" in sub.columns else "",
            "status": status,
        })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
print("\nTOTAL", len(out), "ISSUES", len(issues))
for it in issues:
    print("ISSUE", it)

report = os.path.join(_HERE, "金标准_眼动核对报告.md")
with open(report, "w", encoding="utf-8") as f:
    f.write("# 金标准眼动核对报告（2026-09-11）\n\n")
    f.write("来源：`人工头动眼动标注.pptx` 眼动截图 vs `gold_眼动事件.csv`\n\n")
    f.write(f"## 结论\n\n共 {len(out)} 条，发现 **{len(issues)}** 处问题。\n\n")
    if issues:
        f.write("### 问题明细\n\n")
        f.write("| 动物 | 类型 | 方向 | # | PPT 秒 | CSV 秒 | CSV原文 | 问题 |\n|---|---|---|---:|---|---|---|---|\n")
        for animal, typ, direction, idx, p, c, status in [(a, t, d, i, p, c, s) for (a,t,d,i,p,c,s) in [(k[0],k[1],k[2],i,p,c,s) for k,i,p,c,s in issues]]:
            pass
        for k, i, p, c, status in issues:
            f.write(f"| {k[0]} | {k[1]} | {k[2]} | {i} | {p} | {c} |  | {status} |\n")
        f.write("\n说明：PPT 原文清晰写为 `2:35-2:48`；CSV 的 onset/offset 被误写为 215/168（onset>offset），且「原文」列误录为 `3:35-2:48`。\n")
        f.write("**建议修正**：A14 垂直 上向下 第2条 → onset_s=155, offset_s=168, 原文=`2:35-2:48`。\n")
    f.write("\n## 各组条数\n\n| 组 | PPT | CSV |\n|---|---:|---:|\n")
    for key, spans in PPT.items():
        animal, typ, direction = key
        sub = df[(df.animal == animal) & (df["类型"] == typ) & (df["方向"] == direction)]
        f.write(f"| {animal} {typ} {direction} | {len(spans)} | {len(sub)} |\n")
print("report ->", report)
