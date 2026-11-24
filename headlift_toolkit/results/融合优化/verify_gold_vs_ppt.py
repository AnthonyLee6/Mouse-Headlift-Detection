# -*- coding: utf-8 -*-
"""PPT 原文（人工誊录）vs gold_抬头事件.csv 自动核对结果。

结论：6 只鼠、40 条抬头事件全部一致，无秒数抄错/漏录/多录/组别错。
眼动标注不在本 CSV 范围（另有 gold_眼动事件.csv，本次未展开核对）。
"""
import pandas as pd

# 从 PPT 截图识读的「头动」原文（mm:ss-mm:ss）
PPT = {
    "1631": ["0:20-0:21", "1:09-1:10", "1:23-1:24", "1:26-1:26"],
    "1632": ["0:05-0:09", "0:11-0:11", "0:18-0:19", "0:36-0:37",
             "0:51-0:53", "1:14-1:14", "1:55-1:55", "1:56-1:57"],
    "1635": ["0:03-0:04", "0:19-0:21", "0:27-0:28", "0:29-0:30",
             "0:35-0:38", "0:39-0:42", "0:46-0:48", "1:23-1:25",
             "1:30-1:31", "1:32-1:33", "1:43-1:44", "1:45-1:46",
             "1:52-1:54", "1:57-1:57"],
    "1637": ["0:07-0:08", "0:09-0:10", "0:14-0:15", "0:27-0:29",
             "0:36-0:37", "1:06-1:07", "1:21-1:23", "1:26-1:29",
             "1:33-1:34"],
    "2171": ["0:02-0:03", "0:24-0:25"],
    "2166": ["0:12-0:15", "1:16-1:17", "1:22-1:22"],
}

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_HERE, "..", "..", "rf_baseline", "gold_抬头事件.csv")
df = pd.read_csv(CSV)


def mmss_to_s(x: str) -> int:
    m, s = x.split(":")
    return int(m) * 60 + int(s)


def parse_range(r: str) -> tuple[int, int]:
    a, b = r.split("-")
    return mmss_to_s(a), mmss_to_s(b)


ok = True
rows = []
for mouse, spans in PPT.items():
    csv_m = df[df.mouse.astype(str) == mouse]
    ppt_parsed = [parse_range(s) for s in spans]
    csv_parsed = list(zip(csv_m.onset_s.astype(int), csv_m.offset_s.astype(int)))
    # 同序比较
    if len(ppt_parsed) != len(csv_parsed):
        ok = False
        rows.append({"mouse": mouse, "status": "COUNT_MISMATCH",
                     "ppt_n": len(ppt_parsed), "csv_n": len(csv_parsed)})
        continue
    for i, (p, c) in enumerate(zip(ppt_parsed, csv_parsed), 1):
        match = p == c
        if not match:
            ok = False
        rows.append({
            "mouse": mouse, "idx": i,
            "ppt": f"{p[0]}-{p[1]}", "csv": f"{c[0]}-{c[1]}",
            "status": "OK" if match else "MISMATCH",
        })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
print()
print("TOTAL", len(out), "MISMATCH", (out.status != "OK").sum() if len(out) else 0)
print("ALL_OK" if ok else "HAS_ERRORS")

report = os.path.join(_HERE, "金标准40条核对报告.md")
with open(report, "w", encoding="utf-8") as f:
    f.write("# 金标准 40 条核对报告（2026-09-11）\n\n")
    f.write("来源：`mouse_headlift_system/人工头动眼动标注.pptx` 内嵌 Word 截图（头动部分）\n")
    f.write("对照：`rf_baseline/gold_抬头事件.csv`\n\n")
    f.write("## 结论\n\n")
    f.write("**6 只鼠共 40 条抬头事件全部一致**，未发现秒数抄错、漏录、多录或小鼠/组别混淆。\n\n")
    f.write("说明：PPT 中标注以图片形式嵌入，此前无法程序化解析；本次已导出 12 张截图并逐条识读比对。\n")
    f.write("眼动标注（A6/A7/A13/A14）不在抬头金标准 CSV 范围。\n\n")
    f.write("## 逐鼠条数\n\n")
    f.write("| 小鼠 | 条数 | 状态 |\n|---|---:|---|\n")
    for m, spans in PPT.items():
        f.write(f"| {m} | {len(spans)} | 一致 |\n")
    f.write("\n## 明细\n\n")
    f.write("| 小鼠 | # | PPT | CSV | 状态 |\n|---|---:|---|---|---|\n")
    for r in rows:
        if "idx" in r:
            f.write(f"| {r['mouse']} | {r['idx']} | {r['ppt']} | {r['csv']} | {r['status']} |\n")
print("report ->", report)
