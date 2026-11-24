# -*- coding: utf-8 -*-
"""
生成伪标签（仅供流水线冒烟测试，禁止用于论文！）
==================================================

用规则法输出当"标签"填进 annotation_manifest 的抽样帧，跑通
train_rf.py / shap_analysis.py 的全链路。真实结果必须等 label_gui.py
的人工标注完成后重新运行。

用法：
  python make_pseudo_labels.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.features import build_features  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(BASE)
VIDEO_DIR = os.path.normpath(os.path.join(TOOLKIT, "..", "video"))
VIDEOS = {
    "1631": ("1631 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1631 PBS.WMV"),
    "1632": ("1632 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1632 PBS.WMV"),
    "1635": ("1635 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1635 CNO.WMV"),
    "1637": ("1637 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1637 CNO.WMV"),
}


def main():
    man = pd.read_csv(os.path.join(BASE, "annotation_manifest.csv"))
    rows = []
    for mouse, grp in man.groupby("mouse"):
        csv_name, video_name = VIDEOS[str(mouse)]
        csv_path = os.path.join(VIDEO_DIR, csv_name)
        _, meta = build_features(csv_path, os.path.join(VIDEO_DIR, video_name))
        for _, r in grp.iterrows():
            rows.append({"mouse": mouse, "frame_idx": int(r.frame_idx),
                         "label": int(bool(meta["rule_flags"][int(r.frame_idx)])),
                         "annotator": "PSEUDO(勿用于论文)"})
    out_dir = os.path.join(BASE, "labels")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "labels_PSEUDO_smoketest.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"伪标签（仅调试用）→ {out}\n"
          f"注意：跑完 train_rf.py 后请删除或改名此文件，避免混入真实标注。")


if __name__ == "__main__":
    main()
