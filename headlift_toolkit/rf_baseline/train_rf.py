# -*- coding: utf-8 -*-
"""
随机森林基线 vs 规则法 —— 帧级抬头分类对比
============================================

在同一套人工标签上比较三种方法（正类 = 抬头帧）：
  1. RF-留一鼠交叉验证（LOMO）  训练 3 只、测试第 4 只 —— 泛化性能，论文主指标
  2. RF-随机划分                全体帧混合随机 80/20 —— 刻意展示"不隔离动物时指标虚高"
  3. 规则法（动态阈值 k=1.5）   无需训练的对照

输出（--outdir，默认 results/rf_baseline）：
  RF基线_逐折指标.csv   每折每方法的 precision/recall/F1/kappa
  RF基线_汇总.csv       三方法汇总
  基线对比_RFvs规则法.png   P/R/F1 分组柱状图

用法：
  python train_rf.py                          # 读 labels/ 下所有 labels_*.csv
  python train_rf.py --labels labels/labels_张三.csv --outdir results/rf_baseline
"""
import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.features import build_features  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(BASE)
VIDEO_DIR = os.path.normpath(os.path.join(TOOLKIT, "..", "video"))
VIDEOS = {
    # mouse: (组别, DLC CSV, 原始视频)
    "1631": ("PBS", "1631 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1631 PBS.WMV"),
    "1632": ("PBS", "1632 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1632 PBS.WMV"),
    "1635": ("CNO", "1635 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1635 CNO.WMV"),
    "1637": ("CNO", "1637 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1637 CNO.WMV"),
    "2166": ("PBS", "ct 2166 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "ct 2166 PBS.WMV"),
    "2171": ("CNO", "ct 2171 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "ct 2171 CNO.WMV"),
}

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import precision_score, recall_score, f1_score  # noqa: E402
from headlift import cohen_kappa  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_labels(labels_dir):
    """合并所有标注人的标签；同一帧多人标注时取后提交者并提示（正式实验应先仲裁）。"""
    if os.path.isdir(labels_dir):
        files = sorted(glob.glob(os.path.join(labels_dir, "labels_*.csv")))
    else:
        files = [labels_dir]
    if not files:
        raise FileNotFoundError(f"{labels_dir} 下没有 labels_*.csv，请先用 label_gui.py 标注")
    frames = [pd.read_csv(f).assign(_src=os.path.basename(f)) for f in files]
    lab = pd.concat(frames, ignore_index=True)
    dup = lab.duplicated(["mouse", "frame_idx"], keep="last")
    if dup.any():
        print(f"⚠ {dup.sum()} 帧存在多人重复标注，已取最后提交者；正式实验请先仲裁为单一标签")
    lab = lab[~dup]
    lab = lab[lab.label.isin([0, 1, 9])]  # 9 = 无法判断，训练/评估时丢弃
    print(f"载入 {len(files)} 份标注，共 {len(lab)} 帧"
          f"（抬头 {int((lab.label == 1).sum())} / 非抬头 {int((lab.label == 0).sum())}"
          f" / 丢弃 {int((lab.label == 9).sum())}）")
    return lab


def build_dataset(lab):
    """按 (mouse, frame_idx) 合并标签与特征。"""
    feats, flags = {}, {}
    for mouse in lab.mouse.unique():
        _, csv_name, video_name = VIDEOS[str(mouse)]
        csv_path = os.path.join(VIDEO_DIR, csv_name)
        video_path = os.path.join(VIDEO_DIR, video_name)
        F, meta = build_features(csv_path, video_path)
        feats[mouse] = F
        flags[mouse] = meta["rule_flags"]
    rows = []
    for _, r in lab.iterrows():
        if r.label == 9:
            continue
        F = feats[r.mouse]
        f = int(r.frame_idx)
        if f not in F.index:
            print(f"⚠ 帧 {r.mouse}/{f} 不在特征表中，跳过")
            continue
        rows.append({"mouse": r.mouse, "frame_idx": f, "y": int(r.label),
                     "rule_flag": int(bool(flags[r.mouse][f])), **F.loc[f].to_dict()})
    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c not in ("mouse", "frame_idx", "y", "rule_flag")]
    return df, feat_cols


def metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa(y_true, y_pred),
    }


def rf_model():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default=os.path.join(BASE, "labels"))
    ap.add_argument("--labels", help="直接指定单份标签 CSV（覆盖 --labels-dir）")
    ap.add_argument("--outdir", default=os.path.join(TOOLKIT, "results", "rf_baseline"))
    args = ap.parse_args()

    lab = load_labels(args.labels) if args.labels else load_labels(args.labels_dir)
    df, feat_cols = build_dataset(lab)
    n_pos = int((df.y == 1).sum())
    print(f"\n有效样本 {len(df)} 帧（抬头 {n_pos}），特征 {len(feat_cols)} 维")
    if n_pos < 30:
        print("⚠ 抬头正类样本过少（<30），当前结果只能作为流水线调试，不能写进论文")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    X, y, mice = df[feat_cols].to_numpy(), df.y.to_numpy(), df.mouse.to_numpy()
    rule = df.rule_flag.to_numpy()

    # ---------- 1. RF 留一鼠交叉验证 ----------
    lomo_rows = []
    for m in np.unique(mice):
        tr, te = mice != m, mice == m
        clf = rf_model().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        r = {"test_mouse": m, "方法": "RF-留一鼠", **metrics(y[te], pred)}
        r.update(n_test=len(y[te]), n_pos_test=int(y[te].sum()))
        lomo_rows.append(r)
    # 汇总（拼接所有折的预测再算一次整体指标）
    all_pred = np.zeros_like(y)
    for m in np.unique(mice):
        clf = rf_model().fit(X[mice != m], y[mice != m])
        all_pred[mice == m] = clf.predict(X[mice == m])
    overall = {"test_mouse": "汇总", "方法": "RF-留一鼠", **metrics(y, all_pred)}
    lomo_rows.append(overall)

    # ---------- 2. RF 随机划分（展示虚高效应） ----------
    from sklearn.model_selection import StratifiedKFold
    naive_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        clf = rf_model().fit(X[tr], y[tr])
        naive_pred[te] = clf.predict(X[te])
    naive_rows = [{"test_mouse": "汇总", "方法": "RF-随机划分", **metrics(y, naive_pred),
                   "n_test": len(y), "n_pos_test": n_pos}]

    # ---------- 3. 规则法 ----------
    rule_rows = [{"test_mouse": m, "方法": "规则法", **metrics(y[mice == m], rule[mice == m]),
                  "n_test": int((mice == m).sum()), "n_pos_test": int(y[mice == m].sum())}
                 for m in np.unique(mice)]
    rule_rows.append({"test_mouse": "汇总", "方法": "规则法", **metrics(y, rule),
                      "n_test": len(y), "n_pos_test": n_pos})

    res = pd.DataFrame(lomo_rows + naive_rows + rule_rows)
    for c in ("precision", "recall", "f1", "kappa"):
        res[c] = res[c].round(3)
    res.to_csv(os.path.join(outdir, "RF基线_逐折指标.csv"), index=False, encoding="utf-8-sig")

    summ = (res[res.test_mouse == "汇总"]
            .set_index("方法")[["precision", "recall", "f1", "kappa"]].T)
    summ.to_csv(os.path.join(outdir, "RF基线_汇总.csv"), encoding="utf-8-sig")
    print("\n=== 汇总（对人工标签） ===")
    print(summ.round(3).to_string())

    # ---------- 图4 ----------
    methods = ["规则法", "RF-随机划分", "RF-留一鼠"]
    colors = ["#64748B", "#CBD5E1", "#E8590C"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    for j, (m, c) in enumerate(zip(methods, colors)):
        vals = [summ.loc[k, m] for k in ("precision", "recall", "f1")]
        bars = ax.bar(x + (j - 1) * 0.26, vals, 0.26, label=m, color=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["精确率 Precision", "召回率 Recall", "F1"])
    ax.set_ylim(0, 1.08)
    ax.set_title("随机森林基线 vs 规则法（帧级，对人工标签）")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "基线对比_RFvs规则法.png"))
    plt.close(fig)

    print(f"\n结果已输出 → {outdir}")


if __name__ == "__main__":
    main()
