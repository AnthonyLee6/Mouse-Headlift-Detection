# -*- coding: utf-8 -*-
"""
SHAP 可解释性分析（SimBA 风格的特征贡献）
==========================================

在全量已标注帧上训练 RF（仅用于解释特征贡献，性能以 train_rf.py 的留一鼠结果为准），
输出 SHAP 蜂群图 + 特征贡献条形图。

用法：
  python shap_analysis.py            # 读 labels/ 下所有标注
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.train_rf import BASE, TOOLKIT, load_labels, build_dataset, rf_model  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    lab = load_labels(os.path.join(BASE, "labels"))
    df, feat_cols = build_dataset(lab)
    X = df[feat_cols].to_numpy()
    clf = rf_model().fit(X, df.y.to_numpy())

    outdir = os.path.join(TOOLKIT, "results", "rf_baseline")
    os.makedirs(outdir, exist_ok=True)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X)

    # sklearn 新版二分类返回 (n, k, 2)，取正类通道
    if isinstance(sv, list):
        pos = sv[1]
    else:
        pos = sv[:, :, 1] if sv.ndim == 3 else sv

    plt.figure()
    shap.summary_plot(pos, X, feature_names=feat_cols, show=False, max_display=15)
    plt.gcf().set_size_inches(8, 6)
    plt.gcf().set_dpi(200)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "SHAP_蜂群图.png"))
    plt.close("all")

    mean_abs = np.abs(pos).mean(axis=0)
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    ax.barh([feat_cols[i] for i in order], mean_abs[order], color="#0D9488")
    ax.set_xlabel("mean(|SHAP|)  对抬头判定的平均贡献")
    ax.set_title("随机森林特征贡献（SHAP）")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "SHAP_特征贡献.png"))
    plt.close(fig)

    pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs.round(4)}
                 ).sort_values("mean_abs_shap", ascending=False
                 ).to_csv(os.path.join(outdir, "SHAP特征贡献.csv"),
                          index=False, encoding="utf-8-sig")
    print(f"SHAP 结果已输出 → {outdir}")


if __name__ == "__main__":
    main()
