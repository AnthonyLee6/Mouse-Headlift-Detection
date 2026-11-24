# -*- coding: utf-8 -*-
"""
深度序列基线：GRU 关键点序列分类器 vs 规则法 vs RF（对人工金标准）
==================================================================

回答审稿人问题的最后一块："和监督式时序行为分类（DeepEthogram/SimBA 一类思路）比，
你们的规则法/RF 处于什么水平？"

设计：
  - 输入：与 RF/规则法同一套可解释姿态信号（俯仰角/相对高度/角速度/耳朵/置信度，
    14 维原始逐帧量，不含滑窗统计——时序上下文交给 GRU 自己学）；
  - 模型：2 层 GRU（hidden 64）+ 线性头，输入窗 ±1 s（±20 帧），逐帧二分类；
  - 训练：BCEWithLogits(pos_weight^0.5 软化)，Adam；验证集 = 训练鼠 40–80s 的
    **连续时间段整块**（滑窗 ±1s 重叠会使随机抽帧验证集泄漏、AUPRC 虚高到 1.0，
    这是 v1 教训），缓冲带 ±1s 不参与训练；早停看验证 AUPRC，阈值在验证集上调 F1；
  - 评估：留一鼠（LOMO），帧级 P/R/F1/κ，口径与 validate_gold.py 完全一致
    （前 2 min 窗口、剔除事件边界 ±0.5s），正类 = 人工事件窗 [onset, offset+1)。

论文表述注意：GRU 在同样的 6 段视频上训练，规模极小，属"同数据可达性能"参照，
不是 SimBA/DeepEthogram 的完整复现。

输出 → ../results/金标准验证/：
  GRU留一鼠_帧级.csv / 帧级_三方法对比.csv / 金标准_三方法对比图.png
  GRU_逐帧预测.csv（审计用：test_mouse, frame_idx, y_true, y_pred, p）

用法：
  python seq_baseline.py [--epochs 120] [--hidden 64]
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rf_baseline.features import build_features  # noqa: E402
from rf_baseline.train_rf import VIDEO_DIR, VIDEOS, metrics  # noqa: E402
from rf_baseline.validate_gold import BOUND_TOL, WINDOW, load_gold  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(BASE, "..", "results", "金标准验证"))

HALF_WIN = 20  # 序列窗半宽（帧）：20 帧 @20fps = ±1s
VAL_T = (40.0, 80.0)  # 训练鼠内部验证时间段（秒），连续整块防滑窗泄漏
BUF = 1.0             # 验证段缓冲（秒），缓冲帧不参与训练/验证
SEQ_FEATS = ["ang", "ang_rel", "ang_raw", "vel_deg_s",
             "nose_y_rel", "head_y_rel", "dy_nose_head", "dist_nose_head",
             "ear_roll_deg", "ear_mid_y_rel", "ear_dist",
             "lh_min", "lh_nose", "lh_head"]


# ================================================================ 数据准备
def build_gold_frames(gold):
    """每鼠：全视频特征矩阵、可评估帧号、帧标签（与 validate_gold.py 同一口径）。"""
    data = {}
    for mouse in gold.mouse.unique():
        _, csv_name, video_name = VIDEOS[str(mouse)]
        F, meta = build_features(os.path.join(VIDEO_DIR, csv_name),
                                 os.path.join(VIDEO_DIR, video_name))
        fps, n = meta["fps"], meta["n_frames"]
        g = gold[gold.mouse == mouse]
        t = np.arange(n) / fps
        pos = np.zeros(n, bool)
        band = np.zeros(n, bool)
        for r in g.itertuples():
            s, e = r.onset_s, r.offset_s + 1.0
            pos |= (t >= s) & (t < e)
            band |= (np.abs(t - s) < BOUND_TOL) | (np.abs(t - e) < BOUND_TOL)
        evaluable = (t < WINDOW) & ~band
        data[mouse] = {"F": F[SEQ_FEATS].to_numpy(dtype=np.float32), "meta": meta,
                       "idx": np.where(evaluable)[0], "y": pos[evaluable].astype(np.int64)}
    return data


def make_sequences(X_video, centers):
    """从全视频特征矩阵取 centers 处 ±HALF_WIN 的窗（边缘复制填充）。"""
    pad = np.pad(X_video, ((HALF_WIN, HALF_WIN), (0, 0)), mode="edge")
    return np.stack([pad[i:i + 2 * HALF_WIN + 1] for i in centers])


# ================================================================ 模型
def train_fold(data, tr_mice, hidden, epochs, patience, seed=42):
    """在 tr_mice 上训练，验证集为每鼠 40–80s 连续段。返回 (state, mu, sd, n_ep, diag)。"""
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, f1_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_idx_all, va_idx_all, tr_y, va_y, tr_X, va_X = [], [], [], [], [], []
    for m in tr_mice:
        d = data[m]
        idx, fps = d["idx"], d["meta"]["fps"]
        t = idx / fps
        va = (t >= VAL_T[0]) & (t < VAL_T[1])
        buf = (t >= VAL_T[0] - BUF) & (t < VAL_T[1] + BUF)
        va_idx_all.append(idx[va])
        tr_idx_all.append(idx[~buf])
        va_y.append(d["y"][va])
        tr_y.append(d["y"][~buf])
        va_X.append(make_sequences(d["F"], idx[va]))
        tr_X.append(make_sequences(d["F"], idx[~buf]))
    Xtr = np.concatenate(tr_X)
    ytr = np.concatenate(tr_y)
    Xva = np.concatenate(va_X)
    yva = np.concatenate(va_y)

    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    norm = lambda a: (a - mu) / sd

    Xtr_t = torch.tensor(norm(Xtr), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(norm(Xva), dtype=torch.float32)

    # pos_weight 开平方软化：全强度重加权易塌缩到"全判阳性"
    ratio = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    pos_w = torch.tensor([ratio ** 0.5], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model = nn.GRU(Xtr.shape[-1], hidden, num_layers=2, batch_first=True, dropout=0.2)
    head = nn.Linear(hidden, 1)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                           lr=1e-3, weight_decay=1e-4)

    def val_eval():
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(head(model(Xva_t)[0][:, -1, :])).squeeze(-1).numpy()
        auprc = average_precision_score(yva, p)
        grid = np.linspace(0.05, 0.95, 37)
        f1s = [f1_score(yva, (p >= t).astype(int), zero_division=0) for t in grid]
        k = int(np.argmax(f1s))
        return auprc, float(grid[k]), f1s[k]

    best_state, best_auprc, best_thr, wait = None, -1.0, 0.5, 0
    bs = 256
    n_tr = len(ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        for k in range(0, n_tr, bs):
            sel = perm[k:k + bs]
            opt.zero_grad()
            out = head(model(Xtr_t[sel])[0][:, -1, :]).squeeze(-1)
            loss = lossf(out, ytr_t[sel])
            loss.backward()
            opt.step()
        auprc, thr, vf1 = val_eval()
        if auprc > best_auprc + 1e-4:
            best_auprc, wait, best_thr = auprc, 0, thr
            best_state = (copy.deepcopy(model.state_dict()),
                          copy.deepcopy(head.state_dict()))
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        best_state = (model.state_dict(), head.state_dict())
    model.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(head(model(Xva_t)[0][:, -1, :])).numpy()
    diag = {"val_auprc": round(best_auprc, 3), "thr": round(best_thr, 2),
            "val_f1": round(f1_score(yva, (p_va >= best_thr).astype(int), zero_division=0), 3),
            "val_pos_rate": round(float(yva.mean()), 3), "epochs": ep + 1}
    return (best_state, best_thr), mu, sd, diag


def predict_fold(X_te, state, mu, sd):
    import torch
    import torch.nn as nn

    (model_sd, head_sd), thr = state
    hidden = model_sd["weight_hh_l0"].shape[1]  # (3*hidden, hidden) → 第二维即 hidden
    model = nn.GRU(X_te.shape[-1], hidden, num_layers=2, batch_first=True, dropout=0.2)
    model.load_state_dict(model_sd)
    model.eval()  # 关掉 dropout，否则逐帧随机性污染预测
    head = nn.Linear(hidden, 1)
    head.load_state_dict(head_sd)
    with torch.no_grad():
        x = torch.tensor((X_te - mu) / sd, dtype=torch.float32)
        p = torch.sigmoid(head(model(x)[0][:, -1, :])).squeeze(-1).numpy()
    return (p >= thr).astype(int), p


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    gold = load_gold()
    data = build_gold_frames(gold)
    print("金标准可评估帧："
          + ", ".join(f"{m}={len(d['y'])}(+{int(d['y'].sum())})" for m, d in data.items()))

    mice = list(data.keys())
    dump_rows, all_pred, y_all, m_all = [], [], [], []
    for test in mice:
        tr_mice = [m for m in mice if m != test]
        X_te = make_sequences(data[test]["F"], data[test]["idx"])
        t0 = time.time()
        state, mu, sd, diag = train_fold(data, tr_mice, args.hidden,
                                         args.epochs, args.patience)
        pred, p = predict_fold(X_te, state, mu, sd)
        y_te = data[test]["y"]
        all_pred.append(pred)
        y_all.append(y_te)
        m_all.append(np.full(len(pred), test))
        r = metrics(y_te, pred)
        tp = int(((pred == 1) & (y_te == 1)).sum())
        fp = int(((pred == 1) & (y_te == 0)).sum())
        fn = int(((pred == 0) & (y_te == 1)).sum())
        tn = int(((pred == 0) & (y_te == 0)).sum())
        print(f"[GRU-留一鼠] 测试 {test}: P {r['precision']:.3f} R {r['recall']:.3f} "
              f"F1 {r['f1']:.3f} κ {r['kappa']:.3f} | tp/fp/fn/tn = "
              f"{tp}/{fp}/{fn}/{tn} | 验证AUPRC {diag['val_auprc']} 阈值 {diag['thr']} "
              f"（{diag['epochs']} 轮，{time.time() - t0:.0f}s）")
        dump_rows.append(pd.DataFrame({
            "test_mouse": test, "frame_idx": data[test]["idx"],
            "y_true": y_te, "y_pred": pred, "p": np.round(p, 3)}))

    y = np.concatenate(y_all)
    mice_arr = np.concatenate(m_all)
    pred = np.concatenate(all_pred)

    rows = [{"test_mouse": m, "方法": "GRU-留一鼠",
             **{k: round(v, 3) for k, v in metrics(y[mice_arr == m], pred[mice_arr == m]).items()}}
            for m in mice]
    rows.append({"test_mouse": "汇总", "方法": "GRU-留一鼠",
                 **{k: round(v, 3) for k, v in metrics(y, pred).items()}})

    # 读回 validate_gold.py 已算好的规则法与 RF 结果，拼成三方法对比
    rf = pd.read_csv(os.path.join(OUT, "RF留一鼠_帧级.csv"))
    gru = pd.DataFrame(rows)
    comp = pd.concat([rf[rf["方法"] == "RF-留一鼠"], rf[rf["方法"] == "规则法"], gru],
                     ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    pd.concat(dump_rows, ignore_index=True).to_csv(
        os.path.join(OUT, "GRU_逐帧预测.csv"), index=False, encoding="utf-8-sig")
    gru.to_csv(os.path.join(OUT, "GRU留一鼠_帧级.csv"), index=False, encoding="utf-8-sig")
    comp.to_csv(os.path.join(OUT, "帧级_三方法对比.csv"), index=False, encoding="utf-8-sig")

    print("\n=== 帧级三方法对比（对人工金标准，LOMO） ===")
    print(comp[comp.test_mouse == "汇总"].to_string(index=False))

    # ---------------- 图：三方法对比 ----------------
    summ = comp[comp.test_mouse == "汇总"]
    methods = ["规则法", "RF-留一鼠", "GRU-留一鼠"]
    colors = ["#64748B", "#CBD5E1", "#0D9488"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    for j, (m, c) in enumerate(zip(methods, colors)):
        sub = summ[summ["方法"] == m].iloc[0]
        vals = [sub.precision, sub.recall, sub.f1]
        bars = ax.bar(x + (j - 1) * 0.26, vals, 0.26, label=m, color=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["精确率 Precision", "召回率 Recall", "F1"])
    ax.set_ylim(0, 1.08)
    ax.set_title("规则法 vs RF vs GRU 序列模型（帧级，对人工金标准，留一鼠）")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "金标准_三方法对比图.png"))
    plt.close(fig)
    print(f"\n结果已输出 → {OUT}")


if __name__ == "__main__":
    main()
