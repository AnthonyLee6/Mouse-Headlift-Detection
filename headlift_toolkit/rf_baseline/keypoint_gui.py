# -*- coding: utf-8 -*-
"""
关键点点击标注 GUI（姿态模型人工真值采集）
============================================

对 keypoint_manifest.csv 抽出的 120 帧，逐帧按顺序点击 4 个关键点：
  鼻尖 → 头顶 → 左耳 → 右耳
点完第 4 个点自动保存并跳下一帧。结果写 keypoint_labels/keypoint_labels_<标注人>.csv，
可中断续标。

按键/操作：
  鼠标左键   按顺序点击 4 个关键点
  r          重置本帧重标
  ←          上一帧（修改）
  s / q      保存 / 保存并退出

用法：python keypoint_gui.py --annotator 你的名字
"""
import argparse
import datetime
import os

import cv2
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "keypoint_cache")
LABELS_DIR = os.path.join(BASE, "keypoint_labels")
POINTS = [("nose", "鼻尖", (255, 0, 255)), ("head", "头顶", (0, 255, 255)),
          ("left_ear", "左耳", (255, 255, 0)), ("right_ear", "右耳", (255, 0, 255))]


class KeypointLabeler:
    def __init__(self, annotator, redo=False):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.annotator = annotator

        self.manifest = pd.read_csv(os.path.join(BASE, "keypoint_manifest.csv"))
        self.labels_path = os.path.join(LABELS_DIR, f"keypoint_labels_{annotator}.csv")
        self.done = self._load_done()
        if redo:
            self.done = self.done.iloc[0:0]
        labeled = set(zip(self.done.mouse.astype(str), self.done.frame_idx)) \
            if len(self.done) else set()
        self.pending = self.manifest[
            ~self.manifest.apply(lambda r: (str(r.mouse), r.frame_idx) in labeled, axis=1)
        ].reset_index(drop=True)
        self.i = 0
        self.clicks = []  # [(x, y)] 原图坐标

        if len(self.pending) == 0:
            print(f"[OK] All {len(self.manifest)} frames already labeled, nothing to do."
                  f"\n     file: {self.labels_path}"
                  f"\n     To relabel from scratch, add --redo")
            raise SystemExit(0)

        self.root = tk.Tk()
        self.root.title(f"关键点标注 — {annotator} | 待标 {len(self.pending)} 帧")
        self.canvas = tk.Canvas(self.root, width=1408, height=576, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.info = tk.Label(self.root, font=("Microsoft YaHei", 13), justify="left")
        self.info.pack(anchor="w", padx=8, pady=4)
        for key, fn in [("r", self.reset_frame), ("<Left>", self.back),
                        ("s", self.save), ("q", self.quit)]:
            self.root.bind(key, lambda e, f=fn: f())
        self._photo = None
        self.show()

    def _load_done(self):
        if os.path.exists(self.labels_path):
            df = pd.read_csv(self.labels_path)
            return df[df.annotator == self.annotator]
        return pd.DataFrame()

    # ---------------- 流程 ----------------
    def on_click(self, event):
        x, y = event.x / 2, event.y / 2  # 画布是原图 ×2
        if len(self.clicks) >= 4:
            return
        self.clicks.append((round(x, 1), round(y, 1)))
        if len(self.clicks) == 4:
            self.record()
        self.show()

    def record(self):
        row = self.pending.iloc[self.i]
        rec = {"mouse": row.mouse, "group": row.group, "frame_idx": int(row.frame_idx),
               "annotator": self.annotator,
               "labeled_at": datetime.datetime.now().isoformat(timespec="seconds")}
        for (name, _, _), (x, y) in zip(POINTS, self.clicks):
            rec[f"{name}_x"], rec[f"{name}_y"] = x, y
        self.done = pd.concat([self.done, pd.DataFrame([rec])], ignore_index=True)
        self.done.drop_duplicates(["mouse", "frame_idx"], keep="last", inplace=True)
        self.save(quiet=True)
        self.clicks = []
        self.i += 1
        if self.i >= len(self.pending):
            self.save()
            print("关键点标注完成，感谢！")
            self.root.destroy()
            return
        self.show()

    def reset_frame(self):
        self.clicks = []
        self.show()

    def back(self):
        if self.i > 0:
            self.i -= 1
            self.clicks = []
            self.show()

    def save(self, quiet=False):
        os.makedirs(LABELS_DIR, exist_ok=True)
        self.done.to_csv(self.labels_path, index=False, encoding="utf-8-sig")
        if not quiet:
            print(f"已保存 {len(self.done)} 帧 → {self.labels_path}")

    def quit(self):
        self.save()
        self.root.destroy()

    # ---------------- 显示 ----------------
    def show(self):
        row = self.pending.iloc[self.i]
        p = os.path.join(CACHE, str(row.mouse), f"{int(row.frame_idx):06d}.jpg")
        with open(p, "rb") as fh:
            img = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                         interpolation=cv2.INTER_NEAREST)
        for (x, y), (name, cn, color) in zip(self.clicks, POINTS):
            cv2.circle(img, (int(x * 2), int(y * 2)), 7, color, 2, cv2.LINE_AA)
            cv2.putText(img, cn, (int(x * 2) + 10, int(y * 2) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if len(self.clicks) >= 2:  # 鼻尖→头顶连线
            cv2.line(img, (int(self.clicks[0][0] * 2), int(self.clicks[0][1] * 2)),
                     (int(self.clicks[1][0] * 2), int(self.clicks[1][1] * 2)),
                     (255, 255, 255), 1, cv2.LINE_AA)
        self._photo = self.ImageTk.PhotoImage(self.Image.fromarray(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        nxt = POINTS[len(self.clicks)][1] if len(self.clicks) < 4 else "完成→自动下一帧"
        self.info.config(text=(
            f"小鼠 {row.mouse}（{row.group}）  帧 {row.frame_idx}  t={row.time_s}s  "
            f"分层 {row.stratum}    进度 {self.i + 1}/{len(self.pending)}（已入库 {len(self.done)}）\n"
            f"当前点击顺序：① 鼻尖 → ② 头顶 → ③ 左耳 → ④ 右耳    下一个点：{nxt}    "
            f"[r]重置本帧  [←]上一帧  [s]保存  [q]保存退出"))
        self.root.title(f"{row.mouse} | 帧 {row.frame_idx} | 已点 {len(self.clicks)}/4")

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    KeypointLabeler(args.annotator, redo=args.redo).run()


if __name__ == "__main__":
    main()
