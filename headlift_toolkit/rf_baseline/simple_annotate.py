# -*- coding: utf-8 -*-
"""
简易抬头标注：大图 + 大按钮，像截图一样点一下
================================================

流程：
  1. 逐帧看视频（左右按钮 / 方向键翻页）
  2. 看到抬头起点 → 点「标记起点」
  3. 抬头结束 → 点「标记终点」（自动存一条事件）
  4. 点「跳过」忽略当前帧

也可：在草稿列表里点「是/否」快速确认。

用法：
  python simple_annotate.py --mouse 1631 --annotator 张三
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import cv2
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(BASE)
sys.path.insert(0, TOOLKIT)
import headlift as hl  # noqa: E402

VIDEO_DIR = os.path.normpath(os.path.join(TOOLKIT, "..", "video"))
OUT_CSV = os.path.join(BASE, "gold_抬头事件_v2.csv")
VIDEOS = {
    "1631": ("PBS", "1631 PBS.WMV"),
    "1632": ("PBS", "1632 PBS.WMV"),
    "1635": ("CNO", "1635 CNO.WMV"),
    "1637": ("CNO", "1637 CNO.WMV"),
    "2166": ("PBS", "ct 2166 PBS.WMV"),
    "2171": ("CNO", "ct 2171 CNO.WMV"),
}


class SimpleAnnotator:
    def __init__(self, mouse: str, annotator: str):
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk = tk
        self.Image = Image
        self.ImageTk = ImageTk

        self.mouse = mouse
        self.annotator = annotator
        self.group, video_name = VIDEOS[mouse]
        self.video_path = os.path.join(VIDEO_DIR, video_name)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {self.video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame = 0
        self.mark_start = None
        self.events = self._load_events()

        self.root = tk.Tk()
        self.root.title(f"简易标注 {mouse} ({self.group})")
        self.root.geometry("1000x720")

        self.canvas = tk.Canvas(self.root, width=960, height=520, bg="#111")
        self.canvas.pack(padx=10, pady=8)

        self.info = tk.Label(self.root, text="", font=("Microsoft YaHei", 12),
                             anchor="w")
        self.info.pack(fill="x", padx=12)

        row = tk.Frame(self.root)
        row.pack(pady=10)
        big = {"width": 14, "height": 2, "font": ("Microsoft YaHei", 12, "bold")}
        tk.Button(row, text="◀ 上一帧", command=lambda: self.go(-1), **big).pack(side="left", padx=6)
        tk.Button(row, text="下一帧 ▶", command=lambda: self.go(1), **big).pack(side="left", padx=6)
        tk.Button(row, text="标记起点", bg="#FBBF24", command=self.set_start, **big).pack(side="left", padx=6)
        tk.Button(row, text="标记终点并保存", bg="#34D399", command=self.set_end, **big).pack(side="left", padx=6)
        tk.Button(row, text="撤销起点", command=self.clear_start, **big).pack(side="left", padx=6)

        row2 = tk.Frame(self.root)
        row2.pack(pady=6)
        tk.Button(row2, text="保存退出", command=self.quit, width=12, height=2,
                  font=("Microsoft YaHei", 11)).pack(side="left", padx=6)
        tk.Button(row2, text="删除最后一条", command=self.undo_last, width=12, height=2,
                  font=("Microsoft YaHei", 11)).pack(side="left", padx=6)
        tk.Label(row2, text="方向键也可翻页", font=("Microsoft YaHei", 9), fg="#666"
                 ).pack(side="left", padx=12)

        self.root.bind("<Left>", lambda e: self.go(-5))
        self.root.bind("<Right>", lambda e: self.go(5))
        self.root.bind("<Shift-Left>", lambda e: self.go(-1))
        self.root.bind("<Shift-Right>", lambda e: self.go(1))

    def _load_events(self):
        if not os.path.exists(OUT_CSV):
            return []
        df = pd.read_csv(OUT_CSV)
        df = df[(df.mouse.astype(str) == self.mouse) & (df.annotator == self.annotator)]
        if len(df) == 0:
            return []
        evs = [{
            "onset_frame": int(r.onset_frame),
            "offset_frame": int(r.offset_frame),
            "source": str(r.source) if pd.notna(r.source) else "manual",
            "keep": True,
        } for _, r in df.iterrows()]
        evs.sort(key=lambda e: e["onset_frame"])
        print(f"[load] {len(evs)} events for {self.mouse}")
        return evs

    def save(self):
        rows = []
        for e in self.events:
            if not e.get("keep", True):
                continue
            rows.append({
                "mouse": self.mouse,
                "group": self.group,
                "onset_s": round(e["onset_frame"] / self.fps, 3),
                "offset_s": round(e["offset_frame"] / self.fps, 3),
                "onset_frame": e["onset_frame"],
                "offset_frame": e["offset_frame"],
                "source": e.get("source", "manual"),
                "annotator": self.annotator,
                "confidence": "",
                "note": "",
                "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            })
        new = pd.DataFrame(rows)
        if os.path.exists(OUT_CSV):
            old = pd.read_csv(OUT_CSV)
            old = old[~((old.mouse.astype(str) == self.mouse) &
                        (old.annotator == self.annotator))]
            df = pd.concat([old, new], ignore_index=True)
        else:
            df = new
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[save] {len(rows)} events -> {OUT_CSV}")

    def go(self, delta):
        self.frame = int(np.clip(self.frame + delta, 0, self.n - 1))
        self.draw()

    def set_start(self):
        self.mark_start = self.frame
        self.draw()

    def clear_start(self):
        self.mark_start = None
        self.draw()

    def set_end(self):
        if self.mark_start is None:
            self.mark_start = self.frame
            self.draw()
            return
        s, e = self.mark_start, self.frame
        if e < s:
            s, e = e, s
        self.events.append({
            "onset_frame": s, "offset_frame": e,
            "source": "manual_click", "keep": True,
        })
        self.events.sort(key=lambda x: x["onset_frame"])
        self.mark_start = None
        self.save()
        self.draw()

    def undo_last(self):
        if not self.events:
            return
        self.events.pop()
        self.save()
        self.draw()

    def quit(self):
        self.save()
        self.cap.release()
        self.root.quit()
        self.root.destroy()

    def draw(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
        ok, img = self.cap.read()
        if not ok:
            img = np.zeros((288, 704, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 画已有事件覆盖
        h, w = img.shape[:2]
        scale = min(960 / w, 500 / h)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # 当前帧是否在某事件内
        for e in self.events:
            if e["onset_frame"] <= self.frame <= e["offset_frame"]:
                cv2.rectangle(img, (0, 0), (nw - 1, nh - 1), (52, 211, 153), 6)
                cv2.putText(img, "IN EVENT", (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (52, 211, 153), 2)
                break
        if self.mark_start is not None:
            cv2.putText(img, f"START marked @ {self.mark_start}",
                        (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (251, 191, 36), 2)

        pil = self.Image.fromarray(img)
        self._photo = self.ImageTk.PhotoImage(image=pil)
        self.canvas.delete("all")
        self.canvas.create_image(480, 260, image=self._photo)

        t = self.frame / self.fps
        mm, ss = int(t) // 60, t % 60
        n_ev = sum(1 for e in self.events if e.get("keep", True))
        self.info.config(
            text=f"帧 {self.frame}/{self.n}  |  时间 {mm}:{ss:04.1f}s  |  "
                 f"已标事件 {n_ev} 条  |  "
                 + (f"起点已标记 {self.mark_start}，点「标记终点并保存」" if self.mark_start is not None
                    else "点「标记起点」开始一次抬头")
        )

    def run(self):
        print(f"[start] {mouse_label(self.mouse)} frames={self.n} fps={self.fps:.1f}")
        print(f"[events] already {len(self.events)}")
        self.draw()
        self.root.mainloop()


def mouse_label(m):
    return f"{m} {VIDEOS[m][0]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True, choices=list(VIDEOS))
    ap.add_argument("--annotator", required=True)
    args = ap.parse_args()
    SimpleAnnotator(args.mouse, args.annotator).run()


if __name__ == "__main__":
    main()
