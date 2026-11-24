# -*- coding: utf-8 -*-
"""
全长抬头事件标注 GUI（tkinter 版：系统草稿 + 人工仲裁）
=====================================================

用法：
  python event_annotate_gui.py --mouse 1631 --annotator 张三
  python event_annotate_gui.py --all --annotator 张三

按键：
  空格 / →     下一事件
  ←            上一事件
  a / A        起点 -1 / -5 帧
  d / D        起点 +1 / +5 帧
  z / Z        终点 -1 / -5 帧
  c / C        终点 +1 / +5 帧
  x            删除当前事件
  n            在当前帧新建事件
  m            合并当前与下一条
  s            保存
  q            保存并退出当前鼠（--all 时进入下一只）
  g            跳到指定秒

输出：rf_baseline/gold_抬头事件_v2.csv
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
DEFAULT_K = 1.25


def draft_events(mouse: str) -> list[dict]:
    """生产草稿：鼻尖高于头顶 ≥5px，持续 ≥0.35s（明确大抬头）。"""
    _, video_name = VIDEOS[mouse]
    csv_candidates = [
        f for f in os.listdir(VIDEO_DIR)
        if mouse in f and f.endswith(".csv") and "DLC" in f
    ]
    if not csv_candidates:
        print(f"[draft] mouse {mouse}: no DLC CSV -> manual only")
        return []
    csv_path = os.path.join(VIDEO_DIR, csv_candidates[0])
    video_path = os.path.join(VIDEO_DIR, video_name)
    df = hl.load_dlc_csv(csv_path)
    fps = hl.read_fps(video_path)
    ev, _ = hl.detect_lifts_yrule_df(df, fps=fps)
    out = []
    for _, r in ev.iterrows():
        out.append({
            "onset_frame": int(r.onset_frame),
            "offset_frame": int(r.offset_frame),
            "source": "system_draft",
            "keep": True,
        })
    return out


class EventAnnotator:
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
        self.events = draft_events(mouse)
        self.idx = 0 if self.events else -1
        self.frame = 0
        self._load_saved()

        self.root = tk.Tk()
        self.root.title(f"事件标注 {mouse} ({self.group}) | {annotator}")
        self.root.geometry("1100x640")
        self.canvas = tk.Canvas(self.root, width=1080, height=560, bg="#111")
        self.canvas.pack(padx=8, pady=6)
        self.status = tk.Label(self.root, text="", anchor="w", font=("Microsoft YaHei", 10),
                               justify="left")
        self.status.pack(fill="x", padx=8, pady=(0, 6))
        self.help = tk.Label(
            self.root,
            text="Space/←→ 切换 | a/d 起点 | z/c 终点 | Shift=5帧 | x 删 | n 新建 | m 合并 | s 保存 | q 退出 | g 跳秒",
            anchor="w", font=("Microsoft YaHei", 9), fg="#555",
        )
        self.help.pack(fill="x", padx=8, pady=(0, 4))

        # 按钮兜底（中文输入法可能吞掉 z/x 等键）
        btnbar = tk.Frame(self.root)
        btnbar.pack(fill="x", padx=8, pady=(0, 8))
        for text, cmd in [
            ("起点-", lambda: self.nudge("onset", -1)),
            ("起点+", lambda: self.nudge("onset", 1)),
            ("终点-（收后半截）", lambda: self.nudge("offset", -5)),
            ("终点+", lambda: self.nudge("offset", 5)),
            ("删除", self.delete_ev),
            ("新建", self.new_ev),
            ("保存", self.save_quiet),
        ]:
            tk.Button(btnbar, text=text, command=cmd).pack(side="left", padx=3)

        for key, fn in [
            ("<space>", self.next_ev), ("<Right>", self.next_ev),
            ("<Left>", self.prev_ev),
            ("a", lambda e: self.nudge("onset", -1)),
            ("A", lambda e: self.nudge("onset", -5)),
            ("d", lambda e: self.nudge("onset", 1)),
            ("D", lambda e: self.nudge("onset", 5)),
            ("z", lambda e: self.nudge("offset", -1)),
            ("Z", lambda e: self.nudge("offset", -5)),
            ("c", lambda e: self.nudge("offset", 1)),
            ("C", lambda e: self.nudge("offset", 5)),
            ("x", self.delete_ev), ("n", self.new_ev), ("m", self.merge_ev),
            ("s", self.save_quiet), ("q", self.quit_all), ("g", self.goto_sec),
        ]:
            self.root.bind(key, fn)
            self.canvas.bind(key, fn)
        self.root.after(200, self._focus)

    def _focus(self):
        try:
            self.root.focus_force()
            self.canvas.focus_set()
        except Exception:
            pass

    def _load_saved(self):
        if not os.path.exists(OUT_CSV):
            return
        df = pd.read_csv(OUT_CSV)
        df = df[(df.mouse.astype(str) == self.mouse) & (df.annotator == self.annotator)]
        if len(df) == 0:
            return
        self.events = [{
            "onset_frame": int(r.onset_frame),
            "offset_frame": int(r.offset_frame),
            "source": str(r.source) if pd.notna(r.source) else "manual",
            "keep": True,
        } for _, r in df.iterrows()]
        self.events.sort(key=lambda e: e["onset_frame"])
        self.idx = 0
        print(f"[load] {len(self.events)} events for {self.mouse}/{self.annotator}")

    def read_frame(self, f: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(f, 0, self.n - 1)))
        ok, img = self.cap.read()
        return img if ok else None

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

    def kept(self):
        return [e for e in self.events if e.get("keep", True)]

    def draw(self):
        cur = self.events[self.idx] if 0 <= self.idx < len(self.events) else None
        f = cur["onset_frame"] if cur else self.frame
        f = int(np.clip(f, 0, self.n - 1))
        self.frame = f
        img = self.read_frame(f)
        if img is None:
            img = np.zeros((288, 704, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 简单缩放到画布宽度
        h, w = img.shape[:2]
        scale = min(1080 / w, 540 / h)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        # 事件条
        bar_h = 14
        canvas_w = nw
        for i, e in enumerate(self.events):
            if not e.get("keep", True):
                continue
            x1 = int(e["onset_frame"] / max(self.n - 1, 1) * (canvas_w - 1))
            x2 = int(e["offset_frame"] / max(self.n - 1, 1) * (canvas_w - 1))
            color = (220, 60, 60) if i == self.idx else (255, 140, 0)
            cv2.rectangle(img, (x1, 0), (max(x2, x1 + 2), bar_h), color, -1)
        xf = int(f / max(self.n - 1, 1) * (canvas_w - 1))
        cv2.line(img, (xf, 0), (xf, nh - 1), (255, 255, 0), 1)

        pil = self.Image.fromarray(img)
        self._photo = self.ImageTk.PhotoImage(image=pil)
        self.canvas.delete("all")
        self.canvas.create_image(540, 280, image=self._photo)

        n_keep = len(self.kept())
        t = f / self.fps
        if cur:
            line = (
                f"{self.mouse} | t={t:6.2f}s ({int(t)//60}:{int(t)%60:02d}) frame={f}/{self.n} | "
                f"事件 {self.idx+1 if self.idx>=0 else 0}/{len(self.events)} (有效{n_keep}) | "
                f"onset={cur['onset_frame']} ({cur['onset_frame']/self.fps:.2f}s) "
                f"offset={cur['offset_frame']} ({cur['offset_frame']/self.fps:.2f}s) "
                f"dur={(cur['offset_frame']-cur['onset_frame']+1)/self.fps:.2f}s "
                f"src={cur.get('source')}"
            )
        else:
            line = f"{self.mouse} | t={t:6.2f}s | 无事件，按 n 新建"
        self.status.config(text=line)

    # ---- actions ----
    def next_ev(self, _e=None):
        if self.idx < len(self.events) - 1:
            self.idx += 1
        self.draw()

    def prev_ev(self, _e=None):
        if self.idx > 0:
            self.idx -= 1
        self.draw()

    def nudge(self, which, delta):
        if self.idx < 0 or self.idx >= len(self.events):
            return
        e = self.events[self.idx]
        if which == "onset":
            e["onset_frame"] = int(np.clip(e["onset_frame"] + delta, 0, e["offset_frame"]))
        else:
            e["offset_frame"] = int(np.clip(e["offset_frame"] + delta, e["onset_frame"], self.n - 1))
        e["source"] = "manual_edit"
        self.draw()

    def delete_ev(self, _e=None):
        if self.idx < 0:
            return
        self.events[self.idx]["keep"] = False
        # 跳到下一条 keep
        for i in range(self.idx + 1, len(self.events)):
            if self.events[i].get("keep", True):
                self.idx = i
                break
        else:
            for i in range(self.idx - 1, -1, -1):
                if self.events[i].get("keep", True):
                    self.idx = i
                    break
            else:
                self.idx = -1
        self.draw()

    def new_ev(self, _e=None):
        onset = self.frame
        offset = min(self.n - 1, onset + int(0.5 * self.fps))
        self.events.append({
            "onset_frame": onset, "offset_frame": offset,
            "source": "manual_new", "keep": True,
        })
        self.events.sort(key=lambda e: e["onset_frame"])
        self.idx = next(
            i for i, e in enumerate(self.events)
            if e["onset_frame"] == onset and e["offset_frame"] == offset and e.get("keep", True)
        )
        self.draw()

    def merge_ev(self, _e=None):
        if self.idx < 0 or self.idx >= len(self.events) - 1:
            return
        a, b = self.events[self.idx], self.events[self.idx + 1]
        a["offset_frame"] = max(a["offset_frame"], b["offset_frame"])
        a["source"] = "manual_merge"
        b["keep"] = False
        self.draw()

    def save_quiet(self, _e=None):
        self.save()

    def goto_sec(self, _e=None):
        def ok():
            try:
                sec = float(ent.get())
                self.frame = int(np.clip(sec * self.fps, 0, self.n - 1))
                self.idx = -1
                self.draw()
            except Exception:
                pass
            top.destroy()
        top = self.tk.Toplevel(self.root)
        top.title("跳转到秒")
        ent = self.tk.Entry(top)
        ent.pack(side="left", padx=8, pady=8)
        self.tk.Button(top, text="Go", command=ok).pack(side="left", padx=8)
        ent.focus_set()

    def quit_all(self, _e=None):
        self.save()
        self.cap.release()
        self.root.quit()
        self.root.destroy()

    def run(self):
        print(f"[start] {self.mouse} draft={len(self.events)} fps={self.fps:.2f} frames={self.n}")
        self.draw()
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="1631")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--annotator", required=True)
    args = ap.parse_args()
    mice = list(VIDEOS.keys()) if args.all else [args.mouse]
    for m in mice:
        EventAnnotator(m, args.annotator).run()


if __name__ == "__main__":
    main()
