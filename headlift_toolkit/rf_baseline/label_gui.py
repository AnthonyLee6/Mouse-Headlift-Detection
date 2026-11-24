# -*- coding: utf-8 -*-
"""
抬头/非抬头 逐帧标注 GUI（v2：双法交叉草稿 + 优先级队列 + 运动上下文）
======================================================================

读 annotation_manifest.csv（sample_frames.py 产出），逐帧显示视频画面 + 关键点 +
草稿建议，人工确认或纠正。标注结果实时写入 labels/labels_<标注人>.csv，
可随时中断、随时续标（自动跳过已标帧）。

v2 新增：
  1. 第二意见：y 坐标法（detect_lifts_yrule，时长口径与角度法一致 0.15s）逐帧判定，
     与规则法（清单 suggestion）不一致的帧标"⚠ 重点帧"——这些帧人工价值最高；
  2. 优先级队列（--order priority，默认）：每只鼠内先标"两法不一致"帧，再按分层
     B>A>C>D>E 排序；--order file 恢复按帧号顺序；
  3. 运动上下文：主图下方显示 ±0.5s 的 4 张邻帧缩略图（近似 seek，仅供运动方向参考），
     [p] 可循环播放 ±0.75s 微片段，任意键停止（停止键不会被误记为标签）；
  4. 进度统计：本会话速度（帧/分）与预计剩余时间；
  5. 标签 CSV 新增 suggestion/yrule/elapsed_s 列，兼容 train_rf.py（只用 label 列）。

按键：
  1        该帧 = 抬头
  0        该帧 = 非抬头
  9        无法判断（关键点明显飞了/严重遮挡），训练时丢弃
  Enter    接受规则法建议（建议正确时的快进键）   Space 同 Enter
  p        播放当前帧 ±0.75s 微片段（任意键停止）
  j        跳到下一帧"两法不一致"的待标帧
  ←        上一帧（可改判）
  s        保存
  q        保存并退出

用法：
  python label_gui.py --annotator 张三               # 优先级队列（默认）
  python label_gui.py --annotator 张三 --order file  # 按帧号顺序
  python label_gui.py --annotator 张三 --redo        # 忽略已有标注从头再标
"""
import argparse
import os
import sys
import datetime
import time

import cv2
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(BASE)
sys.path.insert(0, TOOLKIT)
import headlift as hl  # noqa: E402
VIDEO_DIR = os.path.normpath(os.path.join(TOOLKIT, "..", "video"))
VIDEOS = {
    # mouse: (组别, DLC CSV, 原始视频)
    "1631": ("PBS", "1631 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1631 PBS.WMV"),
    "1632": ("PBS", "1632 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1632 PBS.WMV"),
    "1635": ("CNO", "1635 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1635 CNO.WMV"),
    "1637": ("CNO", "1637 CNODLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv", "1637 CNO.WMV"),
}
CACHE = os.path.join(BASE, "frame_cache")
LABELS_DIR = os.path.join(BASE, "labels")
# 与 final_analyze 图例一致：紫=鼻尖 黄=头顶 青=左耳 粉=右耳
KEYPOINTS = [("nose_x", "nose_y", "nose_lh", (255, 0, 255)),
             ("head_x", "head_y", "head_lh", (0, 255, 255)),
             ("left_ear_x", "left_ear_y", "left_ear_lh", (255, 255, 0)),
             ("right_ear_x", "right_ear_y", "right_ear_lh", (255, 0, 255))]
STRATA_RANK = {"B_阈值附近": 1, "A_事件内": 2, "C_高角速度": 3, "D_低置信度": 4, "E_随机背景": 5}
MIN_DUR_S = 0.15  # 与规则法最短持续口径一致
THUMB_OFFSETS = [-10, -5, 5, 10]  # 上下文缩略图偏移（帧）


# ================================================================ 草稿意见（GUI 与报告共用）
def load_kps(mouse):
    _, csv_name, _ = VIDEOS[str(mouse)]
    return hl.load_dlc_csv(os.path.join(VIDEO_DIR, csv_name))


def yrule_flags_for_mouse(mouse):
    """y 坐标法逐帧判定（0/1 数组）。时长口径与角度法一致：连续 >=0.15s 才算抬头。"""
    df = load_kps(mouse)
    fps = hl.read_fps(os.path.join(VIDEO_DIR, VIDEOS[str(mouse)][2]))
    min_frames = max(1, int(round(MIN_DUR_S * fps)))
    _, flags = hl.detect_lifts_yrule(df["nose_y"].to_numpy(), df["head_y"].to_numpy(),
                                     min_frames=min_frames, fps=fps)
    return flags


def add_second_opinion(manifest):
    """给清单补 yrule（y 坐标法判定）与 disagree（两法不一致）两列。"""
    yf = {m: yrule_flags_for_mouse(m) for m in manifest["mouse"].astype(str).unique()}
    yrule, n_over = [], 0
    for r in manifest.itertuples():
        f = int(r.frame_idx)
        flags = yf[str(r.mouse)]
        yrule.append(int(bool(flags[f])) if f < len(flags) else 0)
    manifest = manifest.copy()
    manifest["yrule"] = yrule
    manifest["disagree"] = (manifest["suggestion"] != manifest["yrule"]).astype(int)
    return manifest


# ================================================================ 标注界面
def load_frame(mouse, frame_idx):
    """优先读帧缓存 JPEG（帧号精确），缺失时退回视频 seek（可能差几帧）。"""
    p = os.path.join(CACHE, str(mouse), f"{int(frame_idx):06d}.jpg")
    if os.path.exists(p):
        with open(p, "rb") as fh:
            img = cv2.imdecode(np.frombuffer(fh.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img, True
    video = os.path.join(VIDEO_DIR, VIDEOS[str(mouse)])
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, img = cap.read()
    cap.release()
    return (img, False) if ret else (None, False)


class Labeler:
    def __init__(self, annotator, redo=False, order="priority", smoke=False):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk

        self.annotator = annotator
        self.smoke = smoke
        manifest = pd.read_csv(os.path.join(BASE, "annotation_manifest.csv"))
        self.manifest = add_second_opinion(manifest)
        if order == "priority":
            self.manifest["_rank"] = self.manifest["stratum"].map(STRATA_RANK).fillna(9)
            self.manifest = self.manifest.sort_values(
                ["mouse", "disagree", "_rank", "frame_idx"],
                ascending=[True, False, True, True]).drop(columns="_rank")
        else:
            self.manifest = self.manifest.sort_values(["mouse", "frame_idx"])
        self.manifest = self.manifest.reset_index(drop=True)

        self.labels_path = os.path.join(LABELS_DIR, f"labels_{annotator}.csv")
        self.done = self._load_done()
        if redo:
            self.done = self.done[[]]
        done_keys = set(map(tuple, self.done[["mouse", "frame_idx"]].to_numpy())) if len(self.done) else set()
        self.pending = self.manifest[
            ~self.manifest.apply(lambda r: (r.mouse, r.frame_idx) in done_keys, axis=1)
        ].reset_index(drop=True)
        self.i = 0
        self._kps = {}      # mouse -> DLC 关键点 DataFrame（懒加载）
        self._caps = {}     # mouse -> VideoCapture（懒加载，用于上下文帧）
        self._fps = {}      # mouse -> fps
        self._playing, self._play, self._play_job = False, None, None
        self.t0 = time.time()
        self.n_session = 0
        self._t_show = time.time()

        self.root = tk.Tk()
        self.root.title(f"抬头标注 — {annotator} | 待标 {len(self.pending)} 帧")
        self.canvas = tk.Canvas(self.root, width=1408, height=576)
        self.canvas.pack()
        self.strip = tk.Canvas(self.root, width=1408, height=162, highlightthickness=0)
        self.strip.pack()
        self.info = tk.Label(self.root, font=("Microsoft YaHei", 13), justify="left")
        self.info.pack(anchor="w", padx=8, pady=4)
        actions = [("1", lambda: self.record(1)), ("0", lambda: self.record(0)),
                   ("9", lambda: self.record(9)), ("<Return>", self.accept),
                   ("<space>", self.accept), ("<Key-p>", self.play),
                   ("<Key-j>", self.jump_disagree), ("<Left>", self.back),
                   ("s", self.save), ("q", self.quit)]
        for key, fn in actions:
            self.root.bind(key, lambda e, f=fn: f())
        self.root.bind("<Key>", self._on_any_key)  # 未绑定键也可停止播放
        self._photo = None
        self._strip_photos = []
        self.show()
        if smoke:
            self.root.after(1500, self.root.destroy)

    # ---------------- 标注记录 ----------------
    def _load_done(self):
        if os.path.exists(self.labels_path):
            df = pd.read_csv(self.labels_path)
            return df[df["annotator"] == self.annotator]
        return pd.DataFrame(columns=["mouse", "frame_idx", "label", "annotator"])

    def _on_any_key(self, event):
        if self._playing:
            self._stop_play()

    def _guard_playing(self, fn):
        """播放中先停（播放键不复播），不把停止键误记为标签。"""
        if self._playing:
            self._stop_play()
            return
        fn()

    def record(self, label):
        def go():
            row = self.pending.iloc[self.i]
            elapsed = min(600.0, max(0.0, time.time() - self._t_show))
            rec = {"mouse": row.mouse, "frame_idx": int(row.frame_idx), "label": label,
                   "annotator": self.annotator, "suggestion": int(row.suggestion),
                   "yrule": int(row.yrule), "elapsed_s": round(elapsed, 2),
                   "labeled_at": datetime.datetime.now().isoformat(timespec="seconds")}
            self.done = pd.concat([self.done, pd.DataFrame([rec])], ignore_index=True)
            self.done.drop_duplicates(["mouse", "frame_idx"], keep="last", inplace=True)
            self.save(quiet=True)
            self.n_session += 1
            self.i += 1
            if self.i >= len(self.pending):
                self.save()
                print("全部帧已标注完毕，感谢！")
                self.root.destroy()
                return
            self.show()
        self._guard_playing(go)

    def accept(self):
        row = self.pending.iloc[self.i]
        self.record(int(row.suggestion))

    def back(self):
        def go():
            if self.i > 0:
                self.i -= 1
                self.show()
        self._guard_playing(go)

    def jump_disagree(self):
        def go():
            rest = self.pending.iloc[self.i + 1:]
            hit = rest.index[rest["disagree"] == 1]
            if len(hit):
                self.i = int(hit[0])
                self.show()
            else:
                self.root.title("后面没有“两法不一致”的待标帧了")
        self._guard_playing(go)

    def save(self, quiet=False):
        if self.smoke:
            return
        os.makedirs(LABELS_DIR, exist_ok=True)
        self.done.to_csv(self.labels_path, index=False, encoding="utf-8-sig")
        if not quiet:
            print(f"已保存 {len(self.done)} 条标注 → {self.labels_path}")

    def quit(self):
        def go():
            self.save()
            self.root.destroy()
        self._guard_playing(go)

    # ---------------- 视频上下文（近似 seek，仅供运动参考） ----------------
    def _cap(self, mouse):
        if mouse not in self._caps:
            from sample_frames import open_video_fallback
            self._caps[mouse] = open_video_fallback(
                os.path.join(VIDEO_DIR, VIDEOS[str(mouse)][2]))
        return self._caps[mouse]

    def _fps_of(self, mouse):
        if mouse not in self._fps:
            self._fps[mouse] = hl.read_fps(os.path.join(VIDEO_DIR, VIDEOS[str(mouse)][2]))
        return self._fps[mouse]

    def _window(self, mouse, f, before, after):
        """解码 [f-before, f-after] 的帧（WMV seek 有 ±几帧误差，仅作上下文）。"""
        try:
            cap = self._cap(mouse)
        except Exception:
            return [], f
        start = max(0, int(f) - before)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(before + after + 1):
            ret, img = cap.read()
            if not ret:
                break
            frames.append(img)
        return frames, start

    def _thumb_at(self, frames, start, f, off):
        j = f + off - start
        return frames[j] if 0 <= j < len(frames) else None

    def draw_strip(self, row):
        f, mouse = int(row.frame_idx), str(row.mouse)
        fps = self._fps_of(mouse)
        frames, start = self._window(mouse, f, before=max(THUMB_OFFSETS) + 2,
                                     after=max(THUMB_OFFSETS) + 2)
        self.strip.delete("all")
        self._strip_photos = []  # PhotoImage 须持有引用，否则被 GC 后画布空白
        w_thumb = 1408 // len(THUMB_OFFSETS)
        for k, off in enumerate(THUMB_OFFSETS):
            x0 = k * w_thumb
            img = self._thumb_at(frames, start, f, off)
            if img is None:
                self.strip.create_rectangle(x0, 0, x0 + w_thumb, 146, fill="#333",
                                            outline="#555")
                self.strip.create_text(x0 + w_thumb // 2, 73, fill="#aaa",
                                       text=f"{off / fps:+.2f}s\n缺帧", font=("Arial", 11))
                continue
            h, w = img.shape[:2]
            img = cv2.resize(img, (w_thumb, int(h * w_thumb / w)),
                             interpolation=cv2.INTER_AREA)
            photo = self.ImageTk.PhotoImage(self.Image.fromarray(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            self._strip_photos.append(photo)
            self.strip.create_image(x0, 0, image=photo, anchor="nw")
            self.strip.create_text(x0 + 8, 6, anchor="nw", fill="#ffdd00",
                                   text=f"≈{off / fps:+.2f}s", font=("Arial", 11, "bold"))
        self.strip.create_text(1408 // 2, 152, fill="#bbbbbb",
                               text="运动上下文（近似 seek，仅供判断运动方向；标注只看主图帧）",
                               font=("Microsoft YaHei", 9))

    # ---------------- 微片段播放 ----------------
    def play(self):
        def go():
            row = self.pending.iloc[self.i]
            mouse, f = str(row.mouse), int(row.frame_idx)
            fps = self._fps_of(mouse)
            half = max(1, int(round(0.75 * fps)))
            frames, start = self._window(mouse, f, before=half, after=half)
            if not frames:
                self.root.title("无法打开视频，播放不可用")
                return
            self._playing = True
            self._play = {"frames": frames, "start": start, "fps": fps, "idx": 0}
            self.info.config(text="▶ 播放中……  任意键停止（不会误记标签）")
            self._play_step()
        self._guard_playing(go)

    def _play_step(self):
        if not self._playing or not self._play:
            return
        p = self._play
        img = p["frames"][p["idx"] % len(p["frames"])]
        p["idx"] += 1
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        self._photo = self.ImageTk.PhotoImage(self.Image.fromarray(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self._play_job = self.root.after(int(1000 / p["fps"]), self._play_step)

    def _stop_play(self):
        self._playing = False
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None
        self._play = None
        self.show()

    # ---------------- 显示 ----------------
    def _keypoints(self, mouse):
        if mouse not in self._kps:
            self._kps[mouse] = load_kps(mouse)
        return self._kps[mouse]

    def show(self):
        self._t_show = time.time()
        row = self.pending.iloc[self.i]
        img, exact = load_frame(row.mouse, row.frame_idx)
        if img is None:
            img = np.zeros((288, 704, 3), np.uint8)
        h, w = img.shape[:2]
        kps = self._keypoints(row.mouse)
        f = int(row.frame_idx)
        # 原图坐标绘制关键点（置信度不足画空心圈提示不可靠），随后整体 ×2 显示
        nose_xy, head_xy = None, None
        for xc, yc, lc, color in KEYPOINTS:
            if f >= len(kps):
                continue
            x, y, l = kps[xc].iloc[f], kps[yc].iloc[f], kps[lc].iloc[f]
            if pd.isna(x) or x < 0:
                continue
            if xc == "nose_x":
                nose_xy = (int(x), int(y))
            if xc == "head_x":
                head_xy = (int(x), int(y))
            if l < 0.6:
                cv2.circle(img, (int(x), int(y)), 5, color, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(x), int(y)), 4, color, 2, cv2.LINE_AA)
        if nose_xy and head_xy:
            cv2.line(img, nose_xy, head_xy, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        self._photo = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.draw_strip(row)

        # 进度统计
        done_min = (time.time() - self.t0) / 60
        rate = self.n_session / done_min if done_min > 0.2 else 0
        remain = len(self.pending) - self.i - 1
        eta = f"{remain / rate:.0f} 分钟" if rate >= 1 else "—"
        n_dis_left = int(self.pending.iloc[self.i:]["disagree"].sum())

        sug_a = "抬头" if row.suggestion == 1 else "非抬头"
        sug_b = "抬头" if row.yrule == 1 else "非抬头"
        dis = "⚠ 两法不一致 → 重点判断" if row.disagree == 1 else "两法一致"
        self.info.config(text=(
            f"小鼠 {row.mouse}（{row.group}）  帧 {row.frame_idx}  "
            f"t={row.time_s}s  角度 {row.ang}°  阈值 {row.thr}°\n"
            f"建议A 规则法：{sug_a} | 建议B y坐标法：{sug_b} | {dis}    "
            f"分层：{row.stratum}    {'帧号精确' if exact else '⚠ seek 可能偏帧'}\n"
            f"进度 {self.i + 1}/{len(self.pending)}（已入库 {len(self.done)}）  "
            f"本会话 {rate:.0f} 帧/分  预计剩余 {eta}  未标不一致帧 {n_dis_left}\n"
            f"[1]抬头 [0]非抬头 [9]无法判断 [Enter/Space]接受建议A [p]播放 [j]下一不一致帧 "
            f"[←]上一帧 [s]保存 [q]保存退出"
        ))
        self.root.title(f"{row.mouse} | 帧 {row.frame_idx} | 建议A：{sug_a} | 建议B：{sug_b}"
                        + (" | ⚠不一致" if row.disagree == 1 else ""))

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True, help="标注人姓名（写入文件名）")
    ap.add_argument("--redo", action="store_true", help="忽略已有标注，从头再标")
    ap.add_argument("--order", choices=["priority", "file"], default="priority",
                    help="priority=不一致帧优先（默认）；file=按帧号顺序")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：显示首帧后自动退出，不写文件")
    args = ap.parse_args()
    Labeler(args.annotator, redo=args.redo, order=args.order, smoke=args.smoke).run()


if __name__ == "__main__":
    main()
