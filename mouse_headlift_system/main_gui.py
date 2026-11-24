# -*- coding: utf-8 -*-
"""
小鼠抬头行为自动识别分析系统 V1.0 —— 主程序（GUI / 命令行双模式）
==================================================================

图形界面模式（CustomTkinter 现代深色界面）：
    python main_gui.py [--folder <数据文件夹>] [--out <输出文件夹>]
命令行批处理模式（无界面，适合服务器/定时任务）：
    python main_gui.py --batch <数据文件夹> [--out <输出文件夹>]

数据要求：文件夹内存放 DeepLabCut 输出的关键点 CSV（可含配对视频）。
也可只放视频：勾选「自动生成关键点」后，系统用已训练 DLC 模型先推理坐标再检测。
系统按文件名前缀自动配对视频与 CSV，按文件名自动识别 PBS / CNO 分组，
一键完成批量检测并输出 Excel 报表（检测总览 / 事件明细 / 参数 / 统计图）、
俯仰角轨迹图、组间对比图，可选裁剪抬头片段与生成计数标注视频。
"""
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import pandas as pd

import headlift_core as hl
import report as rp
import pose_infer

APP_NAME = "小鼠抬头行为自动识别分析系统"
VERSION = "V1.0"

# 界面主题（深色：炭灰底 + 青绿主色，避免纯黑）
ACCENT = "#14B8A6"
ACCENT_HOVER = "#0D9488"
ACCENT_SOFT = "#0F766E"
BG = ("#F3F6FA", "#0F1419")
CARD = ("#FFFFFF", "#1A222C")
CARD_EDGE = ("#E2E8F0", "#2A3542")
TEXT = ("#0F172A", "#E8EEF5")
TEXT_MUTED = ("#64748B", "#9AA8B8")
FIELD_BG = ("#F8FAFC", "#243040")
FIELD_H = 38
BTN_H = 40
FONT = ("Microsoft YaHei", 14)
FONT_SMALL = ("Microsoft YaHei", 12)
FONT_LABEL = ("Microsoft YaHei", 12, "bold")
FONT_TITLE = ("Microsoft YaHei", 20, "bold")
FONT_SUB = ("Microsoft YaHei", 12)
FIELD_W = 110
COMBO_W = 300

MODE_LABELS = {"鼻高于头顶（生产推荐）": "yrule",
              "双通道智能（对照）": "v3",
              "垂直仰角（对照）": "pitch",
              "自适应抬升（高召回）": "rise",
              "动态俯仰角（对照）": "dynamic",
              "固定俯仰角（对照）": "fixed"}
MODE_DEFAULT_K = {"yrule": "2", "v3": "3.0", "rise": "5.0", "pitch": "30",
                  "dynamic": "1.25", "fixed": "20"}
MODE_K_LABEL = {"yrule": "像素差阈值",
                "v3": "灵敏度 k（×MAD）",
                "rise": "灵敏度 k（×MAD）",
                "pitch": "判定角（度）",
                "dynamic": "动态 k（×σ）",
                "fixed": "固定偏移（度）"}
MODE_HINT = {
    "yrule": "生产推荐：确认抬头=鼻高于头顶≥2px且≥0.2s（计入统计）；疑似抬头（≥0px且≥0.12s）单独列表供人工判断。",
    "v3": "研究对照：滚动局部基线 + 双通道门控。",
    "rise": "高召回模式：适合漏检代价高、可接受多报的场景。",
    "pitch": "研究对照：俯仰角相对中位数的变化。",
    "dynamic": "研究对照：俯仰角中位数 − k·σ 动态阈值。",
    "fixed": "研究对照：俯仰角固定偏移阈值。",
}


def res_path(name):
    """资源文件路径：兼容源码运行与 PyInstaller onefile 运行。"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


# ================================================================ 批量检测核心
def pair_files(folder, allow_video_only=False):
    """自动配对文件夹内的视频与 DLC CSV。

    allow_video_only=True 时，无 CSV 的视频也返回（csv=None），供后续姿态推理。
    返回 [(视频路径或None, csv路径或None, fps)]。
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    videos = sorted(f for f in files
                    if os.path.splitext(f)[1].lower() in hl.VIDEO_EXTS)
    csvs = sorted(f for f in files if f.lower().endswith(".csv"))
    pairs, used = [], set()
    for v in videos:
        stem = os.path.splitext(os.path.basename(v))[0]
        match = next((c for c in csvs
                      if os.path.basename(c).startswith(stem) and c not in used), None)
        if match:
            used.add(match)
            pairs.append((v, match, hl.read_fps(v)))
        elif allow_video_only:
            pairs.append((v, None, hl.read_fps(v)))
    for c in csvs:
        if c not in used:
            pairs.append((None, c, 20.0))
    return pairs


def ensure_keypoints(pairs, folder, log=print):
    """对 csv=None 的视频跑 DLC 推理，补全关键点 CSV 路径。"""
    missing = [(v, fps) for v, c, fps in pairs if c is None and v]
    if not missing:
        return pairs
    videos = [v for v, _ in missing]
    log(f"[姿态] {len(videos)} 个视频缺少关键点，开始 DLC 推理…")
    results = pose_infer.generate_keypoints_for_videos(videos, destfolder=folder, log=log)
    # 统一用规范路径字符串做 key，避免 Path/str 混用
    csv_map = {}
    for vp, cp in results:
        csv_map[os.path.normcase(os.path.abspath(str(vp)))] = cp
    out = []
    for v, c, fps in pairs:
        key = os.path.normcase(os.path.abspath(v)) if v else None
        if c is None and v and key in csv_map and csv_map[key]:
            out.append((v, str(csv_map[key]), fps))
        elif c is None and v:
            raise RuntimeError(
                f"视频「{os.path.basename(v)}」推理后仍无关键点 CSV。\n"
                "请检查 DLC 环境（HEADLIFT_DLC_PYTHON）或将已有 *DLC*.csv 放到视频同目录。")
        else:
            out.append((v, c, fps))
    return out


def run_batch(cfg, log=print):
    """按配置执行批量检测并生成报表。"""
    gen_pose = bool(cfg.get("gen_pose"))
    pairs = pair_files(cfg["folder"], allow_video_only=gen_pose)
    if not pairs:
        raise RuntimeError(
            "文件夹中未找到可分析数据。\n"
            "请放入 DLC 关键点 CSV，或放入视频并勾选「自动生成关键点」。")
    if gen_pose:
        pairs = ensure_keypoints(pairs, cfg["folder"], log=log)
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    figs_dir = os.path.join(out, "统计图")
    os.makedirs(figs_dir, exist_ok=True)

    summaries, event_rows, suspect_rows, figure_paths = [], [], [], []
    for i, (video, csv, fps) in enumerate(pairs, 1):
        name = os.path.splitext(os.path.basename(video or csv))[0]
        group = hl.detect_group(name)
        log(f"[{i}/{len(pairs)}] {name}（{group}，{fps:g}fps）")

        df = hl.load_dlc_csv(csv)
        susp_ev = None
        if cfg["mode"] == "yrule":  # 生产推荐：鼻尖高于头顶 + 疑似档
            ev, susp_ev, flags = hl.detect_lifts_twotier(
                df, fps, pcutoff=cfg["pcutoff"], smooth_win=cfg["smooth"],
                conf_margin=cfg["k"], conf_min_dur=cfg["min_dur"],
                susp_margin=0.0, susp_min_dur=0.12)
            feat = (df["head_y"] - df["nose_y"]).to_numpy(dtype=float)
            base = np.zeros(len(feat), dtype=float)
            thr_arr = np.full(len(feat), float(cfg["k"]))
            thr_susp = np.zeros(len(feat), dtype=float)
            trace_kw = dict(up=True, base_arr=base, thr_arr=thr_arr,
                            thr_lab=f"确认阈值 ≥{cfg['k']:g}px（疑似≥0px 另表）",
                            ylab="鼻尖-头顶高度差（像素）\n抬头时鼻更高")
        elif cfg["mode"] == "v3":  # 双通道智能（金标准校准）
            ev, flags, feat, trace_kw = hl.detect_lifts_v3(
                df, fps, pcutoff=cfg["pcutoff"], smooth_win=cfg["smooth"],
                k=cfg["k"], min_dur_s=cfg["min_dur"])
        elif cfg["mode"] == "rise":  # 自适应抬升法（高召回）
            feat = hl.nose_height(df, pcutoff=cfg["pcutoff"],
                                  smooth_win=cfg["smooth"])
            fa = np.asarray(feat, dtype=float)
            base = float(np.nanmedian(fa))
            mad = float(np.nanmedian(np.abs(fa - base))) * 1.4826
            thr_off = max(cfg["k"] * mad, 10.0)
            ev, flags = hl.detect_lifts_rise(feat, fps, k=cfg["k"],
                                             min_dur_s=cfg["min_dur"],
                                             merge_gap_s=cfg["band"])
            trace_kw = dict(up=True, thr_deg=thr_off, band=thr_off / 3)
        elif cfg["mode"] == "pitch":  # 鼻尖仰角法（旧）
            feat = hl.head_pitch(df, pcutoff=cfg["pcutoff"],
                                 smooth_win=cfg["smooth"])
            ev, flags = hl.detect_lifts_pitch(feat, fps, thr_deg=cfg["k"],
                                              neutral_band=cfg["band"],
                                              min_dur_s=cfg["min_dur"])
            trace_kw = dict(up=True, thr_deg=cfg["k"], band=cfg["band"])
        else:  # 旧角度法：动态阈值 / 固定阈值
            feat = hl.head_angle(df, pcutoff=cfg["pcutoff"],
                                 smooth_win=cfg["smooth"])
            ev, flags = hl.detect_lifts(feat, fps, mode=cfg["mode"], k=cfg["k"],
                                        fixed_offset=cfg["fixed_offset"],
                                        min_dur_s=cfg["min_dur"])
            trace_kw = dict(up=False)
        n_susp = len(susp_ev) if susp_ev is not None else 0
        s = hl.summarize(name, group, ev, fps, len(df), n_suspect=n_susp)
        summaries.append(s)
        log(f"    确认抬头 {s['抬头次数']} 次 | {s['抬头频率(次/分)']} 次/分"
            f" | 总时长 {s['抬头总时长(s)']}s"
            + (f" | 疑似 {n_susp} 条（见Excel「疑似抬头_待人工」）" if n_susp else ""))

        if len(ev):
            ev.insert(0, "视频", name)
            ev.insert(1, "分组", group)
            event_rows.append(ev)
        ev.to_csv(os.path.join(out, f"{name}_抬头事件.csv"),
                  index=False, encoding="utf-8-sig")
        if susp_ev is not None and len(susp_ev):
            susp_ev = susp_ev.copy()
            susp_ev.insert(0, "视频", name)
            susp_ev.insert(1, "分组", group)
            suspect_rows.append(susp_ev)
            susp_ev.to_csv(os.path.join(out, f"{name}_疑似抬头.csv"),
                           index=False, encoding="utf-8-sig")

        if cfg["trace_fig"]:
            fp = os.path.join(figs_dir, f"轨迹图_{name}.png")
            rp.make_trace_figure(feat, fps, ev, f"{name}（{group}）", fp,
                                 **trace_kw)  # v3 模式含滚动基线/阈值曲线
            figure_paths.append(fp)
        if cfg["clips"] and video:
            if len(ev):
                clip_dir = hl.export_clips(video, ev, os.path.join(out, "抬头片段", name))
                log(f"    裁剪确认抬头片段 {len(ev)} 段 → {clip_dir}")
            else:
                log("    未检出确认抬头，跳过片段裁剪")
            if susp_ev is not None and len(susp_ev):
                sdir = hl.export_clips(
                    video, susp_ev, os.path.join(out, "疑似抬头片段", name),
                    prefix="疑似", tag_label="Suspect")
                log(f"    裁剪疑似抬头片段 {len(susp_ev)} 段 → {sdir}")
        if cfg["counter_video"] and video:
            def _pv(done, total, _name=name):
                log(f"    标注视频 {_name}：{done / total * 100:.0f}%")
            mp4 = hl.render_counter_video(
                video, df, flags, os.path.join(out, "计数视频", name + "_标注.mp4"),
                pcutoff=cfg["pcutoff"], progress=_pv)
            log(f"    生成计数标注视频 → {mp4}")

    summary_df = pd.DataFrame(summaries)
    events_df = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    suspect_df = pd.concat(suspect_rows, ignore_index=True) if suspect_rows else pd.DataFrame()
    gfp = os.path.join(figs_dir, "组间对比图.png")
    rp.make_group_figure(summary_df, gfp)
    figure_paths.insert(0, gfp)
    xlsx = os.path.join(out, f"抬头行为分析报表_{cfg['stamp']}.xlsx")
    rp.write_excel(xlsx, summary_df, events_df, cfg["params_text"], figure_paths,
                   suspect_df=suspect_df)
    n_sus_total = int(summary_df["疑似抬头次数(待人工)"].sum()) if len(summary_df) else 0
    log(f"✅ 全部完成：{len(pairs)} 段视频，确认抬头 {int(summary_df['抬头次数'].sum()) if len(summary_df) else 0} 次"
        f"，疑似 {n_sus_total} 条 → {os.path.basename(xlsx)}")
    return xlsx


# ================================================================ 图形界面
class App(ctk.CTk):
    def __init__(self, folder=None, out=None):
        try:  # 独立任务栏图标（避免显示 python.exe 默认图标），须在创建窗口前设置
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "MouseHeadLiftAnalyzer.V1.0")
        except Exception:
            pass
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.title(f"{APP_NAME} {VERSION}")
        self.geometry("1080x780")
        self.minsize(980, 720)
        self.configure(fg_color=BG)
        try:  # 窗口/任务栏图标
            self.iconbitmap(res_path("HeadLiftAnalyzer.ico"))
        except Exception:
            pass
        self.queue = queue.Queue()
        self.result_dir = None
        self._build()
        if folder:  # 命令行预填数据文件夹（记住上次目录）
            self.var_folder.set(folder)
            self._refresh_pairs()
        if out:
            self.var_out.set(out)
        self.after(100, self._poll)
        if os.environ.get("HEADLIFT_AUTOSTART"):  # 演示/录屏模式：启动后自动开始分析
            self.after(800, self.start)

    # ---------- 界面搭建 ----------
    def _card(self, parent, title):
        card = ctk.CTkFrame(parent, corner_radius=12, fg_color=CARD,
                            border_width=1, border_color=CARD_EDGE)
        card.pack(fill="x", padx=14, pady=(8, 0))
        ctk.CTkLabel(card, text=title, font=("Microsoft YaHei", 13, "bold"),
                     text_color=TEXT, anchor="w").pack(fill="x", padx=14, pady=(10, 0))
        return card

    def _entry(self, parent, var, width=FIELD_W):
        return ctk.CTkEntry(
            parent, textvariable=var, width=width, height=FIELD_H,
            font=FONT, justify="center", corner_radius=8,
            fg_color=FIELD_BG, border_color=CARD_EDGE, border_width=1,
            text_color=TEXT)

    def _btn(self, parent, text, cmd, width=96):
        return ctk.CTkButton(
            parent, text=text, width=width, height=FIELD_H, font=FONT_SMALL,
            fg_color=("#E2E8F0", "#334155"), hover_color=("#CBD5E1", "#475569"),
            text_color=("#0F172A", "#E2E8F0"), corner_radius=8, command=cmd)

    def _field(self, parent, label, var, width=FIELD_W):
        ctk.CTkLabel(parent, text=label, font=FONT_LABEL, text_color=TEXT,
                     anchor="e").pack(side="left", padx=(0, 4))
        self._entry(parent, var, width).pack(side="left", padx=(0, 14))

    def _build(self):
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(12, 0))
        left = ctk.CTkFrame(head, fg_color="transparent")
        left.pack(side="left")
        ctk.CTkLabel(left, text=APP_NAME, font=FONT_TITLE,
                     text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(
            left, text="原视频 / DLC 关键点 → 抬头检测 → Excel 报表 · 一键批量",
            font=FONT_SUB, text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))
        ctk.CTkLabel(head, text=VERSION, font=("Microsoft YaHei", 11, "bold"),
                     text_color="white", corner_radius=8, fg_color=ACCENT,
                     width=52, height=26).pack(side="right", pady=(6, 0))

        # ① 数据
        c1 = self._card(self, "① 数据文件夹")
        row = ctk.CTkFrame(c1, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(6, 2))
        self.var_folder = tk.StringVar()
        ctk.CTkEntry(row, textvariable=self.var_folder, height=FIELD_H,
                     font=FONT, corner_radius=8, fg_color=FIELD_BG,
                     border_color=CARD_EDGE, border_width=1, text_color=TEXT,
                     placeholder_text="选择视频或含 DLC CSV 的文件夹…"
                     ).pack(side="left", fill="x", expand=True)
        self._btn(row, "浏览…", self.pick_folder).pack(side="left", padx=(8, 0))
        self.var_gen_pose = tk.StringVar(value="on")
        ctk.CTkSwitch(
            c1, text="自动生成关键点（仅有视频时用 DLC 姿态估计，需本机已装 DeepLabCut）",
            font=FONT_SMALL, variable=self.var_gen_pose, onvalue="on", offvalue="off",
            progress_color=ACCENT, command=self._refresh_pairs
        ).pack(anchor="w", padx=16, pady=(0, 2))
        self.lbl_pairs = ctk.CTkLabel(
            c1, text="尚未选择文件夹", font=FONT_SMALL,
            text_color=TEXT_MUTED, anchor="w")
        self.lbl_pairs.pack(fill="x", padx=16, pady=(0, 8))

        # ② 参数（两行紧凑）
        c2 = self._card(self, "② 检测参数（确认计入统计；疑似单独输出供人工判断）")
        body = ctk.CTkFrame(c2, fg_color="transparent")
        body.pack(fill="x", padx=14, pady=(4, 6))
        r1 = ctk.CTkFrame(body, fg_color="transparent")
        r1.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(r1, text="判定模式", font=FONT_LABEL, text_color=TEXT,
                     anchor="w").pack(side="left", padx=(0, 4))
        self.cmb_mode = ctk.CTkComboBox(
            r1, values=list(MODE_LABELS), width=COMBO_W, height=FIELD_H,
            font=FONT_SMALL, dropdown_font=FONT_SMALL, state="readonly",
            fg_color=FIELD_BG, border_color=CARD_EDGE, border_width=1,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=TEXT, command=self._on_mode_change)
        self.cmb_mode.set("鼻高于头顶（生产推荐）")
        self.cmb_mode.pack(side="left", padx=(0, 14))
        self.var_k = tk.StringVar(value=MODE_DEFAULT_K["yrule"])
        self._field(r1, "像素差", self.var_k, 80)
        self.var_smooth = tk.StringVar(value="5")
        self._field(r1, "中值滤波", self.var_smooth, 80)
        self.var_mindur = tk.StringVar(value="0.2")
        self._field(r1, "最短持续s", self.var_mindur, 80)
        self.var_pcutoff = tk.StringVar(value="0.5")
        self._field(r1, "置信度", self.var_pcutoff, 80)
        self.lbl_hint = ctk.CTkLabel(
            body, text=MODE_HINT["yrule"], font=FONT_SMALL,
            text_color=TEXT_MUTED, anchor="w", justify="left", wraplength=1000)
        self.lbl_hint.pack(fill="x", pady=(4, 2))

        # ③ 输出
        c3 = self._card(self, "③ 输出")
        opts = ctk.CTkFrame(c3, fg_color="transparent")
        opts.pack(fill="x", padx=14, pady=(6, 2))
        self.var_trace = tk.StringVar(value="on")
        self.var_clips = tk.StringVar(value="on")
        self.var_counter = tk.StringVar(value="on")
        ctk.CTkLabel(opts, text="固定：Excel+统计图", font=FONT_SMALL,
                     text_color=TEXT_MUTED).pack(side="left", padx=(0, 16))
        for text, var in [("轨迹图", self.var_trace),
                          ("片段(确认+疑似)", self.var_clips),
                          ("标注视频", self.var_counter)]:
            ctk.CTkSwitch(opts, text=text, font=FONT_SMALL, variable=var,
                          onvalue="on", offvalue="off",
                          progress_color=ACCENT).pack(side="left", padx=(0, 14))
        row3 = ctk.CTkFrame(c3, fg_color="transparent")
        row3.pack(fill="x", padx=14, pady=(4, 10))
        ctk.CTkLabel(row3, text="输出文件夹", font=FONT_LABEL, text_color=TEXT,
                     anchor="w").pack(side="left", padx=(0, 4))
        self.var_out = tk.StringVar()
        ctk.CTkEntry(row3, textvariable=self.var_out, height=FIELD_H, font=FONT_SMALL,
                     corner_radius=8, fg_color=FIELD_BG,
                     border_color=CARD_EDGE, border_width=1, text_color=TEXT
                     ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn(row3, "浏览…", self.pick_out).pack(side="left")

        # 运行
        run = ctk.CTkFrame(self, fg_color="transparent")
        run.pack(fill="x", padx=14, pady=(10, 4))
        self.btn_run = ctk.CTkButton(
            run, text="开始批量分析", height=42, width=200,
            font=("Microsoft YaHei", 15, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOVER, corner_radius=10,
            command=self.start)
        self.btn_run.pack(side="left")
        self.btn_open = ctk.CTkButton(
            run, text="打开结果", height=42, width=120,
            font=FONT_SMALL, corner_radius=10,
            fg_color=("#E2E8F0", "#334155"), hover_color=("#CBD5E1", "#475569"),
            text_color=("#0F172A", "#E2E8F0"),
            state="disabled", command=self.open_out)
        self.btn_open.pack(side="left", padx=10)
        self.progress = ctk.CTkProgressBar(run, height=10, corner_radius=4,
                                           progress_color=ACCENT)
        self.progress.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.progress.set(0)

        # 日志（固定高度，不滚动整页）
        logcard = ctk.CTkFrame(self, corner_radius=12, fg_color=CARD,
                               border_width=1, border_color=CARD_EDGE, height=170)
        logcard.pack(fill="x", padx=14, pady=(2, 4))
        logcard.pack_propagate(False)
        ctk.CTkLabel(logcard, text="运行日志", font=("Microsoft YaHei", 12, "bold"),
                     text_color=TEXT, anchor="w").pack(fill="x", padx=14, pady=(8, 0))
        self.txt = ctk.CTkTextbox(
            logcard, font=("Consolas", 12),
            fg_color=FIELD_BG, text_color=TEXT, wrap="none", corner_radius=8)
        self.txt.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(
            foot, text="操作定义：鼻尖高于头顶即为抬头",
            font=("Microsoft YaHei", 11), text_color=TEXT_MUTED,
        ).pack(side="left")
        ctk.CTkLabel(foot, text=f"{APP_NAME} {VERSION}",
                     font=("Microsoft YaHei", 11), text_color=TEXT_MUTED
                     ).pack(side="right")

    # ---------- 事件 ----------
    def _on_mode_change(self, mode):
        """切换判定模式时联动：参数标签与默认值跟随模式。"""
        m = MODE_LABELS.get(mode, "yrule")
        self.var_k.set(MODE_DEFAULT_K[m])
        self.lbl_hint.configure(text=MODE_HINT.get(m, ""))
        if m == "yrule":
            self.var_smooth.set("5")
            self.var_mindur.set("0.2")
            self.var_pcutoff.set("0.5")
        elif m == "v3":
            self.var_smooth.set("5")
            self.var_mindur.set("0.2")
            self.var_pcutoff.set("0.6")
        elif m == "pitch":
            self.var_smooth.set("5")
            self.var_mindur.set("0.5")
            self.var_pcutoff.set("0.6")
        elif m == "rise":
            self.var_smooth.set("5")
            self.var_mindur.set("0.3")
            self.var_pcutoff.set("0.6")

    def pick_folder(self):
        d = filedialog.askdirectory(title="选择存放视频与 CSV 的数据文件夹")
        if not d:
            return
        self.var_folder.set(d)
        self._refresh_pairs()
        if not self.var_out.get():
            self.var_out.set(os.path.join(d, "抬头分析结果"))

    def _refresh_pairs(self):
        d = self.var_folder.get().strip()
        if not d or not os.path.isdir(d):
            self.lbl_pairs.configure(text="文件夹不存在", text_color="#EF4444")
            return
        gen = self.var_gen_pose.get() == "on"
        pairs = pair_files(d, allow_video_only=gen)
        n_csv = sum(1 for _, c, _ in pairs if c)
        n_vid = sum(1 for v, c, _ in pairs if v and not c)
        if n_csv and not n_vid:
            msg = f"✔  已配对 {n_csv} 组（视频+关键点），可以开始分析"
            color = ACCENT
        elif n_vid:
            ok, tip = pose_infer.dlc_available()
            if ok:
                msg = f"✔  {n_csv} 组已有关键点；{n_vid} 个视频将自动生成关键点后分析"
                color = ACCENT
            else:
                msg = f"⚠  {n_vid} 个视频缺关键点；未检测到可用 DLC：{tip[:80]}"
                color = "#F59E0B"
        elif n_csv:
            msg = f"✔  已找到 {n_csv} 个关键点 CSV"
            color = ACCENT
        else:
            msg = "✖  未找到视频或 DLC 关键点 CSV"
            color = "#EF4444"
        self.lbl_pairs.configure(text=msg, text_color=color)

    def pick_out(self):
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.var_out.set(d)

    def collect_cfg(self):
        import datetime
        mode_label = self.cmb_mode.get()
        mode = MODE_LABELS.get(mode_label, "pitch")
        return {
            "folder": self.var_folder.get().strip(),
            "out": self.var_out.get().strip() or os.path.join(
                self.var_folder.get(), "抬头分析结果"),
            "mode": mode,
            "k": float(self.var_k.get()),
            "band": 0.8 if mode == "rise" else 12.0,
            "fixed_offset": (float(self.var_k.get()) if mode == "fixed"
                             else 20.0),
            "smooth": int(self.var_smooth.get()),
            "min_dur": float(self.var_mindur.get()),
            "pcutoff": float(self.var_pcutoff.get()),
            "trace_fig": self.var_trace.get() == "on",
            "clips": self.var_clips.get() == "on",
            "counter_video": self.var_counter.get() == "on",
            "gen_pose": self.var_gen_pose.get() == "on",
            "stamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "params_text": {
                "判定模式": mode_label,
                "抬头判定角/系数": self.var_k.get(),
                "中值滤波窗口": self.var_smooth.get(),
                "最短持续(s)": self.var_mindur.get(),
                "置信度阈值": self.var_pcutoff.get(),
                "自动生成关键点": "是" if self.var_gen_pose.get() == "on" else "否",
                "系统版本": VERSION},
        }

    def start(self):
        try:
            cfg = self.collect_cfg()
        except ValueError:
            messagebox.showerror(APP_NAME, "参数格式有误，请检查数值输入")
            return
        if not cfg["folder"]:
            messagebox.showerror(APP_NAME, "请先选择数据文件夹")
            return
        self.btn_run.configure(state="disabled", text="分析中…")
        self.btn_open.configure(state="disabled")
        self.progress.set(0)
        self._n_total = max(1, len(pair_files(
            cfg["folder"], allow_video_only=cfg.get("gen_pose", False))))
        self._log(f"—— {APP_NAME} {VERSION} 开始分析 ——")
        threading.Thread(target=self._worker, args=(cfg,), daemon=True).start()

    def _worker(self, cfg):
        try:
            xlsx = run_batch(cfg, log=lambda m: self.queue.put(("log", m)))
            self.queue.put(("done", xlsx))
        except Exception as exc:  # 界面线程隔离：异常回传日志
            self.queue.put(("error", str(exc)))

    def _poll(self):
        try:
            while True:
                kind, msg = self.queue.get_nowait()
                if kind == "log":
                    self._log(msg)
                    if msg.startswith("[") and "/" in msg.split("]")[0]:
                        done = int(msg[1:msg.index("/")])
                        self.progress.set(done / self._n_total)
                elif kind == "done":
                    self._log("全部任务完成。")
                    self.progress.set(1.0)
                    self.btn_run.configure(state="normal", text="🚀  开始批量分析")
                    self.btn_open.configure(state="normal")
                    self.result_dir = os.path.dirname(msg)
                elif kind == "error":
                    self._log(f"❌ 出错：{msg}")
                    self.btn_run.configure(state="normal", text="🚀  开始批量分析")
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _log(self, msg):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def open_out(self):
        if self.result_dir and os.path.isdir(self.result_dir):
            os.startfile(self.result_dir)


def main():
    folder = sys.argv[sys.argv.index("--folder") + 1] if "--folder" in sys.argv else None
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if "--batch" in sys.argv:  # 命令行批处理模式
        i = sys.argv.index("--batch")
        folder = sys.argv[i + 1] if len(sys.argv) > i + 1 else os.getcwd()
        out = (sys.argv[sys.argv.index("--out") + 1]
               if "--out" in sys.argv else os.path.join(folder, "抬头分析结果"))
        import datetime
        run_batch({"folder": folder, "out": out, "mode": "v3", "k": 3.0,
                   "band": 12.0, "fixed_offset": 20.0, "smooth": 5,
                   "min_dur": 0.2,
                   "pcutoff": 0.6, "trace_fig": True, "clips": True,
                   "counter_video": True,
                   "stamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                   "params_text": {"判定模式": "双通道智能(k=3.0×MAD，金标准校准)",
                                   "灵敏度系数": 3.0,
                                   "中值滤波窗口": 5, "最短持续(s)": 0.25,
                                   "置信度阈值": 0.6, "系统版本": VERSION}})
        return
    App(folder=folder, out=out).mainloop()


if __name__ == "__main__":
    main()
