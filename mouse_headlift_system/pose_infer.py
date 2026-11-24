# -*- coding: utf-8 -*-
"""
从原始视频生成 DeepLabCut 关键点 CSV（姿态估计层）。

依赖：
  1. 已训练好的 DLC 工程（默认：Mouse Main/Mouse_Head_Lift_Detection-User-2026-04-13）
  2. 安装了 deeplabcut 的 Python 环境（可用 HEADLIFT_DLC_PYTHON 指定解释器）

本模块不绑定 GUI；GUI / CLI 都可调用 generate_keypoints_for_videos()。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# 相对本文件（mouse_headlift_system/）定位 DLC 工程
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # Mouse Main/
# PyInstaller onefile：资源在临时目录；模型应在 exe 旁的 dlc_model/
_FROZEN_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None


def _candidate_project_roots() -> List[Path]:
    roots = []
    env = os.environ.get("HEADLIFT_DLC_PROJECT")
    if env:
        roots.append(Path(env))
    if _FROZEN_DIR is not None:
        roots.append(_FROZEN_DIR / "dlc_model")
        roots.append(_FROZEN_DIR)
    roots.append(_ROOT / "Mouse_Head_Lift_Detection-User-2026-04-13")
    roots.append(_ROOT / "dlc_infer_2166_2171" / "project")
    return roots


def resolve_dlc_project() -> Optional[Path]:
    """找到含 dlc-models 与可用 config 的工程根目录。"""
    for root in _candidate_project_roots():
        if not root.exists():
            continue
        # 工程根：有 dlc-models 子目录
        if (root / "dlc-models").is_dir():
            return root
        # 或 root 本身就是 config.yaml + 旁有 dlc-models
        if (root / "config.yaml").exists() and (root / "dlc-models").is_dir():
            return root
    return None


DEFAULT_DLC_PROJECT = (_ROOT / "Mouse_Head_Lift_Detection-User-2026-04-13")
DEFAULT_DLC_CONFIG = DEFAULT_DLC_PROJECT / "config.yaml"

_MINIMAL_CONFIG = """# Project definitions (do not edit)
Task: Mouse_Head_Lift_Detection
scorer: User
date: Apr13
multianimalproject: false
identity:

    # Project path (change when moving around)
project_path: {project_path}

    # Annotation data set configuration (and individual video cropping parameters)
video_sets: {{}}

bodyparts:
- nose
- head_top
- left_ear
- right_ear

start: 0
stop: 1
numframes2pick: 20

    # Plotting configuration
skeleton:
- - nose
  - head_top
- - nose
  - left_ear
- - nose
  - right_ear
skeleton_color: black
pcutoff: 0.6
dotsize: 10
alphavalue: 0.7
colormap: rainbow

    # Training,Evaluation and Analysis configuration
TrainingFraction:
- 0.95
iteration: 0
default_net_type: resnet_50
default_augmenter: default
snapshotindex: -1
batch_size: 8

    # Cropping Parameters (for analysis and outlier frame detection)
cropping: false
x1: 0
x2: 640
y1: 277
y2: 624

    # Refinement configuration
corner2move2:
- 50
- 50
move2corner: true
"""


def _write_yaml_text(path: Path, text: str) -> Path:
    """写 YAML 用 GBK（中文 Windows 系统默认编码）。

    DeepLabCut 读取 config 时会先用系统默认编码打开文本流再交给 ruamel，
    UTF-8 BOM 也救不了（BOM 字节会先按 GBK 解码失败）。因此必须写 GBK。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("gbk", errors="replace"))
    return path


def _read_yaml_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace")


def ensure_config(project: Path) -> Path:
    """确保工程目录有 config.yaml；没有则写最小可用配置（GBK）。"""
    cfg = project / "config.yaml"
    if cfg.exists():
        # 统一转成 GBK，避免 DLC 按系统编码读 UTF-8 失败
        try:
            text = _read_yaml_text(cfg)
            _write_yaml_text(cfg, text)
        except Exception:
            pass
        return cfg
    _write_yaml_text(
        cfg,
        _MINIMAL_CONFIG.format(project_path=str(project).replace("\\", "/")))
    return cfg


def _candidate_python_exes(python_exe: Optional[str] = None) -> List[str]:
    """按优先级列出可能装了 DeepLabCut 的 Python 解释器。

    注意：打包后的 exe 不是 Python，绝不能用 sys.executable 去 import。
    """
    cands: List[str] = []
    if python_exe:
        cands.append(python_exe)
    env = os.environ.get("HEADLIFT_DLC_PYTHON")
    if env:
        cands.append(env)

    home = Path.home()
    user = os.environ.get("USERNAME", "")
    prefixes = [
        home / "miniconda3",
        home / "anaconda3",
        home / "miniforge3",
        Path(os.environ.get("LOCALAPPDATA", "")) / "miniconda3",
        Path(os.environ.get("LOCALAPPDATA", "")) / "anaconda3",
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/anaconda3"),
        Path(os.environ.get("USERPROFILE", str(home))) / "scoop" / "apps" / "miniconda3" / "current",
    ]
    env_names = ["headlift_dlc", "dlc", "DLC", "deeplabcut", "openai", "base"]
    for prefix in prefixes:
        if not prefix or not Path(prefix).exists():
            continue
        # envs
        envs_dir = Path(prefix) / "envs"
        for name in env_names:
            py = envs_dir / name / "python.exe"
            if py.exists():
                cands.append(str(py))
        # base
        base_py = Path(prefix) / "python.exe"
        if base_py.exists():
            cands.append(str(base_py))

    # 源码运行时才可用当前解释器
    if not getattr(sys, "frozen", False):
        cands.append(sys.executable)

    # 去重保序
    seen, out = set(), []
    for p in cands:
        if p and p not in seen and Path(p).exists():
            seen.add(p)
            out.append(p)
    return out


def find_dlc_python(timeout: int = 20) -> Tuple[Optional[str], str]:
    """自动搜索可用的 DeepLabCut Python。返回 (path or None, message)。"""
    cands = _candidate_python_exes()
    if not cands:
        return None, "未找到候选 Python（未设置 HEADLIFT_DLC_PYTHON，也未发现 conda 环境）"
    tried = []
    for py in cands:
        probe = "import deeplabcut;print('OK', deeplabcut.__version__)"
        try:
            r = subprocess.run(
                [py, "-c", probe],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            tried.append(f"  {py}: {exc}")
            continue
        if r.returncode == 0 and "OK" in (r.stdout or ""):
            ver = (r.stdout or "").strip().splitlines()[-1]
            return py, f"已找到 DeepLabCut：{py}（{ver}）"
        tail = ((r.stderr or r.stdout or "").strip().splitlines() or ["失败"])[-1]
        tried.append(f"  {py}: {tail[:80]}")
    return None, "以下 Python 均无法导入 DeepLabCut：\n" + "\n".join(tried[:8])


def dlc_available(python_exe: Optional[str] = None) -> Tuple[bool, str]:
    """检查能否跑 DLC 推理。返回 (ok, message)。"""
    project = resolve_dlc_project()
    if project is None:
        return False, (
            "未找到 DLC 模型目录（需含 dlc-models/snapshot）。"
            "可将模型放在 exe 同级 dlc_model\\，或设置 HEADLIFT_DLC_PROJECT")
    if not list(project.glob("dlc-models/**/snapshot-*.index")):
        return False, f"模型目录无 snapshot：{project / 'dlc-models'}"
    env_cfg = os.environ.get("HEADLIFT_DLC_CONFIG")
    if env_cfg and Path(env_cfg).exists():
        cfg_msg = env_cfg
    else:
        try:
            ensure_config(project)
            cfg_msg = str(project / "config.yaml")
        except Exception as exc:
            return False, f"无法写入 DLC config：{exc}"

    py, py_msg = find_dlc_python()
    if not py:
        return False, py_msg
    return True, f"{py_msg}\n模型 {project.name} | 配置 {cfg_msg}"


def _write_runtime_config(dest: Path) -> Path:
    """写一份 project_path 指向本机工程的临时 config，避免原 yaml 路径失效。"""
    project = resolve_dlc_project()
    if project is None:
        raise RuntimeError("未找到 DLC 模型目录")
    src = project / "config.yaml"
    if not src.exists():
        src = ensure_config(project)
    text = _read_yaml_text(src)
    lines = []
    proj = str(project).replace("\\", "/")
    for line in text.splitlines():
        if line.strip().startswith("project_path:"):
            lines.append(f"project_path: {proj}")
        elif line.strip().startswith("engine:"):
            lines.append("engine: tensorflow")
        else:
            lines.append(line)
    if not any(l.strip().startswith("engine:") for l in lines):
        # DLC 3 默认可能写成 pytorch；本模型是 TF ResNet50
        lines.insert(0, "engine: tensorflow")
    _write_yaml_text(dest, "\n".join(lines) + "\n")
    return dest


def _find_generated_csv(video: Path, destfolder: Path) -> Optional[Path]:
    stem = video.stem
    # DLC 命名：<video_stem>DLC_..._<snapshot>.csv
    cands = sorted(destfolder.glob(f"{stem}DLC*.csv"))
    if not cands:
        cands = sorted(video.parent.glob(f"{stem}DLC*.csv"))
    return cands[-1] if cands else None


def generate_keypoints_for_videos(
    videos: List[Path],
    destfolder: Optional[Path] = None,
    python_exe: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> List[Tuple[Path, Optional[Path]]]:
    """
    对缺少 CSV 的视频跑 DLC analyze_videos，生成关键点 CSV。

    返回 [(video, csv_path_or_None), ...]
    """
    ok, msg = dlc_available(python_exe)
    if not ok:
        raise RuntimeError(
            "无法自动生成关键点坐标：\n"
            f"{msg}\n\n"
            "处理方式：\n"
            "1. 双击交付包中的「一键修复DLC环境.bat」（会自动搜索本机已有 DLC）\n"
            "2. 或设置环境变量 HEADLIFT_DLC_PYTHON 指向已装 DLC 的 python.exe\n"
            "3. 或把 *DLC*.csv 放到视频同目录后再分析\n"
            "4. 模型路径可用 HEADLIFT_DLC_PROJECT / HEADLIFT_DLC_CONFIG 覆盖"
        )
    log(f"[姿态] {msg}")

    dest = Path(destfolder) if destfolder else None
    if dest is None:
        dest = Path(tempfile.mkdtemp(prefix="headlift_dlc_"))
    dest.mkdir(parents=True, exist_ok=True)

    py, _ = find_dlc_python()
    if not py:
        raise RuntimeError("未找到可用的 DeepLabCut Python，请运行交付包中的「一键修复DLC环境.bat」")

    # 配置必须放在「模型工程目录」里，且 project_path 指向该目录。
    # 若放在输出目录，DLC 会把 project_path 改成输出目录，从而找不到模型。
    project = resolve_dlc_project()
    if project is None:
        raise RuntimeError("未找到 DLC 模型目录")
    ensure_config(project)
    cfg_file = project / "config.yaml"
    # 再次写入正确 project_path（GBK）
    _write_runtime_config(cfg_file)

    results: List[Tuple[Path, Optional[Path]]] = []
    todo: List[Path] = []
    for raw_v in videos:
        v = Path(raw_v)  # ensure Path (GUI 传入的是 str)
        csv = _find_generated_csv(v, v.parent)
        if csv and csv.exists():
            log(f"[姿态] 已有关键点，跳过推理：{csv.name}")
            results.append((v, csv))
        else:
            todo.append(v)

    if not todo:
        return results

    # 分批调用，避免单次命令过长；每批最多 4 个视频
    batch_size = 4
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        paths = [str(p) for p in batch]
        script = (
            "import deeplabcut as dlc\n"
            f"cfg = r'''{cfg_file}'''\n"
            f"videos = {paths!r}\n"
            f"dest = r'''{dest}'''\n"
            "print('[DLC] analyze_videos start', len(videos))\n"
            "dlc.analyze_videos(cfg, videos, save_as_csv=True, destfolder=dest)\n"
            "print('[DLC] analyze_videos done')\n"
        )
        log(f"[姿态] 开始推理 {len(batch)} 个视频 → {dest}")
        # 配置文件为 GBK，与中文 Windows 系统默认编码一致。
        # 不要设 PYTHONUTF8=1，否则 Python 会用 UTF-8 打开 GBK 的 yaml 再炸。
        env = os.environ.copy()
        env.pop("PYTHONUTF8", None)
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if r.stdout:
            for line in r.stdout.splitlines():
                if line.strip():
                    log("    " + line.strip())
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()[-8:]
            raise RuntimeError("DLC 推理失败：\n" + "\n".join(tail))

        for v in batch:
            # 优先在 dest 找，再回落视频目录
            csv = _find_generated_csv(v, dest)
            if csv is None:
                csv = _find_generated_csv(v, v.parent)
            if csv is None:
                log(f"[姿态] ⚠ 未找到生成的 CSV：{v.name}")
                results.append((v, None))
            else:
                # 若生成在 dest，拷贝/链接到视频旁便于配对
                target = v.parent / csv.name
                if csv.resolve() != target.resolve():
                    try:
                        if not target.exists():
                            target.write_bytes(csv.read_bytes())
                        csv = target
                    except Exception:
                        pass  # 仍用 dest 内路径
                log(f"[姿态] ✔ {v.name} → {csv.name}")
                results.append((v, csv))

    return results


if __name__ == "__main__":
    # 简单 CLI：python pose_infer.py 视频1 [视频2 ...]
    vids = [Path(a) for a in sys.argv[1:] if Path(a).suffix.lower()
            in {".avi", ".mp4", ".wmv", ".mov", ".mkv"}]
    if not vids:
        print("用法: python pose_infer.py <video1> [video2 ...]")
        sys.exit(1)
    for v, c in generate_keypoints_for_videos(vids):
        print(v.name, "->", c)
