# -*- coding: utf-8 -*-
"""School server helper: probe, upload, run DLC infer, download CSVs."""
from __future__ import annotations

import os
import sys
import time

import paramiko

HOST = os.environ.get("DLC_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("DLC_SERVER_PORT", "22"))
USER = os.environ.get("DLC_SERVER_USER", "user")
PASS = os.environ.get("DLC_SERVER_PASS", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_PACK = os.environ.get("DLC_LOCAL_PACK", _HERE)
LOCAL_CSV_DIR = os.environ.get("DLC_LOCAL_CSV_DIR", os.path.join(_HERE, "..", "sample_data"))
REMOTE_DIR = os.environ.get("DLC_REMOTE_DIR", "~/dlc_infer")


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return c


def run(c, cmd, timeout=600):
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out[-4000:])
    if err.strip():
        print("[stderr]", err[-2000:])
    print(f"[exit {code}]")
    return code, out, err


def sftp_put_dir(sftp, local, remote):
    """Recursive upload."""
    os.makedirs  # noqa
    if not sftp.exists(remote):
        sftp.mkdir(remote)
    for name in os.listdir(local):
        lp = os.path.join(local, name)
        rp = f"{remote}/{name}"
        if os.path.isdir(lp):
            sftp_put_dir(sftp, lp, rp)
        else:
            size = os.path.getsize(lp)
            t0 = time.time()
            print(f"  up {lp} ({size/1e6:.1f}MB) -> {rp}")
            sftp.put(lp, rp)
            dt = time.time() - t0
            print(f"     done {dt:.1f}s ({size/1e6/max(dt,0.1):.1f} MB/s)")


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "probe"
    c = connect()
    try:
        if action == "probe":
            run(c, "uname -a; which conda python3 nvidia-smi; nvidia-smi | head -20; df -h ~ | tail -1; free -h | head -2")
        elif action == "upload":
            sftp = c.open_sftp()
            print("uploading pack...")
            sftp_put_dir(sftp, LOCAL_PACK, REMOTE_DIR.replace("~", "/root") if False else REMOTE_DIR)
            # paramiko sftp may not expand ~; use absolute
            sftp.close()
        elif action == "upload_abs":
            sftp = c.open_sftp()
            remote = "/root/dlc_infer_2166_2171"
            print("uploading to", remote)
            sftp_put_dir(sftp, LOCAL_PACK, remote)
            sftp.close()
        elif action == "setup":
            cmds = [
                "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true",
                "which conda || ls ~/miniconda3/bin/conda /opt/conda/bin/conda 2>/dev/null",
                "conda --version 2>/dev/null || true",
            ]
            run(c, "bash -lc 'which conda; conda --version; python3 --version; ls /root | head'")
        elif action == "install":
            script = r'''
set -e
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then . "/opt/conda/etc/profile.d/conda.sh"
fi
conda create -n dlc_tmp python=3.9 -y
conda activate dlc_tmp
pip install -U pip
# TF backbone to match ResNet50 .meta checkpoints
pip install "deeplabcut[tf]" -q
python -c "import deeplabcut as d; print('DLC', d.__version__)"
'''
            run(c, f"bash -lc {script!r}", timeout=1800)
        elif action == "infer":
            script = r'''
set -e
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then . "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate dlc_tmp
cd /root/dlc_infer_2166_2171
python run_infer.py
ls -lah videos/
'''
            run(c, f"bash -lc {script!r}", timeout=3600)
        elif action == "download":
            sftp = c.open_sftp()
            remote_videos = "/root/dlc_infer_2166_2171/videos"
            for name in sftp.listdir(remote_videos):
                if name.endswith(".csv"):
                    rp = f"{remote_videos}/{name}"
                    lp = os.path.join(LOCAL_CSV_DIR, name)
                    print("downloading", name)
                    sftp.get(rp, lp)
                    print("  ->", lp, os.path.getsize(lp))
            sftp.close()
        elif action == "cleanup":
            script = r'''
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then . "/opt/conda/etc/profile.d/conda.sh"
fi
conda deactivate || true
conda env remove -n dlc_tmp -y || true
echo CLEANUP_OK
'''
            run(c, f"bash -lc {script!r}", timeout=300)
        else:
            print("unknown action", action)
    finally:
        c.close()


if __name__ == "__main__":
    main()
