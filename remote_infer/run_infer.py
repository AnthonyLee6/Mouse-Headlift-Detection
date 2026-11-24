#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run DLC analyze_videos for 2166/2171 using the clean server config."""
from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "project")
CONFIG = os.path.join(PROJECT, "config.yaml")
CLEAN = os.path.join(HERE, "config_server.yaml")
VIDEOS = [
    os.path.join(HERE, "videos", "ct 2166 PBS.WMV"),
    os.path.join(HERE, "videos", "ct 2171 CNO.WMV"),
]


def main():
    for v in VIDEOS:
        if not os.path.exists(v):
            print("[ERROR] missing video:", v)
            sys.exit(1)
    shutil.copy(CLEAN, CONFIG)
    print("[config] using clean", CLEAN)
    import deeplabcut as dlc
    print("[dlc]", dlc.__version__)
    print("[analyze] start", len(VIDEOS), "videos")
    dlc.analyze_videos(
        CONFIG,
        VIDEOS,
        save_as_csv=True,
        destfolder=os.path.join(HERE, "videos"),
    )
    print("[analyze] done")
    for name in sorted(os.listdir(os.path.join(HERE, "videos"))):
        if name.endswith(".csv"):
            print(" CSV", name)


if __name__ == "__main__":
    main()
