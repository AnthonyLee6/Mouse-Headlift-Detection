# -*- coding: utf-8 -*-
"""按窗口标题精确截图（PrintWindow 直读窗口内容，被遮挡/失焦也可用）。
用法: python capture_window.py <窗口标题> <输出.png>"""
import ctypes
import sys

import win32gui
import win32ui
from PIL import Image

try:  # 高 DPI 下 GetWindowRect 返回物理像素，避免截图比窗口矮
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass


def capture(title, out_path):
    hwnd = win32gui.FindWindow(None, title)
    assert hwnd, f"未找到窗口: {title}"
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    w, h = r - l, b - t
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bmp)
    ok = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
    assert ok, "PrintWindow 失败"
    info = bmp.GetInfo()
    data = bmp.GetBitmapBits(True)
    Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]), data, "raw",
                     "BGRX", 0, 1).save(out_path)
    win32gui.DeleteObject(bmp.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    print(f"已保存 {out_path} ({w}x{h})")


if __name__ == "__main__":
    capture(sys.argv[1], sys.argv[2])
