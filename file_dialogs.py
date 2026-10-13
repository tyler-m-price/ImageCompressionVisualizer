"""Native OS open/save dialogs without extra dependencies.

macOS: AppleScript via osascript (Tk cannot share the main thread with the
       DearPyGui window on macOS, so tkinter is avoided there).
Windows / Linux: tkinter, falling back to zenity on Linux.
Each function returns a path string, or None if the user cancelled.
"""

import shutil
import subprocess
import sys

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"]
SAVE_TYPES = [("PNG image", ".png"), ("JPEG image", ".jpg"), ("Bitmap", ".bmp")]


def _applescript_string(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _osascript(script):
    proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:  # -128 = user cancelled
        return None
    path = proc.stdout.strip()
    return path or None


def _zenity(args):
    if not shutil.which("zenity"):
        raise RuntimeError("No file dialog available: install python3-tk or zenity.")
    proc = subprocess.run(["zenity", "--file-selection"] + args, capture_output=True, text=True)
    path = proc.stdout.strip()
    return path if proc.returncode == 0 and path else None


def _tk_dialog(kind, **options):
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "open":
            path = filedialog.askopenfilename(parent=root, **options)
        else:
            path = filedialog.asksaveasfilename(parent=root, **options)
    finally:
        root.destroy()
    return path or None


def open_image_dialog():
    if sys.platform == "darwin":
        return _osascript(
            'POSIX path of (choose file with prompt "Choose an image to compress" '
            'of type {"public.image"})'
        )
    patterns = " ".join("*" + ext for ext in IMAGE_EXTENSIONS)
    try:
        return _tk_dialog("open", title="Choose an image to compress",
                          filetypes=[("Images", patterns), ("All files", "*.*")])
    except ImportError:
        return _zenity(["--title=Choose an image to compress", f"--file-filter=Images | {patterns}"])


def save_image_dialog(default_name):
    if sys.platform == "darwin":
        return _osascript(
            "POSIX path of (choose file name with prompt "
            f'"Save compressed image" default name {_applescript_string(default_name)})'
        )
    try:
        return _tk_dialog("save", title="Save compressed image", initialfile=default_name,
                          defaultextension=".png",
                          filetypes=[(label, "*" + ext) for label, ext in SAVE_TYPES])
    except ImportError:
        return _zenity(["--save", "--confirm-overwrite", "--title=Save compressed image",
                        f"--filename={default_name}"])
