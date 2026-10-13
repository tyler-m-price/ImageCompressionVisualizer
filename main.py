"""Image Compression Visualizer.

Config Menu -> pick an image, technique, amount; run compression; save result.
Image       -> original and compressed image.
Visualizer  -> the image as a matrix, the math summary, and Desmos graphs.
"""

import os
import threading
import time
import traceback

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image, ImageOps

import compressors
import visualizer as viz
from file_dialogs import IMAGE_EXTENSIONS, open_image_dialog, save_image_dialog

VIEWPORT_W, VIEWPORT_H = 1360, 880
IMAGE_PANEL_W = 560
CONFIG_H = 330
DISPLAY_MAX_W, DISPLAY_MAX_H = 520, 370

AMOUNTS = [f"{p}%" for p in range(0, 101, 5)]
DEFAULT_AMOUNT = "50%"
# The decomposition cost grows like n^3, so large photos are scaled down to a
# working copy first. The saved image has the working copy's size.
RESOLUTIONS = {"256 px (fast)": 256, "384 px (balanced)": 384, "512 px (slow)": 512}
DEFAULT_RESOLUTION = "384 px (balanced)"
MATRIX_ROWS, MATRIX_COLS = 12, 12
VIEW_ORIGINAL, VIEW_COMPRESSED, VIEW_DIFF = "Original A", "Compressed A_k", "Difference A - A_k"

# Controls that need an image (or a result) before they can be used.
IMAGE_CONTROLS = ["amount", "compress_btn", "matrix_view", "matrix_channel", "row_start", "col_start"]
BUSY_CONTROLS = ["select_btn", "resolution", "method"] + IMAGE_CONTROLS


class AppState:
    def __init__(self):
        self.path = None
        self.source = None  # full-size PIL image
        self.image = None  # working copy, float (H, W, 3) in [0, 255]
        self.analysis = {}  # method name -> cached analyze() output
        self.result = None
        self.result_method = None
        self.textures = {}
        self.busy = False


state = AppState()


# ---------------------------------------------------------------- helpers

def set_status(text):
    dpg.set_value("status", text)


def set_progress(fraction, text=""):
    dpg.set_value("progress", fraction)
    dpg.configure_item("progress", overlay=text)


def set_enabled(tags, enabled):
    for tag in tags:
        dpg.configure_item(tag, enabled=enabled)


def refresh_controls():
    has_image = state.image is not None
    has_result = state.result is not None
    set_enabled(["select_btn", "resolution", "method"], not state.busy)
    set_enabled(IMAGE_CONTROLS, has_image and not state.busy)
    set_enabled(["save_btn"], has_result and not state.busy)
    for button in dpg.get_item_children("desmos_buttons", 1):
        dpg.configure_item(button, enabled=has_result and not state.busy)


def current_compressor():
    return compressors.get(dpg.get_value("method"))


def current_channel():
    return viz.CHANNEL_NAMES.index(dpg.get_value("matrix_channel"))


def show_image(slot, image_rgb):
    """Upload an (H, W, 3) array as a texture and show it in the slot's group."""
    group = f"{slot}_group"
    old_texture = state.textures.pop(slot, None)
    with dpg.mutex():
        dpg.delete_item(group, children_only=True)
        if old_texture is not None:
            dpg.delete_item(old_texture)
        if image_rgb is None:
            dpg.add_text("(nothing yet)", parent=group, color=(150, 150, 150))
            return
        h, w = image_rgb.shape[:2]
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = image_rgb / 255.0
        texture = dpg.add_raw_texture(w, h, rgba.ravel(), format=dpg.mvFormat_Float_rgba,
                                      parent="texture_registry")
        scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h)
        dpg.add_image(texture, width=max(1, int(w * scale)), height=max(1, int(h * scale)), parent=group)
    state.textures[slot] = texture


def clear_result():
    state.result = None
    state.result_method = None
    show_image("compressed", None)
    dpg.set_value("compressed_label", "Compressed")
    dpg.set_value("summary", "Run compression to see the math behind the result.")


# ---------------------------------------------------------------- image loading

def prepare_working_image():
    side = RESOLUTIONS[dpg.get_value("resolution")]
    img = state.source
    full_w, full_h = img.size
    if max(full_w, full_h) > side:
        scale = side / max(full_w, full_h)
        img = img.resize((max(1, round(full_w * scale)), max(1, round(full_h * scale))), Image.LANCZOS)
    state.image = np.asarray(img, dtype=float)
    state.analysis.clear()
    clear_result()

    h, w = state.image.shape[:2]
    name = os.path.basename(state.path)
    dpg.set_value("file_label", name)
    dpg.set_value("original_label", f"Original  ({full_w}x{full_h}, working copy {w}x{h})")
    show_image("original", state.image)

    dpg.configure_item("row_start", max_value=max(0, h - MATRIX_ROWS))
    dpg.configure_item("col_start", max_value=max(0, w - MATRIX_COLS))
    dpg.set_value("row_start", min(dpg.get_value("row_start"), max(0, h - MATRIX_ROWS)))
    dpg.set_value("col_start", min(dpg.get_value("col_start"), max(0, w - MATRIX_COLS)))
    refresh_matrix()
    refresh_controls()


def on_select_image():
    path = open_image_dialog()
    if not path:
        return
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        set_status(f"Could not open image: {exc}")
        return
    state.path = path
    state.source = img
    prepare_working_image()
    set_progress(0.0)
    set_status("Image loaded. Pick a compression amount and press Compress.")


def on_resolution_change():
    if state.source is not None:
        prepare_working_image()
        set_status("Working resolution changed; the decomposition will be recomputed.")


def on_method_change():
    compressor = current_compressor()
    dpg.set_value("amount_help", compressor.amount_help)
    dpg.delete_item("desmos_buttons", children_only=True)
    for label, builder in compressor.desmos_views():
        dpg.add_button(label=label, parent="desmos_buttons", callback=on_open_desmos, user_data=builder)
    if state.image is not None:
        clear_result()
        refresh_matrix()
    refresh_controls()


# ---------------------------------------------------------------- compression

def on_compress():
    if state.busy or state.image is None:
        return
    method = dpg.get_value("method")
    amount = int(dpg.get_value("amount").rstrip("%"))
    state.busy = True
    refresh_controls()
    threading.Thread(target=run_compression, args=(method, amount), daemon=True).start()


def run_compression(method, amount):
    compressor = compressors.get(method)
    try:
        analysis = state.analysis.get(method)
        if analysis is None:
            started = time.time()
            set_status("Computing the decomposition (only needed once per image)...")
            analysis = compressor.analyze(state.image, progress=set_progress)
            state.analysis[method] = analysis
            set_status(f"Decomposition finished in {time.time() - started:.1f} s.")
        result = compressor.compress(analysis, amount)

        state.result = result
        state.result_method = method
        show_image("compressed", result.image_rgb)
        dpg.set_value("compressed_label", f"Compressed  ({method}, {amount}%)")
        refresh_summary()
        refresh_matrix()
        set_progress(1.0, "Done")
        set_status("Compression finished. Try another amount: it is instant now.")
    except Exception as exc:
        traceback.print_exc()
        set_progress(0.0, "Failed")
        set_status(f"Compression failed: {exc}")
    finally:
        state.busy = False
        refresh_controls()


def on_save():
    if state.result is None:
        return
    stem = os.path.splitext(os.path.basename(state.path))[0]
    amount = dpg.get_value("amount").rstrip("%")
    path = save_image_dialog(f"{stem}_compressed_{amount}pct.png")
    if not path:
        return
    if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
        path += ".png"
    try:
        Image.fromarray(state.result.image_rgb.astype(np.uint8)).save(path)
        set_status(f"Saved {path}")
    except Exception as exc:
        set_status(f"Could not save: {exc}")


def on_open_desmos(sender, app_data, builder):
    if state.result is None:
        return
    try:
        graph = builder(state.analysis[state.result_method], state.result,
                        current_channel(), dpg.get_value("row_start"))
        viz.open_graph(graph)
        set_status("Opened the Desmos graph in your browser (needs an internet connection).")
    except Exception as exc:
        traceback.print_exc()
        set_status(f"Could not build the Desmos graph: {exc}")


# ---------------------------------------------------------------- visualizer

def refresh_summary():
    if state.result is None:
        return
    text = compressors.get(state.result_method).summary(state.result, current_channel())
    dpg.set_value("summary", text)


def refresh_matrix():
    dpg.delete_item("matrix_group", children_only=True)
    if state.image is None:
        dpg.add_text("Load an image to see it as a matrix of numbers.", parent="matrix_group")
        return

    view = dpg.get_value("matrix_view")
    channel = current_channel()
    original = state.image[:, :, channel]
    if view != VIEW_ORIGINAL and state.result is None:
        dpg.add_text("Run compression first to see A_k.", parent="matrix_group")
        return
    if view == VIEW_ORIGINAL:
        data, mode = original, "value"
    elif view == VIEW_COMPRESSED:
        data, mode = state.result.image_rgb[:, :, channel], "value"
    else:
        data, mode = original - state.result.image_rgb[:, :, channel], "diff"
    max_abs = float(np.abs(data).max()) if mode == "diff" else 255.0

    h, w = data.shape
    r0, c0 = dpg.get_value("row_start"), dpg.get_value("col_start")
    rows = range(r0, min(h, r0 + MATRIX_ROWS))
    cols = range(c0, min(w, c0 + MATRIX_COLS))

    dpg.add_text(
        f"{viz.CHANNEL_NAMES[channel]} channel: a {h}x{w} matrix, one entry per pixel (0 = dark, 255 = bright).\n"
        f"Showing rows {rows.start}-{rows.stop - 1}, columns {cols.start}-{cols.stop - 1}.",
        parent="matrix_group",
    )
    if mode == "diff":
        dpg.add_text(f"Blue = A_k too dark, red = too bright. Largest error here: {max_abs:.0f}",
                     parent="matrix_group", color=(170, 170, 170))

    with dpg.table(parent="matrix_group", header_row=True, borders_innerH=True, borders_innerV=True,
                   borders_outerH=True, borders_outerV=True, policy=dpg.mvTable_SizingFixedFit) as table:
        dpg.add_table_column(label="i \\ j")
        for c in cols:
            dpg.add_table_column(label=str(c))
        for i, r in enumerate(rows):
            with dpg.table_row():
                dpg.add_text(str(r), color=(170, 170, 170))
                for j, c in enumerate(cols):
                    value = float(data[r, c])
                    bg, fg = viz.cell_colors(value, mode, channel, max_abs)
                    dpg.add_text(f"{value:4.0f}", color=fg)
                    dpg.highlight_table_cell(table, i, j + 1, bg)


def on_matrix_change():
    refresh_matrix()
    refresh_summary()


# ---------------------------------------------------------------- layout

def build_disabled_theme():
    grey_text = (115, 115, 115)
    flat = (42, 42, 45)
    with dpg.theme() as theme:
        for item_type in (dpg.mvButton, dpg.mvCombo, dpg.mvInputInt):
            with dpg.theme_component(item_type, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Text, grey_text)
                for col in (dpg.mvThemeCol_Button, dpg.mvThemeCol_ButtonHovered, dpg.mvThemeCol_ButtonActive,
                            dpg.mvThemeCol_FrameBg, dpg.mvThemeCol_FrameBgHovered, dpg.mvThemeCol_FrameBgActive):
                    dpg.add_theme_color(col, flat)
    return theme


def build_ui():
    dpg.add_texture_registry(tag="texture_registry")
    right_w = VIEWPORT_W - IMAGE_PANEL_W
    window_flags = dict(no_close=True, no_collapse=True)

    with dpg.window(label="Image", width=IMAGE_PANEL_W, height=VIEWPORT_H, pos=(0, 0), **window_flags):
        dpg.add_text("Original", tag="original_label")
        dpg.add_group(tag="original_group")
        dpg.add_separator()
        dpg.add_text("Compressed", tag="compressed_label")
        dpg.add_group(tag="compressed_group")

    with dpg.window(label="Config Menu", width=right_w, height=CONFIG_H, pos=(IMAGE_PANEL_W, 0),
                    **window_flags):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select Image...", tag="select_btn", callback=on_select_image)
            dpg.add_text("No image selected", tag="file_label")
        dpg.add_combo(list(RESOLUTIONS), default_value=DEFAULT_RESOLUTION, label="Working resolution",
                      tag="resolution", width=220, callback=on_resolution_change)
        dpg.add_combo(compressors.list_names(), default_value=compressors.list_names()[0],
                      label="Compression technique", tag="method", width=220, callback=on_method_change)
        dpg.add_combo(AMOUNTS, default_value=DEFAULT_AMOUNT, label="Compression amount", tag="amount", width=220)
        dpg.add_text("", tag="amount_help", color=(170, 170, 170), wrap=right_w - 30)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Compress", tag="compress_btn", width=120, callback=on_compress)
            dpg.add_button(label="Save compressed image...", tag="save_btn", callback=on_save)
        dpg.add_progress_bar(tag="progress", default_value=0.0, width=-1)
        dpg.add_text("Select an image to begin.", tag="status", wrap=right_w - 30)

    with dpg.window(label="Compression Visualizer", width=right_w, height=VIEWPORT_H - CONFIG_H,
                    pos=(IMAGE_PANEL_W, CONFIG_H), **window_flags):
        with dpg.group(horizontal=True):
            dpg.add_combo([VIEW_ORIGINAL, VIEW_COMPRESSED, VIEW_DIFF], default_value=VIEW_ORIGINAL,
                          tag="matrix_view", width=170, callback=on_matrix_change)
            dpg.add_combo(viz.CHANNEL_NAMES, default_value=viz.CHANNEL_NAMES[0], tag="matrix_channel",
                          width=80, callback=on_matrix_change)
            dpg.add_input_int(label="row", tag="row_start", width=110, default_value=0, min_value=0,
                              max_value=0, min_clamped=True, max_clamped=True, step=1,
                              step_fast=MATRIX_ROWS, callback=on_matrix_change)
            dpg.add_input_int(label="col", tag="col_start", width=110, default_value=0, min_value=0,
                              max_value=0, min_clamped=True, max_clamped=True, step=1,
                              step_fast=MATRIX_COLS, callback=on_matrix_change)
        dpg.add_group(tag="matrix_group")
        dpg.add_separator()
        dpg.add_text("", tag="summary")
        dpg.add_text("Desmos (opens in your browser; the scanline graph uses the first row shown above):",
                     color=(170, 170, 170))
        dpg.add_group(tag="desmos_buttons", horizontal=True)


def main():
    dpg.create_context()
    build_ui()
    dpg.bind_theme(build_disabled_theme())
    on_method_change()
    show_image("original", None)
    clear_result()
    refresh_matrix()
    refresh_controls()

    dpg.create_viewport(title="Image Compression Visualizer", width=VIEWPORT_W, height=VIEWPORT_H)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
