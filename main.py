"""Image Compression Visualizer.

Config Menu -> pick an image, technique, amount; run compression; save result.
Image       -> original and compressed image.
Visualizer  -> the image as a matrix, the math summary, and a Desmos graph.
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
# The SVD costs about m n min(m, n), so very large photos can be scaled down to
# a working copy first. The saved image has the working copy's size.
RESOLUTIONS = {"512 px (fast)": 512, "1024 px (balanced)": 1024, "2048 px (sharp)": 2048, "Full size (slow)": None}
DEFAULT_RESOLUTION = "1024 px (balanced)"
# Enough to show a whole 384x512 image at once. Dear ImGui tables allow at most
# 512 columns, and one of them holds the row index, so 511 value columns fit.
MATRIX_ROWS, MATRIX_COLS = 384, 511
MATRIX_VIEW_H = 280  # the table scrolls inside this height
SHARP_MIN_CELL_PX = 6  # full-screen matrix cells are drawn sharply once they are this wide
LABEL_MIN_CELL_PX = 30  # ...and get numbers once they are this wide
VIEW_ORIGINAL, VIEW_COMPRESSED, VIEW_DIFF = "Original A", "Compressed A_k", "Difference A - A_k"

# Controls that need an image (or a result) before they can be used.
IMAGE_CONTROLS = ["amount", "compress_btn", "matrix_view", "matrix_channel", "row_start", "col_start",
                  "fullscreen_btn"]
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
        self.overview = None  # block shown in the full-screen matrix, for hover lookups


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
    if side is not None and max(full_w, full_h) > side:
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
        # Clip before the uint8 cast: out-of-range values would wrap (260 -> 4) and show as speckles.
        pixels = np.rint(np.clip(state.result.image_rgb, 0.0, 255.0)).astype(np.uint8)
        Image.fromarray(pixels).save(path)
        set_status(f"Saved {path}")
    except Exception as exc:
        set_status(f"Could not save: {exc}")


def on_open_desmos(sender, app_data, builder):
    if state.result is None:
        return
    try:
        graph = builder(state.analysis[state.result_method], state.result, current_channel(),
                        dpg.get_value("row_start"), dpg.get_value("col_start"))
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
    build_matrix("matrix_group")
    if dpg.is_item_shown("matrix_popup"):
        build_matrix_overview()


def matrix_block():
    """The matrix shown in the visualizer, or a message saying why there is none.

    Returns (message, None) or (None, dict with data, mode, channel, max_abs, rows, cols).
    """
    if state.image is None:
        return "Load an image to see it as a matrix of numbers.", None
    view = dpg.get_value("matrix_view")
    channel = current_channel()
    original = state.image[:, :, channel]
    if view != VIEW_ORIGINAL and state.result is None:
        return "Run compression first to see A_k.", None
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
    return None, dict(data=data, mode=mode, channel=channel, max_abs=max_abs, rows=rows, cols=cols)


def add_matrix_caption(block, parent):
    h, w = block["data"].shape
    rows, cols = block["rows"], block["cols"]
    dpg.add_text(
        f"{viz.CHANNEL_NAMES[block['channel']]} channel: a {h}x{w} matrix, one entry per pixel "
        f"(0 = dark, 255 = bright).\n"
        f"Showing rows {rows.start}-{rows.stop - 1}, columns {cols.start}-{cols.stop - 1}.",
        parent=parent,
    )
    if block["mode"] == "diff":
        dpg.add_text(f"Blue = A_k too dark, red = too bright. Largest error here: {block['max_abs']:.0f}",
                     parent=parent, color=(170, 170, 170))


def build_matrix(parent):
    """Fill the parent group with the scrolling matrix table."""
    dpg.delete_item(parent, children_only=True)
    message, block = matrix_block()
    if message:
        dpg.add_text(message, parent=parent)
        return
    add_matrix_caption(block, parent)
    data, mode, channel, max_abs = block["data"], block["mode"], block["channel"], block["max_abs"]

    # Scroll inside the table, keep the index row/column pinned, and let the
    # clipper skip rows that are off screen so large views stay responsive.
    with dpg.table(parent=parent, header_row=True, borders_innerH=True, borders_innerV=True,
                   borders_outerH=True, borders_outerV=True, policy=dpg.mvTable_SizingFixedFit,
                   scrollX=True, scrollY=True, freeze_rows=1, freeze_columns=1, clipper=True,
                   height=MATRIX_VIEW_H) as table:
        dpg.add_table_column(label="i \\ j")
        for c in block["cols"]:
            dpg.add_table_column(label=str(c))
        for i, r in enumerate(block["rows"]):
            with dpg.table_row():
                dpg.add_text(str(r), color=(170, 170, 170))
                for j, c in enumerate(block["cols"]):
                    value = float(data[r, c])
                    bg, fg = viz.cell_colors(value, mode, channel, max_abs)
                    dpg.add_text(f"{value:4.0f}", color=fg)
                    dpg.highlight_table_cell(table, i, j + 1, bg)


def build_matrix_overview():
    """Zoomable view of the whole matrix for the full-screen window. It starts zoomed out so
    every entry fits; the mouse wheel zooms, dragging pans, and numbers appear once cells are
    big enough."""
    parent = "matrix_popup_group"
    state.overview = None
    with dpg.mutex():
        dpg.delete_item(parent, children_only=True)
        old_texture = state.textures.pop("overview", None)
        if old_texture is not None:
            dpg.delete_item(old_texture)
    message, block = matrix_block()
    if message:
        dpg.add_text(message, parent=parent)
        return
    # Unlike the table, the plot has no column limit, so show every row and column.
    h, w = block["data"].shape
    block = dict(block, rows=range(h), cols=range(w))
    add_matrix_caption(block, parent)
    dpg.add_text("Scroll to zoom, drag to pan, double-click to fit. Numbers appear when zoomed in.",
                 tag="overview_hover", parent=parent, color=(170, 170, 170))

    sub = block["data"]
    if block["mode"] == "diff":
        colormap = "colormap_diff"
        scale = (-max(block["max_abs"], 1e-9), max(block["max_abs"], 1e-9))
    else:
        colormap = f"colormap_value_{block['channel']}"
        scale = (0.0, 255.0)

    # One rectangle per entry is too slow for a whole photo, so the matrix is drawn as a
    # single texture, and update_overview_labels adds sharp cells for the part in view.
    rgba = viz.cell_color_array(sub, block["mode"], block["channel"], block["max_abs"])
    texture = dpg.add_raw_texture(w, h, rgba.ravel(), format=dpg.mvFormat_Float_rgba,
                                  parent="texture_registry")
    state.textures["overview"] = texture

    # x is the column index and y the row index; cell (r, c) covers [c, c + 1] x [r, r + 1].
    # The y axis is inverted so row 0 is at the top. Images and heat series both put their
    # first row at the top of their bounds, so they are flipped to match.
    with dpg.plot(tag="overview_plot", parent=parent, width=-1, height=-1, equal_aspects=True,
                  no_menus=True, no_box_select=True, no_mouse_pos=True):
        dpg.add_plot_axis(dpg.mvXAxis, label="column j", tag="overview_x", opposite=True)
        with dpg.plot_axis(dpg.mvYAxis, label="row i", tag="overview_y", invert=True):
            dpg.add_image_series(texture, (0, 0), (w, h), uv_min=(0, 1), uv_max=(1, 0))
    dpg.bind_colormap("overview_plot", colormap)
    state.overview = dict(block=block, sub=sub, scale=scale, labels=None)


def update_overview_labels():
    """Draw the cells in view sharply once zoomed in (the texture blurs when enlarged), and
    number them once they are wide enough to read. Doing this for every cell at once is too
    slow, so only the visible ones get a copy drawn on top."""
    overview = state.overview
    if overview is None or not dpg.does_item_exist("overview_plot"):
        return
    rows, cols = overview["block"]["rows"], overview["block"]["cols"]
    x_min, x_max = dpg.get_axis_limits("overview_x")
    y_min, y_max = dpg.get_axis_limits("overview_y")
    plot_w = dpg.get_item_rect_size("overview_plot")[0]
    cell_px = plot_w / max(x_max - x_min, 1e-9)

    key = None
    if cell_px >= SHARP_MIN_CELL_PX:
        c0, c1 = max(cols.start, int(np.floor(x_min))), min(cols.stop, int(np.ceil(x_max)))
        r0, r1 = max(rows.start, int(np.floor(y_min))), min(rows.stop, int(np.ceil(y_max)))
        if c0 < c1 and r0 < r1:
            key = (c0, c1, r0, r1, cell_px >= LABEL_MIN_CELL_PX)
    if key == overview["labels"]:
        return
    overview["labels"] = key
    if dpg.does_item_exist("overview_labels"):
        dpg.delete_item("overview_labels")
    if key is None:
        return
    c0, c1, r0, r1, numbered = key
    part = overview["sub"][r0 - rows.start:r1 - rows.start, c0 - cols.start:c1 - cols.start]
    dpg.add_heat_series(np.flipud(part).ravel().tolist(), part.shape[0], part.shape[1],
                        tag="overview_labels", parent="overview_y", format="%.0f" if numbered else "",
                        scale_min=overview["scale"][0], scale_max=overview["scale"][1],
                        bounds_min=(c0, r0), bounds_max=(c1, r1), contribute_to_bounds=False)


def on_mouse_move():
    """Show the row, column, and value under the mouse in the full-screen matrix."""
    if state.overview is None or not dpg.does_item_exist("overview_plot"):
        return
    if not dpg.is_item_hovered("overview_plot"):
        return
    x, y = dpg.get_plot_mouse_pos()
    rows, cols = state.overview["block"]["rows"], state.overview["block"]["cols"]
    r, c = int(np.floor(y)), int(np.floor(x))
    if r in rows and c in cols:
        value = state.overview["sub"][r - rows.start, c - cols.start]
        dpg.set_value("overview_hover", f"Row {r}, column {c}: {value:.0f}")


def on_matrix_change():
    refresh_matrix()
    refresh_summary()


def on_open_fullscreen():
    """Show the matrix block in a zoomable window that covers the viewport."""
    width, height = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
    dpg.configure_item("matrix_popup", pos=(0, 0), width=width, height=height, show=True)
    dpg.focus_item("matrix_popup")
    build_matrix_overview()


def on_close_fullscreen():
    dpg.configure_item("matrix_popup", show=False)
    # Drop the plot and its texture so they are not kept in memory while hidden.
    state.overview = None
    with dpg.mutex():
        dpg.delete_item("matrix_popup_group", children_only=True)
        old_texture = state.textures.pop("overview", None)
        if old_texture is not None:
            dpg.delete_item(old_texture)


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
    with dpg.colormap_registry():
        for channel in range(len(viz.CHANNEL_NAMES)):
            dpg.add_colormap(viz.colormap_colors("value", channel), False, tag=f"colormap_value_{channel}")
        dpg.add_colormap(viz.colormap_colors("diff", 0), False, tag="colormap_diff")
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
            dpg.add_button(label="Full screen", tag="fullscreen_btn", callback=on_open_fullscreen)
        dpg.add_group(tag="matrix_group")
        dpg.add_text("Desmos (opens in your browser). The graph uses the 2x2 block at the top-left cell\n"
                     "of the matrix above, in the channel selected.", color=(170, 170, 170))
        dpg.add_group(tag="desmos_buttons", horizontal=True)
        dpg.add_separator()
        dpg.add_text("", tag="summary")

    with dpg.window(label="Image Matrix (full screen)", tag="matrix_popup", show=False, no_collapse=True,
                    no_move=True, no_resize=True, on_close=on_close_fullscreen):
        dpg.add_button(label="Close full screen", callback=on_close_fullscreen)
        dpg.add_group(tag="matrix_popup_group")

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=on_mouse_move)


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
    while dpg.is_dearpygui_running():
        update_overview_labels()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
