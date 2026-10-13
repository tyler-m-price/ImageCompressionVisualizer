"""Helpers for the compression visualizer: matrix cell styling and Desmos graphs."""

import os
import tempfile
import webbrowser
from pathlib import Path

from PyDesmos import Graph

CHANNEL_NAMES = ["Red", "Green", "Blue"]
DESMOS_LIST_LIMIT = 10000  # Desmos rejects list literals longer than this


# ---------------------------------------------------------------- matrix view

def cell_colors(value, mode, channel, max_abs_diff=1.0):
    """Background and text colors for one matrix cell.

    mode "value": tint by the channel color, brightness = value / 255.
    mode "diff": diverging blue (positive) / red (negative) scale.
    """
    if mode == "diff":
        level = min(1.0, abs(value) / max(max_abs_diff, 1e-9))
        shade = int(255 * (1.0 - level))
        bg = (shade, shade, 255) if value >= 0 else (255, shade, shade)
        return bg + (255,), (0, 0, 0, 255)

    level = max(0.0, min(255.0, float(value)))
    rgb = [0.0, 0.0, 0.0]
    rgb[channel] = level
    # Lift the other channels a little so dark cells are not pure black.
    bg = tuple(int(c * 0.85 + 25) for c in rgb)
    brightness = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    text = (0, 0, 0, 255) if brightness > 110 else (255, 255, 255, 255)
    return bg + (255,), text


# ---------------------------------------------------------------- Desmos

def desmos_number(x, decimals=4):
    """Fixed-point text (Desmos reads '1e-5' as Euler's e times -5)."""
    text = f"{float(x):.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text


def desmos_list(values, decimals=4):
    values = list(values)
    if len(values) > DESMOS_LIST_LIMIT:
        raise ValueError(f"Desmos lists are limited to {DESMOS_LIST_LIMIT} entries")
    return r"\left[" + ",".join(desmos_number(v, decimals) for v in values) + r"\right]"


def new_graph(name):
    """Graph whose HTML file lives in a temp folder (works inside a packaged app)."""
    folder = Path(tempfile.gettempdir()) / "ImageCompressionVisualizer"
    folder.mkdir(parents=True, exist_ok=True)
    graph = Graph(str(folder / name))
    # PyDesmos leaves the calculator div unsized; make it fill the browser window.
    graph.html = graph.html.replace(
        '<div id="calculator"></div>',
        '<div id="calculator" style="position:absolute;top:0;left:0;width:100vw;height:100vh;"></div>',
    )
    return graph


def expr(graph, latex, **settings):
    """Add raw LaTeX (skips PyDesmos' to_latex rewriting, which mangles names
    such as \\beta -> \\b\\eta)."""
    graph.set_expression(latex=latex, **settings)


def slider(graph, name, value, low, high, step=1):
    graph.set_expression(
        latex=f"{name}={desmos_number(value)}",
        sliderBounds={"min": desmos_number(low), "max": desmos_number(high), "step": desmos_number(step)},
    )


def set_bounds(graph, left, right, bottom, top):
    graph.html += (
        "\tcalculator.setMathBounds({"
        f"left: {left}, right: {right}, bottom: {bottom}, top: {top}"
        "});\n"
    )


def open_graph(graph):
    graph.save()
    webbrowser.open(Path(os.path.abspath(graph.file_name)).as_uri())
    return graph.file_name
