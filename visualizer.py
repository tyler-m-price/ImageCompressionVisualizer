"""Helpers for the compression visualizer: matrix cell styling and Desmos graphs."""

import json
import os
import tempfile
import webbrowser
from pathlib import Path

from PyDesmos import Graph

CHANNEL_NAMES = ["Red", "Green", "Blue"]
DESMOS_LIST_LIMIT = 10000  # Desmos rejects list literals longer than this
DESMOS_API_VERSION = "v1.11"


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
    # Declare UTF-8 so Desmos symbols (e.g. the summation sign) render correctly.
    graph.html = '<meta charset="utf-8">\n' + graph.html
    # PyDesmos pins API v1.7, which cannot draw one polygon per list entry.
    graph.html = graph.html.replace("/api/v1.7/", f"/api/{DESMOS_API_VERSION}/")
    # advancedStyling enables per-expression styling such as colorLatex.
    graph.html = graph.html.replace(
        "Desmos.GraphingCalculator(elt);", "Desmos.GraphingCalculator(elt, {advancedStyling: true});"
    )
    # PyDesmos leaves the calculator div unsized; make it fill the browser window.
    graph.html = graph.html.replace(
        '<div id="calculator"></div>',
        '<div id="calculator" style="position:absolute;top:0;left:0;width:100vw;height:100vh;"></div>',
    )
    graph.color_latex = {}  # expression id -> LaTeX color (applied in finish())
    graph.data_latex = []  # big list definitions (added in finish(), inside a collapsed folder)
    graph.bounds_js = ""
    graph.finished = False
    return graph


def expr(graph, latex, **settings):
    """Add raw LaTeX (skips PyDesmos' to_latex rewriting, which mangles names
    such as \\beta -> \\b\\eta).

    colorLatex (a color or list of colors computed in Desmos) is not accepted
    by setExpression, so it is recorded here and applied by finish().
    """
    color_latex = settings.pop("colorLatex", None)
    if color_latex is not None:
        settings["id"] = f"styled{len(graph.color_latex)}"
        graph.color_latex[settings["id"]] = color_latex
    graph.set_expression(latex=latex, **settings)


def data(graph, latex):
    """Define a large precomputed list. Desmos typesets every open expression,
    about 1 ms per number, so these go into a collapsed "Data" folder that
    finish() adds without ever showing them expanded."""
    graph.data_latex.append(latex)


def _latex_value(value):
    return value if isinstance(value, str) else desmos_number(value)


def slider(graph, name, value, low, high, step=1):
    """Slider; bounds may be numbers or LaTeX strings (e.g. "2\\pi"). step=None is continuous."""
    # Desmos needs a step key; an empty string means a continuous slider.
    bounds = {"min": _latex_value(low), "max": _latex_value(high),
              "step": "" if step is None else _latex_value(step)}
    graph.set_expression(latex=f"{name}={_latex_value(value)}", sliderBounds=bounds)


def curve(graph, x_latex, y_latex, t_min="0", t_max="1", **settings):
    """Parametric curve (x(t), y(t)) for t in [t_min, t_max]."""
    graph.set_expression(
        latex=rf"\left({x_latex},{y_latex}\right)",
        parametricDomain={"min": _latex_value(t_min), "max": _latex_value(t_max)},
        **settings,
    )


def stepped_list(graph, name, blocks, index="k", decimals=3):
    """Define list `name` to equal blocks[index - 1] as the slider `index` moves.

    Desmos list literals are capped at DESMOS_LIST_LIMIT entries, so the
    blocks are concatenated into chunk lists name_{1}, name_{2}, ... and
    `name` slices the right one with a piecewise expression. (Precomputing
    every step keeps Desmos fast; summing layers inside Desmos with list
    comprehensions made the calculator hang.)
    """
    length = len(blocks[0])
    per_chunk = max(1, DESMOS_LIST_LIMIT // length)
    branches = []
    for number, start in enumerate(range(0, len(blocks), per_chunk), 1):
        chunk = blocks[start:start + per_chunk]
        chunk_name = f"{name}_{{{number}}}"
        data(graph, f"{chunk_name}={desmos_list([x for block in chunk for x in block], decimals)}")
        # Block number index - start inside this chunk spans entries lo..hi.
        lo = rf"{length}\left({index}-{start + 1}\right)+1"
        hi = rf"{length}\left({index}-{start}\right)" if start else f"{length}{index}"
        branches.append((start + len(chunk), rf"{chunk_name}\left[{lo},...,{hi}\right]"))
    if len(branches) == 1:
        expr(graph, f"{name}={branches[0][1]}")
        return
    cases = ",".join(rf"{index}\le{end}:{piece}" for end, piece in branches[:-1])
    expr(graph, rf"{name}=\left\{{{cases},{branches[-1][1]}\right\}}")


def set_bounds(graph, left, right, bottom, top, square=False):
    """Viewport to show once the page loads. square=True widens one axis so
    one unit is the same length in x and y (circles stay round)."""
    bounds = f"{{left: {left}, right: {right}, bottom: {bottom}, top: {top}}}"
    if not square:
        graph.bounds_js = f"\tcalculator.setMathBounds({bounds});\n"
        return
    graph.bounds_js = (
        f"\t(function () {{ var b = {bounds};\n"
        "\t  var px = calculator.graphpaperBounds.pixelCoordinates;\n"
        "\t  var s = Math.max((b.right - b.left) / px.width, (b.top - b.bottom) / px.height);\n"
        "\t  var cx = (b.left + b.right) / 2, cy = (b.top + b.bottom) / 2;\n"
        "\t  calculator.setMathBounds({left: cx - s * px.width / 2, right: cx + s * px.width / 2,\n"
        "\t                           bottom: cy - s * px.height / 2, top: cy + s * px.height / 2}); })();\n"
    )


def finish(graph):
    """Append the deferred data folder, colorLatex styling and viewport.

    Everything goes in through one setState call, so the big data lists are
    never typeset in the open expression list.
    """
    if graph.finished:
        return graph
    if graph.color_latex or graph.data_latex:
        folder = [{"type": "folder", "id": "data", "title": "Precomputed data", "collapsed": True}]
        folder += [{"type": "expression", "id": f"data{i}", "folderId": "data", "latex": latex}
                   for i, latex in enumerate(graph.data_latex)]
        graph.html += (
            f"\t(function () {{ var colors = {json.dumps(graph.color_latex)};\n"
            f"\t  var data = {json.dumps(folder if graph.data_latex else [])};\n"
            "\t  var state = calculator.getState();\n"
            "\t  state.expressions.list.forEach(function (e) { if (colors[e.id]) e.colorLatex = colors[e.id]; });\n"
            "\t  state.expressions.list = state.expressions.list.concat(data);\n"
            "\t  calculator.setState(state); })();\n"
        )
    graph.html += graph.bounds_js
    graph.finished = True
    return graph


def open_graph(graph):
    finish(graph)
    graph.save()
    webbrowser.open(Path(os.path.abspath(graph.file_name)).as_uri())
    return graph.file_name
