"""Rank-k SVD image compression.

Each color channel is an m x n matrix A. Its SVD A = U S V^T writes A as a sum
of rank-one "layers" s_i u_i v_i^T ordered by importance (s_1 >= s_2 >= ...).
Keeping only the first k layers gives the best rank-k approximation A_k
(Eckart-Young), stored with k(m + n + 1) numbers instead of m * n.

Compression amount is the fraction of singular values dropped:
  0%   -> keep full rank (no loss)
  100% -> keep one singular value (maximum compression)
"""

import numpy as np

import visualizer as viz
from compressors.base import Compressor, CompressionResult, register
from linalg_manual import frobenius_norm_sq, jacobi_svd, reconstruct

ESTIMATED_SWEEPS = 8  # typical Jacobi sweeps for photos; only used for the progress bar
PATCH = 6  # side of the pixel patch drawn in the Desmos transformation view


def _rank_from_amount(rank, amount_percent):
    amount = max(0.0, min(100.0, float(amount_percent)))
    if amount <= 0.0:
        return rank
    if amount >= 100.0:
        return 1
    kept = int(round((1.0 - amount / 100.0) * rank))
    return max(1, min(rank, kept))


def _clip_round(a):
    """Clamp to the displayable range [0, 255] and round to whole intensities."""
    return np.floor(np.minimum(np.maximum(a, 0.0), 255.0) + 0.5)


class SVDCompressor(Compressor):
    name = "SVD (rank-k approximation)"
    amount_help = "Compression % = share of singular values thrown away (0% keeps all of them)."

    def analyze(self, image_rgb, progress=None):
        image = np.asarray(image_rgb, dtype=float)
        channels = [image[:, :, c] for c in range(3)]

        # Grayscale images have identical channels: decompose only once.
        if (np.abs(channels[0] - channels[1]).max() == 0
                and np.abs(channels[0] - channels[2]).max() == 0):
            stack, channel_map = np.stack([channels[0]]), [0, 0, 0]
        else:
            stack, channel_map = np.stack(channels), [0, 1, 2]

        def on_sweep(sweep, fraction):
            if progress is not None:
                done = min(0.97, (sweep + fraction) / ESTIMATED_SWEEPS)
                progress(done, f"Jacobi sweep {sweep + 1}: orthogonalizing column pairs")

        u, s, v = jacobi_svd(stack, progress=on_sweep)
        u, s, v = list(u), list(s), list(v)  # one entry per decomposed matrix
        return {
            "image": image,
            "u": [u[i] for i in channel_map],
            "s": [s[i] for i in channel_map],
            "v": [v[i] for i in channel_map],
        }

    def compress(self, analysis, amount_percent):
        image = analysis["image"]
        height, width, _ = image.shape
        rank = min(height, width)
        k = _rank_from_amount(rank, amount_percent)

        rebuilt = {}
        channels = []
        for c in range(3):
            key = id(analysis["u"][c])  # shared decomposition for grayscale
            if key not in rebuilt:
                rebuilt[key] = _clip_round(reconstruct(analysis["u"][c], analysis["s"][c], analysis["v"][c], k))
            channels.append(rebuilt[key])
        compressed = np.stack(channels, axis=2)

        # ||A||_F^2 = sum s_i^2 and ||A - A_k||_F^2 = sum_{i>k} s_i^2.
        total = sum(frobenius_norm_sq(s) for s in analysis["s"])
        dropped = sum(frobenius_norm_sq(s[k:]) for s in analysis["s"])

        extra = {
            "k": k,
            "rank": rank,
            "shape": (height, width),
            "amount": amount_percent,
            "energy_ratio": (total - dropped) / total if total > 0 else 1.0,
            "relative_error": (dropped / total) ** 0.5 if total > 0 else 0.0,
            "stored_numbers": 3 * k * (height + width + 1),
            "raw_numbers": 3 * height * width,
            "sigmas": analysis["s"],
        }
        return CompressionResult(compressed, extra)

    def summary(self, result, channel):
        e = result.extra
        m, n = e["shape"]
        k, r = e["k"], e["rank"]
        sigmas = e["sigmas"][channel]
        ratio = e["stored_numbers"] / e["raw_numbers"]
        lines = [
            "Each channel is a matrix A (m x n), factored as A = U S V^T:",
            f"  A: {m}x{n}   U: {m}x{r}   S: {r}x{r} (diagonal)   V^T: {r}x{n}",
            "",
            "A_k keeps the k largest singular values s_1 >= ... >= s_k:",
            "  A_k = s_1 u_1 v_1^T + s_2 u_2 v_2^T + ... + s_k u_k v_k^T",
            f"  k = {k} of {r} singular values kept  (compression setting {e['amount']}%)",
            "",
            f"Storage per channel: k(m+n+1) = {k * (m + n + 1):,} numbers",
            f"           vs raw:   m*n      = {m * n:,} numbers  ({100 * ratio:.1f}%)",
            f"Energy kept   sum_{{i<=k}} s_i^2 / sum s_i^2 = {100 * e['energy_ratio']:.3f}%",
            f"Rel. error    ||A - A_k||_F / ||A||_F       = {100 * e['relative_error']:.3f}%",
        ]
        if ratio >= 1.0:
            lines.append("  (k is large enough that the factors take MORE space than the raw pixels)")
        top = ", ".join(f"{x:.1f}" for x in sigmas[:6])
        lines += ["", f"Largest singular values ({viz.CHANNEL_NAMES[channel]}): {top}, ..."]
        return "\n".join(lines)

    def desmos_views(self):
        return [
            ("Desmos: 2x2 block as a transformation", self._desmos_transform),
        ]

    # ------------------------------------------------------------ Desmos

    def _desmos_transform(self, analysis, result, channel, row, col):
        """A 2x2 block of A, [[a, b], [c, d]], seen as a map of the plane.

        The unit circle goes to an ellipse. Rotating an orthonormal pair
        (v_1, v_2) by an angle phi (theta is reserved for polar plots in
        Desmos) until A v_1 and A v_2 are perpendicular finds the
        block's singular vectors; that angle is the Jacobi rotation used in
        linalg_manual.jacobi_svd. The k slider swaps a, b, c, d for the
        entries of A_k, showing how compression reshapes the transformation
        and the surrounding pixels.
        """
        u, s, v = analysis["u"][channel], analysis["s"][channel], analysis["v"][channel]
        image = analysis["image"]
        height, width, _ = image.shape
        size = max(2, min(PATCH, height, width))
        row = max(0, min(height - size, int(row)))
        col = max(0, min(width - size, int(col)))
        rank = len(s)

        # Row i of us, column j of vp: the pieces of layer l that land on the
        # patch, so A_k[row+i, col+j] = sum_{l<k} us[i, l] * vp[j, l].
        us = (u[row:row + size, :] * s) / 255.0
        vp = v[col:col + size, :]
        block_layers = {
            "a": us[0] * vp[0], "b": us[0] * vp[1],
            "c": us[1] * vp[0], "d": us[1] * vp[1],
        }
        block_original = {
            "a": image[row, col, channel], "b": image[row, col + 1, channel],
            "c": image[row + 1, col, channel], "d": image[row + 1, col + 1, channel],
        }
        patches, running = [], np.zeros((size, size))
        for i in range(rank):
            running = running + us[:, i][:, None] * vp[:, i][None, :]
            patches.append(np.minimum(np.maximum(running, 0.0), 1.0).ravel())
        patch_original = image[row:row + size, col:col + size, channel].ravel() / 255.0

        name = viz.CHANNEL_NAMES[channel]
        g = viz.new_graph(f"svd_transform_{name.lower()}_r{row}_c{col}")

        # --- controls
        viz.slider(g, "k", result.extra["k"], 1, rank)
        viz.slider(g, r"\phi", 0, 0, r"2\pi", step=None)
        viz.slider(g, "q_{1}", 0.8, -1.5, 1.5, step=None)
        viz.slider(g, "q_{2}", 0.4, -1.5, 1.5, step=None)

        # --- the 2x2 block: A_k entries (sum of the first k layers) and the original A
        for key in "abcd":
            viz.expr(g, rf"{key}=\operatorname{{total}}\left(L_{{{key}}}\left[1,...,k\right]\right)")
        for i, key in enumerate("abcd", 1):
            viz.expr(g, rf"{key}_{{0}}=B_{{0}}\left[{i}\right]")

        # --- unit circle and its images: dashed = original block, red = A_k block
        viz.expr(g, "x^{2}+y^{2}=1", color="#999999")
        viz.curve(g, r"a_{0}\cos t+b_{0}\sin t", r"c_{0}\cos t+d_{0}\sin t", 0, r"2\pi",
                  color="#555555", lineStyle="DASHED")
        viz.curve(g, r"a\cos t+b\sin t", r"c\cos t+d\sin t", 0, r"2\pi", color="#c74440", lineWidth=3)

        # --- rotating orthonormal pair v_1, v_2 and their images
        viz.expr(g, r"v_{11}=\cos\phi")
        viz.expr(g, r"v_{12}=\sin\phi")
        viz.expr(g, r"v_{21}=-\sin\phi")
        viz.expr(g, r"v_{22}=\cos\phi")
        viz.expr(g, r"w_{11}=av_{11}+bv_{12}")
        viz.expr(g, r"w_{12}=cv_{11}+dv_{12}")
        viz.expr(g, r"w_{21}=av_{21}+bv_{22}")
        viz.expr(g, r"w_{22}=cv_{21}+dv_{22}")
        for idx, color in (("1", "#2d70b3"), ("2", "#388c46")):
            sub = "₁₂"[int(idx) - 1]
            viz.curve(g, f"tv_{{{idx}1}}", f"tv_{{{idx}2}}", color=color, lineWidth=2)
            viz.expr(g, rf"\left(v_{{{idx}1}},v_{{{idx}2}}\right)", color=color, pointSize=6,
                     showLabel=True, label=f"v{sub}")
            viz.curve(g, f"tw_{{{idx}1}}", f"tw_{{{idx}2}}", color=color, lineWidth=4)
            viz.expr(g, rf"\left(w_{{{idx}1}},w_{{{idx}2}}\right)", color=color, pointSize=9,
                     showLabel=True, label=f"Aₖv{sub}")

        # --- the answer: angle where A v_1 . A v_2 = 0 (same formula as a Jacobi rotation)
        viz.expr(g, r"p_{d}=w_{11}w_{21}+w_{12}w_{22}")
        viz.expr(g, r"\phi_{s}=\frac{1}{2}\arctan\left(2\left(ab+cd\right),a^{2}+c^{2}-b^{2}-d^{2}\right)")
        viz.expr(g, r"s_{1}=\sqrt{\left(a\cos\phi_{s}+b\sin\phi_{s}\right)^{2}"
                    r"+\left(c\cos\phi_{s}+d\sin\phi_{s}\right)^{2}}")
        viz.expr(g, r"s_{2}=\sqrt{\left(b\cos\phi_{s}-a\sin\phi_{s}\right)^{2}"
                    r"+\left(d\cos\phi_{s}-c\sin\phi_{s}\right)^{2}}")
        viz.curve(g, r"t\left(a\cos\phi_{s}+b\sin\phi_{s}\right)",
                  r"t\left(c\cos\phi_{s}+d\sin\phi_{s}\right)", -1, 1,
                  color="#c74440", lineStyle="DOTTED")
        viz.curve(g, r"t\left(b\cos\phi_{s}-a\sin\phi_{s}\right)",
                  r"t\left(d\cos\phi_{s}-c\sin\phi_{s}\right)", -1, 1,
                  color="#c74440", lineStyle="DOTTED")
        viz.expr(g, r"m_{1}=\operatorname{round}\left(p_{d},4\right)")
        viz.expr(g, r"m_{2}=\operatorname{round}\left(\phi_{s},3\right)")
        viz.expr(g, r"m_{3}=\operatorname{round}\left(s_{1},3\right)")
        viz.expr(g, r"m_{4}=\operatorname{round}\left(s_{2},3\right)")
        viz.expr(g, r"\left(-2.4,3.55\right)", color="#000000", pointSize=0, showLabel=True,
                 labelOrientation="right", label="Aₖv₁ · Aₖv₂ = ${m_1}  (0 at φ = ${m_2}: singular vectors)")
        viz.expr(g, r"\left(-2.4,3.2\right)", color="#000000", pointSize=0, showLabel=True,
                 labelOrientation="right", label="block singular values: σ₁ = ${m_3}, σ₂ = ${m_4}")

        # --- a draggable vector x and where the original / compressed block send it
        viz.expr(g, r"\left(q_{1},q_{2}\right)", color="#6042a6", showLabel=True, label="x")
        viz.expr(g, r"\left(a_{0}q_{1}+b_{0}q_{2},c_{0}q_{1}+d_{0}q_{2}\right)", color="#555555",
                 showLabel=True, label="Ax")
        viz.expr(g, r"\left(aq_{1}+bq_{2},cq_{1}+dq_{2}\right)", color="#6042a6",
                 showLabel=True, label="Aₖx")

        # --- the surrounding pixels: original patch (top) and A_k patch (bottom)
        cell = 0.4
        half = viz.desmos_number(cell / 2)
        left, top_o, top_k = 2.9, 2.9, -0.5
        for y_name, color_name in (("Y_{o}", "C_{o}"), ("Y_{k}", "C_{k}")):
            corners = [(f"X-{half}", f"{y_name}-{half}"), (f"X+{half}", f"{y_name}-{half}"),
                       (f"X+{half}", f"{y_name}+{half}"), (f"X-{half}", f"{y_name}+{half}")]
            points = ",".join(rf"\left({x},{y}\right)" for x, y in corners)
            viz.expr(g, rf"\operatorname{{polygon}}\left({points}\right)",
                     colorLatex=color_name, fillOpacity=1, lineWidth=1, lineOpacity=0.3)
        for top in (top_o, top_k):
            right, bottom = left + 2 * cell, top - 2 * cell
            corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
            points = ",".join(rf"\left({viz.desmos_number(x)},{viz.desmos_number(y)}\right)" for x, y in corners)
            viz.expr(g, rf"\operatorname{{polygon}}\left({points}\right)", color="#f0b400", fillOpacity=0, lineWidth=4)
        last_r, last_c = row + size - 1, col + size - 1
        viz.expr(g, rf"\left({left},{viz.desmos_number(top_o + 0.2)}\right)", color="#000000", pointSize=0,
                 showLabel=True, labelOrientation="right",
                 label=f"Original A ({name}), rows {row}-{last_r}, cols {col}-{last_c}")
        viz.expr(g, rf"\left({left},{viz.desmos_number(top_k + 0.2)}\right)", color="#c74440", pointSize=0,
                 showLabel=True, labelOrientation="right",
                 label="Aₖ with k = ${k}  (yellow: block [a b; c d])")

        # --- data (goes into a collapsed folder at the bottom of the expression list)
        originals = [block_original[key] / 255.0 for key in "abcd"]
        viz.data(g, f"B_{{0}}={viz.desmos_list(originals, 5)}")
        for key in "abcd":
            viz.data(g, f"L_{{{key}}}={viz.desmos_list(block_layers[key], 6)}")
        xs = [left + (j + 0.5) * cell for i in range(size) for j in range(size)]
        ys = [-(i + 0.5) * cell for i in range(size) for j in range(size)]
        viz.data(g, f"X={viz.desmos_list(xs, 3)}")
        viz.data(g, rf"Y_{{o}}={viz.desmos_list([top_o + y for y in ys], 3)}")
        viz.data(g, rf"Y_{{k}}={viz.desmos_list([top_k + y for y in ys], 3)}")
        viz.data(g, f"H_{{0}}={viz.desmos_list(patch_original, 3)}")
        viz.stepped_list(g, "H", patches)
        rgb = ["25", "25", "25"]
        for list_name, color_name in (("H_{0}", "C_{o}"), ("H", "C_{k}")):
            rgb[channel] = f"25+216.75{list_name}"
            viz.expr(g, rf"{color_name}=\operatorname{{rgb}}\left({rgb[0]},{rgb[1]},{rgb[2]}\right)")

        viz.set_bounds(g, -2.5, 6.4, -3.0, 3.8, square=True)
        return g


register(SVDCompressor())
