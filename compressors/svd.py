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
SCANLINE_SAMPLES = 128  # columns sampled for the Desmos scanline view


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
            ("Desmos: singular value spectrum", self._desmos_spectrum),
            ("Desmos: rebuild a row layer by layer", self._desmos_scanline),
        ]

    # ------------------------------------------------------------ Desmos

    def _desmos_spectrum(self, analysis, result, channel, row):
        s = [float(x) for x in analysis["s"][channel] if x > 1e-9]
        count = len(s)
        total = frobenius_norm_sq(s)
        cumulative, running = [], 0.0
        for x in s:
            running += x * x
            cumulative.append(running / total)
        k0 = min(result.extra["k"], count)
        logs = [np.log10(x) for x in s]

        g = viz.new_graph(f"svd_spectrum_{viz.CHANNEL_NAMES[channel].lower()}")
        viz.expr(g, f"S={viz.desmos_list(s, 3)}")
        viz.expr(g, f"Q={viz.desmos_list(cumulative, 6)}")
        viz.expr(g, rf"I=\left[1,...,{count}\right]")
        viz.slider(g, "k", k0, 1, count)
        viz.expr(g, r"p=\operatorname{round}\left(100Q\left[k\right],2\right)")
        # All singular values (log scale) in gray, the kept ones in red.
        viz.expr(g, r"\left(I,\log\left(S\right)\right)", color="#888888", pointSize=4)
        viz.expr(g, r"\left(I\left[1,...,k\right],\log\left(S\left[1,...,k\right]\right)\right)",
                 color="#c74440", pointSize=6)
        viz.expr(g, "x=k", color="#c74440", lineStyle="DASHED")
        viz.expr(g, r"\left(k,\log\left(S\left[k\right]\right)\right)", color="#c74440",
                 showLabel=True, label="keep k=${k} layers: ${p}% of the energy")
        viz.set_bounds(g, -0.05 * count, 1.05 * count, min(logs) - 0.5, max(logs) + 0.5)
        return g

    def _desmos_scanline(self, analysis, result, channel, row):
        u, s, v = analysis["u"][channel], analysis["s"][channel], analysis["v"][channel]
        image = analysis["image"]
        height, width, _ = image.shape
        row = max(0, min(height - 1, int(row)))

        step = max(1, width // SCANLINE_SAMPLES)
        cols = list(range(0, width, step))
        n = len(cols)
        k_max = max(1, min(len(s), viz.DESMOS_LIST_LIMIT // n))

        # C holds every layer's contribution to this row, flattened:
        # C[(i-1) n + j] = s_i * u_i[row] * v_i[col_j]
        layers = []
        for i in range(k_max):
            coeff = s[i] * u[row, i]
            for col in cols:
                layers.append(coeff * v[col, i])

        original = [image[row, col, channel] for col in cols]
        app_row = [result.image_rgb[row, col, channel] for col in cols]

        g = viz.new_graph(f"svd_scanline_{viz.CHANNEL_NAMES[channel].lower()}_row{row}")
        viz.expr(g, f"n={n}")
        viz.expr(g, f"J={viz.desmos_list(cols, 0)}")
        viz.expr(g, f"O={viz.desmos_list(original, 1)}")
        viz.expr(g, f"P={viz.desmos_list(app_row, 1)}")
        viz.expr(g, f"C={viz.desmos_list(layers, 3)}")
        viz.slider(g, "k", min(result.extra["k"], k_max), 1, k_max)
        # R = row of A_k (sum of the first k layers); T = the k-th layer alone.
        viz.expr(g, r"R=\left[\sum_{i=1}^{k}C\left[\left(i-1\right)n+j\right]"
                    r"\operatorname{for}j=\left[1,...,n\right]\right]")
        viz.expr(g, r"T=\left[C\left[\left(k-1\right)n+j\right]\operatorname{for}j=\left[1,...,n\right]\right]")
        viz.expr(g, r"\left(J,O\right)", color="#000000", lines=True, points=False, lineWidth=2)
        viz.expr(g, r"\left(J,R\right)", color="#c74440", lines=True, points=False, lineWidth=3)
        viz.expr(g, r"\left(J,T\right)", color="#2d70b3", lines=True, points=False, lineStyle="DASHED")
        viz.expr(g, r"\left(J,P\right)", color="#388c46", lines=True, points=False, hidden=True)
        viz.set_bounds(g, -0.02 * width, 1.02 * width, -100, 300)
        return g


register(SVDCompressor())
