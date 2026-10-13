# ImageCompressionVisualizer

Visualization program for image compression techniques. Built for Linear Algebra research project Fall 2026.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

1. **Config Menu**: *Select Image...* opens your OS file picker. Then choose a technique and a compression amount, and press *Compress*. *Save compressed image...* writes the result to disk.
2. **Image**: the original (working copy) and the compressed result.
3. **Compression Visualizer**: one color channel shown as a matrix of numbers (original, compressed, or the difference), a summary of the math, and buttons that open interactive Desmos graphs in your browser (internet connection required). Each graph has a `k` slider for the number of singular values kept:
   - **2x2 block as a transformation**: the 2x2 block at the matrix view's top-left cell acts on the plane, mapping the unit circle to an ellipse. Rotate the orthonormal pair (v1, v2) with `φ` until A·v1 ⟂ A·v2 to find the block's singular vectors (the same rotation the Jacobi SVD performs). Drag `k` to watch compression reshape the ellipse, and watch the 6x6 patch of pixels next to it change.
   - **Singular value spectrum**: every singular value (log scale), with the kept ones highlighted, and the share of energy kept.
   - **Rebuild a row layer by layer**: one row of pixels as the sum of its first `k` rank-1 layers.

Large photos are scaled down to a *working resolution* (256/384/512 px on the long side) before compressing, because the hand-written SVD costs O(n^3). The first compression of an image computes the decomposition. After that, changing the amount is instant.

## How the SVD is computed

`linalg_manual.py` implements a one-sided (Hestenes) Jacobi SVD by hand. Numpy is used only to store arrays and do elementwise arithmetic; `numpy.linalg` is never called. The algorithm repeatedly applies Givens rotations to pairs of columns until every pair is orthogonal. Rotations on disjoint column pairs are scheduled into "rounds" (tournament circle method) so each round runs at once.

## Adding a compression technique

Create `compressors/<name>.py` with a `Compressor` subclass (see `compressors/base.py` and `compressors/svd.py`), call `register(...)` at the bottom, and import it in `compressors/__init__.py`. It then shows up in the technique dropdown, and its `summary()` text and `desmos_views()` buttons appear in the visualizer automatically.

## Packaging

```bash
pip install pyinstaller
pyinstaller --windowed --name "Image Compression Visualizer" main.py
```
