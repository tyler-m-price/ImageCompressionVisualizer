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
3. **Compression Visualizer**: one color channel shown as a matrix of numbers (original, compressed, or the difference), a summary of the math, and a button that opens an interactive Desmos graph in your browser (internet connection required). The graph has a `k` slider for the number of singular values kept:
   - **2x2 block as a transformation**: the 2x2 block at the matrix view's top-left cell acts on the plane, mapping the unit circle to an ellipse. Rotate the orthonormal pair (v1, v2) with `φ` until A·v1 ⟂ A·v2 to find the block's singular vectors (the same rotation a Jacobi SVD performs). Drag `k` to watch compression reshape the ellipse, and watch the 6x6 patch of pixels next to it change.

Photos can be scaled down to a *working resolution* (512/1024/2048 px on the long side, or full size) before compressing. The first compression of an image computes the decomposition. After that, changing the amount is instant.

## How the SVD is computed

`compressors/svd.py` follows the textbook steps for each color channel (a grayscale image has one matrix):

1. Treat the channel as an m × n matrix A of pixel values.
2. Compute A = U Σ Vᵀ with `np.linalg.svd`. The singular values σ₁ ≥ σ₂ ≥ … ≥ 0 usually drop off fast.
3. Choose k. The *compression amount* is the share of storage saved, so k is the largest rank whose factors fit in (100 − amount)% of the raw pixel count (0% keeps every singular value).
4. Truncate to U_k (m × k), Σ_k (k × k) and V_kᵀ (k × n).
5. Rebuild A_k = U_k Σ_k V_kᵀ, the best rank-k approximation of A, then clip it to 0–255 and round.
6. A_k needs k(m + n + 1) numbers instead of m·n. For example, a 1000 × 1000 image with k = 50 needs about 100,000 numbers instead of 1,000,000.

## Adding a compression technique

Create `compressors/<name>.py` with a `Compressor` subclass (see `compressors/base.py` and `compressors/svd.py`), call `register(...)` at the bottom, and import it in `compressors/__init__.py`. It then shows up in the technique dropdown, and its `summary()` text and `desmos_views()` buttons appear in the visualizer automatically.

## Packaging

```bash
pip install pyinstaller
pyinstaller --windowed --name "Image Compression Visualizer" main.py
```
