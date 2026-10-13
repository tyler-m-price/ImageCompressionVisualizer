"""Registry and interface for image compression techniques.

To add a new technique:
  1. Create compressors/<name>.py with a Compressor subclass.
  2. Implement analyze(), compress(), summary() and (optionally) desmos_views().
  3. Call register(YourCompressor()) at the bottom of the module.
  4. Import the module in compressors/__init__.py.
The UI picks it up automatically (method dropdown, summary text, Desmos buttons).
"""


class CompressionResult:
    """Output of Compressor.compress().

    image_rgb: float array (H, W, 3) with values in [0, 255].
    extra: technique-specific data for summary() / desmos_views().
    """

    def __init__(self, image_rgb, extra=None):
        self.image_rgb = image_rgb
        self.extra = extra or {}


class Compressor:
    name = "base"
    # One-line explanation of what the 0-100% compression amount means.
    amount_help = ""

    def analyze(self, image_rgb, progress=None):
        """Expensive, amount-independent work (e.g. a matrix decomposition).

        Called once per image and cached by the UI, so changing the amount
        afterwards is instant. progress(fraction, message) may be called with
        fraction in [0, 1].
        """
        return image_rgb

    def compress(self, analysis, amount_percent):
        """Return a CompressionResult for the given amount (0-100)."""
        raise NotImplementedError

    def summary(self, result, channel):
        """Plain-ASCII text explaining the math of this result."""
        return ""

    def desmos_views(self):
        """List of (button label, builder) pairs.

        builder(analysis, result, channel, row, col) -> PyDesmos Graph, where
        (row, col) is the top-left cell shown in the matrix view.
        """
        return []


COMPRESSORS = {}


def register(compressor):
    COMPRESSORS[compressor.name] = compressor
    return compressor


def list_names():
    return list(COMPRESSORS.keys())


def get(name):
    if name not in COMPRESSORS:
        raise KeyError(f"Unknown compression method: {name}")
    return COMPRESSORS[name]
