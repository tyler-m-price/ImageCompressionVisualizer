from compressors.base import COMPRESSORS, CompressionResult, Compressor, get, list_names, register
from compressors import svd  # noqa: F401  (registers SVD)

__all__ = ["COMPRESSORS", "CompressionResult", "Compressor", "get", "list_names", "register"]
