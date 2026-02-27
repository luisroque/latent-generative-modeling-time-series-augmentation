"""
Root-level pytest configuration. Forces CPU device before PyTorch
initialises and sets matplotlib to a non-interactive backend.
"""
import os

os.environ["LGTA_DEVICE"] = "cpu"

import torch  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
