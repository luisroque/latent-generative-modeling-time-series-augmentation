"""
Pytest configuration for lgta tests. Root-level conftest.py handles
PyTorch device override and matplotlib backend. Provides helpers for
skipping tests when external datasets are unavailable.
"""
import os
import unittest

import pytest


def _dataset_file(name: str, ext: str = "csv") -> str:
    return os.path.join("assets", "data", "original_datasets", f"{name}.{ext}")


def _dataset_available(name: str) -> bool:
    """Check if the raw dataset file exists locally (CSV, XLSX, or ZIP)."""
    for ext in ("csv", "xlsx", "zip"):
        if os.path.isfile(_dataset_file(name, ext)):
            return True
    return False


def skip_unless_dataset(name: str) -> None:
    """Call inside setUp to skip when a dataset isn't locally available."""
    if not _dataset_available(name):
        raise unittest.SkipTest(
            f"Dataset '{name}' not available locally; "
            "download it first or check network connectivity."
        )
