__version__ = "0.5.69"

# Workaround: PyTorch >=2.8 on macOS ARM segfaults on the first
# normalization kernel call if it happens inside a deep import chain.
# A single eager call here triggers the lazy init safely.
import torch as _torch
_torch.nn.functional.layer_norm(_torch.randn(2, 4), [4])
del _torch

from lgta import preprocessing
from lgta import transformations
from lgta import visualization
from lgta import evaluation
from lgta import model
from lgta import postprocessing
from lgta import e2e
