"""Wine scanner package and existing macOS OpenMP compatibility settings.

The production decider uses CatBoost. Importing this package no longer eagerly imports
historical LightGBM. CPU thread limits on macOS remain configurable.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if sys.platform == "darwin":
    import torch

    torch.set_num_threads(int(os.environ.get("WINE_TORCH_THREADS", "1")))
