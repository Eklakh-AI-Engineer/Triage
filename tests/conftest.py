"""Suite-wide environment guards.

Kaggle notebook kernels export MPLBACKEND=module://matplotlib_inline.
backend_inline into their environment; the py312 side-env does not have
matplotlib_inline installed, and importing matplotlib with that value
raises at import time (matplotlib validates the backend by importing the
named module). Force the headless Agg backend before any test can import
matplotlib - the same guard src/evaluation/plots.py applies for script-mode
benchmark runs.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
