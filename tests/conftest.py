"""
Pytest configuration for MDMARD test suite.
"""

import os

# Force headless matplotlib backend
os.environ["MPLBACKEND"] = "Agg"
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
