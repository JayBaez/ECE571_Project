"""
check_setup.py

Run this after setting up your Python environment to confirm the
project foundation works. This does NOT train any model or run any
experiment - it only checks that imports, packages, the dataset file,
and the small utility functions behave as expected.

Usage (from the project root):
    python scripts/check_setup.py
"""

import os
import sys

# Allow running this script directly from the project root without
# installing the project as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def check(label: str, fn) -> bool:
    """Run one check, print PASS/FAIL, and return whether it passed."""
    try:
        result = fn()
        print(f"[PASS] {label}" + (f" -> {result}" if result is not None else ""))
        return True
    except Exception as e:
        print(f"[FAIL] {label} -> {type(e).__name__}: {e}")
        return False


def main():
    results = []

    # --- Package imports ---
    def _import_core_packages():
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import openpyxl  # noqa: F401
        import pandas  # noqa: F401
        import seaborn  # noqa: F401
        import sklearn  # noqa: F401
        import yaml  # noqa: F401
        return "pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl, PyYAML all import OK"

    results.append(check("Core package imports", _import_core_packages))

    def _import_torch():
        import torch
        return f"torch {torch.__version__}"

    results.append(check("PyTorch import (optional)", _import_torch))

    # --- src/ module imports ---
    def _import_src_modules():
        from src import (  # noqa: F401
            data_loader,
            evaluation,
            experiment_runner,
            feature_engineering,
            preprocessing,
            splitting,
            utils,
            visualization,
        )
        return "all 8 src/ modules import OK"

    results.append(check("src/ module imports", _import_src_modules))

    # --- Data loader can find and open the workbook ---
    def _list_sheets():
        from src import data_loader
        sheets = data_loader.list_sheets()
        assert len(sheets) > 0, "workbook opened but no sheets found"
        return f"{len(sheets)} sheets found"

    results.append(check("Excel workbook can be opened", _list_sheets))

    # --- A single sheet can be loaded ---
    def _load_one_sheet():
        from src import data_loader
        sheets = data_loader.list_sheets()
        df = data_loader.load_sheet(sheets[0])
        assert len(df) > 0, "sheet loaded but has zero rows"
        return f"loaded '{sheets[0]}' -> {df.shape[0]} rows, {df.shape[1]} columns"

    results.append(check("A single sheet can be loaded", _load_one_sheet))

    # --- Seed setting works ---
    def _set_seed():
        import random
        import numpy as np
        from src import utils
        utils.set_seed(42)
        a = (random.random(), np.random.rand())
        utils.set_seed(42)
        b = (random.random(), np.random.rand())
        assert a == b, "same seed produced different random values"
        return "same seed reproduces identical random values"

    results.append(check("set_seed() reproducibility", _set_seed))

    # --- GPU detection ---
    def _get_device():
        from src import utils
        device = utils.get_device()
        assert device in ("cuda", "cpu")
        return device

    results.append(check("get_device() runs without error", _get_device))

    # --- Summary ---
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed.")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
