"""
utils.py

Small, general-purpose helpers used across the whole project:
- set_seed(): make experiments reproducible.
- get_device(): find out whether PyTorch can use the GPU.
- ensure_dir(): create an output folder if it doesn't exist yet.

Nothing in this file is specific to Problems 1-5 - it's plumbing that
every problem will reuse.
"""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    Set the random seed for every source of randomness we use.

    Why this matters (ML concept): many parts of ML are randomized on
    purpose (e.g. shuffling data, initializing neural network weights,
    choosing a random subset of labels for semi-supervised learning).
    If we don't fix the seed, re-running the same experiment gives
    slightly different results every time, which makes it impossible
    to reproduce or fairly compare methods. Course context:
    reproducibility is a REQUIRED grading item (see
    course_context/TEACHER_EXPECTATIONS.md).

    This function controls:
    - Python's built-in `random` module.
    - NumPy's random number generator.
    - PyTorch's CPU and GPU random number generators, if PyTorch is
      installed.

    It does NOT make every algorithm perfectly deterministic on the GPU
    (some GPU operations are inherently non-deterministic for speed
    reasons), but it removes the main sources of run-to-run variation.

    Parameters
    ----------
    seed : int
        The seed value to use everywhere, e.g. 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # PyTorch isn't installed yet - that's fine, we just skip it.
        pass


def get_device() -> str:
    """
    Decide whether PyTorch should use the GPU ("cuda") or the CPU
    ("cpu") for training.

    ML concept: deep learning models (MLP, RNN, LSTM/GRU - see
    course_context/ML_METHOD_MAP.md) train much faster on a GPU because
    GPUs are built for the kind of large matrix multiplications that
    neural networks do. This function checks once whether a compatible
    NVIDIA GPU and CUDA driver are available, so the rest of the code
    can just ask "which device should I use?" instead of repeating this
    check everywhere.

    Returns
    -------
    str
        "cuda" if a CUDA-capable GPU is available to PyTorch,
        otherwise "cpu". The project must also run correctly on "cpu"
        (e.g. on a laptop with no NVIDIA GPU) - GPU is a speed-up, not
        a requirement.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        # PyTorch isn't installed - fall back to CPU-only tools.
        return "cpu"


def ensure_dir(path: str) -> str:
    """
    Create a directory (and any missing parent directories) if it
    doesn't already exist. Safe to call even if the directory already
    exists.

    This is used before saving files (figures, results, models) so
    that code doesn't crash just because a folder hasn't been created
    yet.

    Parameters
    ----------
    path : str
        Directory path to create.

    Returns
    -------
    str
        The same path, for convenient chaining, e.g.
        `save_path = os.path.join(ensure_dir("figures/problem1"), "cm.png")`.
    """
    os.makedirs(path, exist_ok=True)
    return path
