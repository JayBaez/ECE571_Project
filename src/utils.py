"""
utils.py

Small, general-purpose helpers used across the whole project:
- set_seed(): make experiments reproducible.
- get_device(): find out whether PyTorch can use the GPU.
- ensure_dir(): create an output folder if it doesn't exist yet.
- log_step(): print simple, readable progress messages.

Nothing in this file is specific to Problems 1-5 - it's plumbing that
every problem will reuse.
"""

import os
import random

import numpy as np

# The project spec (course_context/TEACHER_EXPECTATIONS.md) requires
# regression results averaged over at least 3 random seeds. These are
# just example values to use consistently across experiments - not
# special numbers with any statistical meaning.
DEFAULT_SEEDS = [42, 123, 2026]


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
      installed, plus PyTorch's deterministic-algorithm settings.

    IMPORTANT - what this does NOT do: scikit-learn has no single
    global seed. Every scikit-learn model/function that accepts a
    `random_state` argument (e.g. `RandomForestClassifier(random_state=seed)`,
    `train_test_split(random_state=seed)`) needs that seed passed to
    it explicitly, every time. Calling set_seed() does not make
    scikit-learn deterministic on its own - pass `random_state=seed`
    everywhere scikit-learn accepts it.

    PERFORMANCE NOTE: forcing PyTorch into fully deterministic mode
    (below) can make GPU training somewhat slower, because some fast
    CUDA operations don't have a deterministic version and PyTorch has
    to fall back to a slower one. This is a correctness-over-speed
    trade-off we're accepting deliberately for reproducibility - see
    course_context/TEACHER_EXPECTATIONS.md's reproducibility
    requirement.

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
            # Ask cuDNN to use only deterministic algorithms, even
            # though this can be slower than its default "pick
            # whatever's fastest" behavior.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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


def log_step(step_number: int, total_steps: int, message: str) -> None:
    """
    Print one readable progress line, e.g. "[2/6] Cleaning data...".

    This is deliberately just a `print()` wrapper - no logging
    framework, no log files, no configuration. For a project this
    size, plain printed status lines are easier to read and easier to
    explain than a general-purpose logging setup.

    Parameters
    ----------
    step_number : int
        Which step this is (1-indexed).
    total_steps : int
        How many steps the whole process has.
    message : str
        What's happening, e.g. "Cleaning data...".
    """
    print(f"[{step_number}/{total_steps}] {message}")



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
