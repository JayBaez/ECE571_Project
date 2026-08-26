"""
Tests for src/utils.py: reproducibility and GPU device detection.
"""

import random

import numpy as np

from src import utils


def test_set_seed_makes_python_random_reproducible():
    utils.set_seed(42)
    a = [random.random() for _ in range(5)]
    utils.set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_random_reproducible():
    utils.set_seed(7)
    a = np.random.rand(5)
    utils.set_seed(7)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_seed_different_seeds_usually_differ():
    utils.set_seed(1)
    a = np.random.rand(5)
    utils.set_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_get_device_returns_valid_device_string():
    device = utils.get_device()
    assert device in ("cuda", "cpu")


def test_ensure_dir_creates_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = utils.ensure_dir(str(target))
    assert target.exists()
    assert result == str(target)


def test_default_seeds_has_at_least_three_seeds():
    # The project spec requires at least 3 seeds for regression
    # reporting (course_context/TEACHER_EXPECTATIONS.md).
    assert len(utils.DEFAULT_SEEDS) >= 3
