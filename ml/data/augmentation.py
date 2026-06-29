"""
Data augmentation utilities for class-imbalanced solar flare datasets.

All functions operate on NumPy arrays and are designed to be applied
*after* windowing but *before* batching.
"""

from __future__ import annotations

import copy
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic flare injection
# ═══════════════════════════════════════════════════════════════════════════


def inject_synthetic_flare(
    window: np.ndarray,
    amplitude: float = 1.0,
    rise_time: int = 10,
    decay_time: int = 30,
) -> np.ndarray:
    """Inject a synthetic flare profile into a time-series window.

    The flare is modelled as a two-sided Gaussian: a fast rise followed
    by a slower exponential-like decay, placed at a random position
    inside the window.

    Parameters
    ----------
    window : np.ndarray
        1-D array of length ``T`` (a single time-series window).
    amplitude : float
        Peak height of the injected flare (in normalised units).
    rise_time : int
        Standard deviation (in time-steps) of the rising edge.
    decay_time : int
        Standard deviation (in time-steps) of the decaying tail.

    Returns
    -------
    np.ndarray
        A copy of ``window`` with the synthetic flare added.
    """
    window = window.copy().astype(np.float64)
    T = len(window)
    # Place the peak somewhere in the middle 60 % of the window
    peak_idx = np.random.randint(int(T * 0.2), int(T * 0.8))
    t = np.arange(T, dtype=np.float64)

    rise = np.exp(-0.5 * ((t[:peak_idx] - peak_idx) / max(rise_time, 1)) ** 2)
    decay = np.exp(-0.5 * ((t[peak_idx:] - peak_idx) / max(decay_time, 1)) ** 2)
    flare_profile = np.concatenate([rise, decay])

    window += amplitude * flare_profile
    return window


# ═══════════════════════════════════════════════════════════════════════════
# Additive Gaussian noise
# ═══════════════════════════════════════════════════════════════════════════


def jitter_noise(
    window: np.ndarray,
    sigma: float = 0.05,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to a window.

    Parameters
    ----------
    window : np.ndarray
        1-D or 2-D array (channels × time or just time).
    sigma : float
        Standard deviation of the added noise.

    Returns
    -------
    np.ndarray
    """
    noise = np.random.normal(0.0, sigma, size=window.shape)
    return (window + noise).astype(window.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Temporal warping
# ═══════════════════════════════════════════════════════════════════════════


def time_warp(
    window: np.ndarray,
    sigma: float = 0.2,
) -> np.ndarray:
    """Apply random temporal stretching / compression via cubic interpolation.

    The time axis is warped by a smooth random curve so that some
    regions appear stretched and others compressed, while preserving the
    total window length.

    Parameters
    ----------
    window : np.ndarray
        1-D array of length ``T``.
    sigma : float
        Controls the magnitude of the warp.  Higher = more distortion.

    Returns
    -------
    np.ndarray
        Warped window of the same length.
    """
    T = len(window)
    # Generate a smooth warp path by cumulating random perturbations
    warp_steps = np.random.normal(loc=1.0, scale=sigma, size=T)
    warp_steps = np.maximum(warp_steps, 0.1)  # avoid negative / zero steps
    time_steps = np.cumsum(warp_steps)
    # Normalise to span [0, T-1]
    time_steps = (time_steps - time_steps[0]) / (time_steps[-1] - time_steps[0]) * (T - 1)

    # Linear interpolation onto original indices
    orig_indices = np.arange(T, dtype=np.float64)
    warped = np.interp(orig_indices, time_steps, window.astype(np.float64))
    return warped.astype(window.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Minority oversampling
# ═══════════════════════════════════════════════════════════════════════════


class _OversampledDataset(Dataset):
    """Internal wrapper that appends duplicated minority samples."""

    def __init__(
        self,
        base_dataset: Dataset,
        extra_indices: List[int],
    ) -> None:
        self._base = base_dataset
        self._extra = extra_indices
        self._total_len = len(base_dataset) + len(extra_indices)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> Tuple:
        if idx < len(self._base):  # type: ignore[arg-type]
            return self._base[idx]
        extra_idx = self._extra[idx - len(self._base)]  # type: ignore[arg-type]
        return self._base[extra_idx]


def oversample_minority(
    dataset: Dataset,
    target_ratio: float = 0.3,
) -> Dataset:
    """Duplicate minority-class windows until they reach ``target_ratio``.

    The function inspects each sample's label (assumed to be the second
    element of the tuple returned by ``__getitem__``) and treats any
    non-zero / ``True`` value as the minority (positive) class.

    Parameters
    ----------
    dataset : Dataset
        A PyTorch ``Dataset`` that returns ``(features, label, ...)``.
    target_ratio : float
        Desired fraction of positive samples in the oversampled set.
        For example 0.3 means 30 % positives.

    Returns
    -------
    Dataset
        A new dataset with extra copies of minority samples appended.
    """
    positive_indices: List[int] = []
    total = len(dataset)  # type: ignore[arg-type]

    for i in range(total):
        sample = dataset[i]
        label = sample[1]
        # Handle tensor, numpy, or scalar labels
        if isinstance(label, torch.Tensor):
            is_positive = bool(label.any())
        elif isinstance(label, np.ndarray):
            is_positive = bool(label.any())
        else:
            is_positive = bool(label)

        if is_positive:
            positive_indices.append(i)

    n_pos = len(positive_indices)
    n_neg = total - n_pos

    if n_pos == 0 or n_pos / total >= target_ratio:
        return dataset  # nothing to do

    # Compute how many copies of the positive set we need
    # target_ratio = (n_pos * k) / (n_neg + n_pos * k)
    #  => k = target_ratio * n_neg / (n_pos * (1 - target_ratio))
    k = int(np.ceil(target_ratio * n_neg / (n_pos * (1.0 - target_ratio))))
    k = max(k, 1)

    extra_indices: List[int] = []
    for _ in range(k - 1):  # we already have 1× of the positives
        extra_indices.extend(positive_indices)

    return _OversampledDataset(dataset, extra_indices)
