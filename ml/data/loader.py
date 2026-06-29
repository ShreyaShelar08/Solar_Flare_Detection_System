"""
Data loading, sliding-window creation, and PyTorch ``Dataset`` classes
for HEL1OS (hard X-ray) and SoLEXS (soft X-ray) instruments.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.config import (
    FLARE_CLASS_NAMES,
    FLARE_LABEL_HORIZON,
    FLARE_THRESHOLDS,
    HELIOS_CSV,
    SOLEXS_CSV,
    WINDOW_SIZE_FORECASTER,
)


# ═══════════════════════════════════════════════════════════════════════════
# CSV loaders
# ═══════════════════════════════════════════════════════════════════════════


def load_helios(band_filter: Optional[str] = None) -> pd.DataFrame:
    """Load the HEL1OS cleaned CSV and optionally filter by energy band.

    Parameters
    ----------
    band_filter : str, optional
        If provided, keep only rows whose ``energy_band`` matches this
        value (e.g. ``'CZT1_LC_BAND_20.00KEV_TO_40.00KEV'``).

    Returns
    -------
    pd.DataFrame
        Sorted by ``datetime_utc`` ascending.  The ``zscore`` column is
        coerced to float (empty strings → NaN) but **not** filled here —
        use :class:`RuleBasedCleaner` for that.
    """
    df = pd.read_csv(str(HELIOS_CSV), low_memory=False)

    # Coerce zscore to numeric (blank when rolling_std == 0)
    if "zscore" in df.columns:
        df["zscore"] = pd.to_numeric(df["zscore"], errors="coerce")

    if band_filter is not None and "energy_band" in df.columns:
        df = df[df["energy_band"] == band_filter].reset_index(drop=True)

    # Ensure chronological order
    if "datetime_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
        df = df.sort_values("datetime_utc").reset_index(drop=True)

    # Coerce label column to boolean
    if "is_outlier_flare" in df.columns:
        df["is_outlier_flare"] = (
            df["is_outlier_flare"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(["TRUE", "1", "1.0"])
        )

    return df


def load_solexs() -> pd.DataFrame:
    """Load the SoLEXS combined cleaned CSV.

    Returns
    -------
    pd.DataFrame
        Sorted by ``datetime_utc`` ascending.  Missing ``counts`` are
        *not* interpolated here — use :class:`RuleBasedCleaner`.
    """
    df = pd.read_csv(str(SOLEXS_CSV), low_memory=False)

    if "datetime_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
        df = df.sort_values("datetime_utc").reset_index(drop=True)

    if "zscore" in df.columns:
        df["zscore"] = pd.to_numeric(df["zscore"], errors="coerce")

    if "is_outlier_flare" in df.columns:
        df["is_outlier_flare"] = (
            df["is_outlier_flare"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(["TRUE", "1", "1.0"])
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Sliding-window utility
# ═══════════════════════════════════════════════════════════════════════════


def create_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Generate sliding windows over a 1-D or 2-D array.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(T,)`` for univariate or ``(T, C)`` for multivariate.
    window_size : int
        Number of time-steps per window.
    stride : int
        Step between consecutive window start positions.

    Returns
    -------
    np.ndarray
        Shape ``(N, window_size)`` or ``(N, window_size, C)``.
    """
    data = np.asarray(data)
    T = data.shape[0]
    if T < window_size:
        raise ValueError(
            f"Data length ({T}) < window_size ({window_size})."
        )
    starts = range(0, T - window_size + 1, stride)
    windows = np.array([data[s : s + window_size] for s in starts])
    return windows


# ═══════════════════════════════════════════════════════════════════════════
# HEL1OS Dataset (Nowcaster + Autoencoder)
# ═══════════════════════════════════════════════════════════════════════════


class HEL1OSDataset(Dataset):
    """Sliding-window dataset over HEL1OS time-series.

    Each sample is a ``(window, label)`` pair where:

    * ``window`` — float32 tensor of shape ``(window_size,)``
    * ``label``  — float32 scalar, **1.0** if *any* of the last
      ``FLARE_LABEL_HORIZON`` time-steps has ``is_outlier_flare == True``,
      else **0.0**.

    Parameters
    ----------
    values : np.ndarray
        1-D array of cleaned signal values (e.g. ``counts_per_sec_clean``).
    labels : np.ndarray
        Boolean array of same length (``is_outlier_flare``).
    window_size : int
        Sliding window length.
    stride : int
        Step size between windows.
    """

    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        window_size: int = 64,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride

        # Pre-compute windows and labels
        self.windows = create_windows(values.astype(np.float32), window_size, stride)
        label_windows = create_windows(labels.astype(np.float32), window_size, stride)

        # Positive if ANY of the trailing ``FLARE_LABEL_HORIZON`` steps is True
        horizon = min(FLARE_LABEL_HORIZON, window_size)
        self.labels = (label_windows[:, -horizon:].sum(axis=1) > 0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ═══════════════════════════════════════════════════════════════════════════
# SoLEXS Dataset (Forecaster)
# ═══════════════════════════════════════════════════════════════════════════


def _classify_peak(peak_counts: float) -> int:
    """Map a peak count value to a flare class index (0-3)."""
    for cls_idx, cls_name in enumerate(FLARE_CLASS_NAMES):
        lo, hi = FLARE_THRESHOLDS[cls_name]
        if lo <= peak_counts < hi:
            return cls_idx
    return len(FLARE_CLASS_NAMES) - 1  # X class if above all thresholds


class SoLEXSDataset(Dataset):
    """Multi-task sliding-window dataset for SoLEXS soft X-ray.

    Each sample yields ``(window, class_label, time_to_peak)``:

    * ``window``       — float32 tensor ``(window_size,)``
    * ``class_label``  — int64 scalar in ``{0, 1, 2, 3}`` for B/C/M/X,
      determined by the **peak counts_clean** in a forward-looking
      window of the same size.
    * ``time_to_peak`` — float32 scalar, hours until that peak from the
      window's last step.

    Parameters
    ----------
    values : np.ndarray
        1-D cleaned counts signal.
    timestamps : np.ndarray
        Array of ``datetime64`` for each time-step (same length as *values*).
    window_size : int
        Input window length.
    forward_window : int
        How many steps ahead to search for the peak.
    stride : int
        Step between consecutive windows.
    """

    def __init__(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        window_size: int = 512,
        forward_window: int = 512,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride

        values = values.astype(np.float32)
        T = len(values)
        max_start = T - window_size - forward_window

        if max_start < 0:
            raise ValueError(
                f"Data length ({T}) too short for window_size={window_size} "
                f"+ forward_window={forward_window}."
            )

        starts = list(range(0, max_start + 1, stride))
        self.windows = np.empty((len(starts), window_size), dtype=np.float32)
        self.class_labels = np.empty(len(starts), dtype=np.int64)
        self.time_to_peak = np.empty(len(starts), dtype=np.float32)

        for i, s in enumerate(starts):
            self.windows[i] = values[s : s + window_size]

            fwd_start = s + window_size
            fwd_end = fwd_start + forward_window
            fwd_slice = values[fwd_start:fwd_end]

            peak_rel_idx = int(np.argmax(fwd_slice))
            peak_counts = float(fwd_slice[peak_rel_idx])
            self.class_labels[i] = _classify_peak(peak_counts)

            # Time-to-peak in hours
            t_window_end = np.datetime64(timestamps[s + window_size - 1])
            t_peak = np.datetime64(timestamps[fwd_start + peak_rel_idx])
            delta_ns = (t_peak - t_window_end).astype("timedelta64[ns]").astype(np.float64)
            self.time_to_peak[i] = float(max(delta_ns / 3.6e12, 0.0))  # ns → hours

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx])
        cls = torch.tensor(self.class_labels[idx], dtype=torch.long)
        ttp = torch.tensor(self.time_to_peak[idx], dtype=torch.float32)
        return x, cls, ttp
