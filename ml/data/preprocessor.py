"""
Rule-based data cleaning, feature engineering, and normalisation utilities.

All transformations are *deterministic* and invertible where possible so that
evaluation metrics can be reported on the original scale.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Rule-based cleaning
# ═══════════════════════════════════════════════════════════════════════════


class RuleBasedCleaner:
    """Collection of stateless cleaning transformations.

    Each method takes a ``DataFrame`` (or ``Series``) and returns the
    cleaned version *in-place* for efficiency.  The caller can chain calls:

    .. code-block:: python

        cleaner = RuleBasedCleaner()
        df = cleaner.fill_null_zscore(df)
        df = cleaner.interpolate_missing_counts(df)
    """

    # ------------------------------------------------------------------ #
    # Z-score imputation
    # ------------------------------------------------------------------ #

    @staticmethod
    def fill_null_zscore(df: pd.DataFrame, column: str = "zscore") -> pd.DataFrame:
        """Replace empty / NaN z-score values with 0.0.

        In the HEL1OS data the z-score is left blank whenever
        ``rolling_std == 0`` (constant signal window).  Setting these to
        0.0 is semantically correct — no deviation from the mean.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing a z-score column.
        column : str
            Name of the z-score column (default ``"zscore"``).

        Returns
        -------
        pd.DataFrame
            The same dataframe with NaN z-scores filled.
        """
        if column in df.columns:
            # Coerce non-numeric empty strings first
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        return df

    # ------------------------------------------------------------------ #
    # Missing count interpolation
    # ------------------------------------------------------------------ #

    @staticmethod
    def interpolate_missing_counts(
        df: pd.DataFrame,
        column: str = "counts_clean",
    ) -> pd.DataFrame:
        """Linearly interpolate missing (NaN) count values.

        SoLEXS raw ``counts`` may be missing; the pre-cleaned column
        ``counts_clean`` already contains some fractional imputed values
        but may still have gaps.  We fill them with linear interpolation
        and forward/backward-fill any remaining edge NaNs.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        column : str
            Name of the counts column to interpolate.

        Returns
        -------
        pd.DataFrame
        """
        if column in df.columns:
            df[column] = (
                df[column]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
        return df

    # ------------------------------------------------------------------ #
    # Outlier clipping
    # ------------------------------------------------------------------ #

    @staticmethod
    def clip_outliers(
        df: pd.DataFrame,
        column: str,
        n_sigma: float = 5.0,
    ) -> pd.DataFrame:
        """Clip values beyond ±n_sigma standard deviations from the mean.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        column : str
            Column name whose values will be clipped.
        n_sigma : float
            Number of standard deviations for the clip boundary.

        Returns
        -------
        pd.DataFrame
        """
        if column not in df.columns:
            return df
        mean = df[column].mean()
        std = df[column].std()
        if std == 0 or np.isnan(std):
            return df
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
        df[column] = df[column].clip(lower=lower, upper=upper)
        return df

    # ------------------------------------------------------------------ #
    # Median filter (scipy-free)
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_median_filter(
        series: pd.Series,
        kernel_size: int = 5,
    ) -> pd.Series:
        """Apply a rolling-median filter without scipy.

        Parameters
        ----------
        series : pd.Series
            1-D signal to smooth.
        kernel_size : int
            Must be odd.  The width of the rolling window.

        Returns
        -------
        pd.Series
            Smoothed series with the same index.  Edge values are
            forward/backward-filled so the length is preserved.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # enforce odd
        smoothed = series.rolling(window=kernel_size, center=True, min_periods=1).median()
        return smoothed


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════


def compute_derivative(series: pd.Series) -> pd.Series:
    """First-order finite difference (rate-of-change).

    ``d[i] = series[i] - series[i-1]`` with the first value set to 0.

    Parameters
    ----------
    series : pd.Series
        1-D time series (must already be sorted by time).

    Returns
    -------
    pd.Series
        Same length as input; first element is 0.0.
    """
    derivative = series.diff().fillna(0.0)
    return derivative


# ═══════════════════════════════════════════════════════════════════════════
# Min-Max Scaler (fit on train, apply on val/test)
# ═══════════════════════════════════════════════════════════════════════════


class MinMaxScaler:
    """Channel-wise min-max normalisation to [0, 1].

    Designed to be *fit* on the training partition only and then applied
    identically to validation / test data to avoid data leakage.

    Supports both 1-D arrays (single feature) and 2-D arrays
    (samples × features).
    """

    def __init__(self) -> None:
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._range: Optional[np.ndarray] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------ #

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        """Compute per-feature min and max from *training* data.

        Parameters
        ----------
        data : np.ndarray
            Array of shape ``(n_samples,)`` or ``(n_samples, n_features)``.

        Returns
        -------
        MinMaxScaler
            ``self`` for method chaining.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._min = data.min(axis=0)
        self._max = data.max(axis=0)
        self._range = self._max - self._min
        # Avoid division by zero for constant features
        self._range[self._range == 0] = 1.0
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale *data* using previously fitted statistics.

        Parameters
        ----------
        data : np.ndarray
            Same shape convention as ``fit``.

        Returns
        -------
        np.ndarray
            Scaled data in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("MinMaxScaler has not been fitted yet.")
        data = np.asarray(data, dtype=np.float64)
        squeeze = data.ndim == 1
        if squeeze:
            data = data.reshape(-1, 1)
        scaled = (data - self._min) / self._range
        return scaled.squeeze() if squeeze else scaled

    # ------------------------------------------------------------------ #

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the scaling to recover original values.

        Parameters
        ----------
        data : np.ndarray
            Scaled array.

        Returns
        -------
        np.ndarray
        """
        if not self._fitted:
            raise RuntimeError("MinMaxScaler has not been fitted yet.")
        data = np.asarray(data, dtype=np.float64)
        squeeze = data.ndim == 1
        if squeeze:
            data = data.reshape(-1, 1)
        original = data * self._range + self._min
        return original.squeeze() if squeeze else original

    # ------------------------------------------------------------------ #

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Convenience: fit + transform in one call."""
        return self.fit(data).transform(data)


# ═══════════════════════════════════════════════════════════════════════════
# Per-band normalisation
# ═══════════════════════════════════════════════════════════════════════════


def normalize_per_band(
    df: pd.DataFrame,
    band_column: str = "energy_band",
    value_column: str = "counts_per_sec_clean",
) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """Normalise a value column independently per energy band.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``band_column`` and ``value_column``.
    band_column : str
        Column that identifies the energy band / detector.
    value_column : str
        Numeric column to normalise.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, MinMaxScaler]]
        * The dataframe with a new column ``{value_column}_norm``.
        * A mapping ``{band_name: fitted_scaler}`` for inverse transforms.
    """
    df = df.copy()
    norm_col = f"{value_column}_norm"
    df[norm_col] = np.nan
    scalers: Dict[str, MinMaxScaler] = {}

    for band, grp in df.groupby(band_column):
        scaler = MinMaxScaler()
        values = grp[value_column].values
        scaled = scaler.fit_transform(values)
        df.loc[grp.index, norm_col] = scaled
        scalers[str(band)] = scaler

    return df, scalers
