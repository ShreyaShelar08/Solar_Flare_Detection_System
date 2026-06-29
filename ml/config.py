"""
Central configuration for the Aditya-L1 Solar Flare Early Warning System.

All hyper-parameters, file paths, device selection, and domain-specific
constants live here so every downstream module imports a single source of
truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Device auto-detection: CUDA > MPS > CPU
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = _detect_device()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(r"d:\Project\Solar_Flare_system")
ML_ROOT: Path = PROJECT_ROOT / "ml"

HELIOS_CSV: Path = PROJECT_ROOT / "HEL1OS_cleaned_Data.csv"
SOLEXS_CSV: Path = PROJECT_ROOT / "SoLEXS_combined_cleaned.csv"

MODEL_SAVE_DIR: Path = ML_ROOT / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

AUTOENCODER_CKPT: Path = MODEL_SAVE_DIR / "autoencoder.pth"
NOWCASTER_CKPT: Path = MODEL_SAVE_DIR / "nowcaster.pth"
FORECASTER_CKPT: Path = MODEL_SAVE_DIR / "forecaster.pth"

# ---------------------------------------------------------------------------
# Sliding-window sizes
# ---------------------------------------------------------------------------

WINDOW_SIZE_AUTOENCODER: int = 64
WINDOW_SIZE_NOWCASTER: int = 128
WINDOW_SIZE_FORECASTER: int = 512

# ---------------------------------------------------------------------------
# Training hyper-parameters (shared defaults)
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
EPOCHS: int = 50

# ---------------------------------------------------------------------------
# HEL1OS-specific
# ---------------------------------------------------------------------------

PRIMARY_HELIOS_BAND: str = "CZT1_LC_BAND_20.00KEV_TO_40.00KEV"

# Number of trailing time-steps inspected for a positive label in the
# nowcaster dataset — if *any* of the last ``FLARE_LABEL_HORIZON`` steps
# has ``is_outlier_flare == True``, the window is labelled positive.
FLARE_LABEL_HORIZON: int = 15

# ---------------------------------------------------------------------------
# SoLEXS-specific — flare classification thresholds (counts_clean)
# ---------------------------------------------------------------------------
# B  : counts < 10
# C  : 10 ≤ counts < 25
# M  : 25 ≤ counts < 40
# X  : counts ≥ 40

FLARE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "B": (0.0, 10.0),
    "C": (10.0, 25.0),
    "M": (25.0, 40.0),
    "X": (40.0, float("inf")),
}

FLARE_CLASS_NAMES: List[str] = ["B", "C", "M", "X"]
NUM_FLARE_CLASSES: int = len(FLARE_CLASS_NAMES)

# ---------------------------------------------------------------------------
# Model-specific hyper-parameters
# ---------------------------------------------------------------------------


@dataclass
class AutoencoderConfig:
    """Hyper-parameters for the Denoising Autoencoder."""

    window_size: int = WINDOW_SIZE_AUTOENCODER
    in_channels: int = 1
    latent_channels: int = 64
    encoder_channels: List[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    noise_std: float = 0.1
    l1_lambda: float = 1e-4
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS


@dataclass
class NowcasterConfig:
    """Hyper-parameters for the CNN Nowcaster."""

    window_size: int = WINDOW_SIZE_NOWCASTER
    in_channels: int = 4  # counts_clean, rolling_mean, rolling_std, derivative
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128])
    kernel_size: int = 3
    pool_size: int = 2
    dropout: float = 0.5
    fc_hidden: int = 64
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    oversample_factor: int = 10
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS


@dataclass
class ForecasterConfig:
    """Hyper-parameters for the BiLSTM Forecaster."""

    window_size: int = WINDOW_SIZE_FORECASTER
    in_features: int = 3  # counts_clean, rolling_mean, rolling_std
    hidden_size: int = 128
    num_layers: int = 2
    lstm_dropout: float = 0.3
    attention_dropout: float = 0.4
    num_classes: int = NUM_FLARE_CLASSES
    time_loss_weight: float = 0.5
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
