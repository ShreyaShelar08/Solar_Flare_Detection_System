"""Data loading, preprocessing, and augmentation utilities."""

from .loader import (
    load_helios,
    load_solexs,
    create_windows,
    HEL1OSDataset,
    SoLEXSDataset,
)
from .preprocessor import (
    RuleBasedCleaner,
    compute_derivative,
    MinMaxScaler,
    normalize_per_band,
)
from .augmentation import (
    inject_synthetic_flare,
    jitter_noise,
    time_warp,
    oversample_minority,
)

__all__ = [
    "load_helios",
    "load_solexs",
    "create_windows",
    "HEL1OSDataset",
    "SoLEXSDataset",
    "RuleBasedCleaner",
    "compute_derivative",
    "MinMaxScaler",
    "normalize_per_band",
    "inject_synthetic_flare",
    "jitter_noise",
    "time_warp",
    "oversample_minority",
]
