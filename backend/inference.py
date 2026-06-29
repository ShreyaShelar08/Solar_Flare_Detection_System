"""
Inference pipeline for the Solar Flare Early Warning System.

Loads the three PyTorch models (autoencoder, CNN nowcaster, BiLSTM
forecaster) from disk and exposes a clean Python API for running
inference.  When model weights are missing, each method transparently
falls back to a heuristic / simulated prediction so the dashboard
remains fully functional during development and training.

Usage
-----
>>> pipeline = InferencePipeline("d:/Project/Solar_Flare_system/ml/saved_models")
>>> prob, critical = pipeline.nowcast(hxr_features)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Allow imports from the top-level project so we can reach ml.models.*
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.models.autoencoder import DenoisingAutoencoder        # noqa: E402
from ml.models.cnn_nowcaster import CNNNowcaster               # noqa: E402
from ml.models.bilstm_forecaster import BiLSTMForecaster       # noqa: E402

logger = logging.getLogger("solarflare.inference")

# Alert threshold for nowcast probability
_NOWCAST_ALERT_THRESHOLD: float = 0.65

# Flare-class labels (same order as BiLSTMForecaster output)
_FLARE_CLASSES: list[str] = ["B", "C", "M", "X"]


class InferencePipeline:
    """Unified inference pipeline wrapping all three ML models.

    Parameters
    ----------
    models_dir : str | Path
        Directory that contains the saved ``.pth`` weight files.
        Expected filenames:
        - ``autoencoder.pth``
        - ``cnn_nowcaster.pth``
        - ``bilstm_forecaster.pth``
    """

    # Mapping: friendly name → (filename, model class, constructor kwargs)
    _MODEL_REGISTRY: dict[str, tuple[str, type, dict[str, Any]]] = {
        "autoencoder": (
            "autoencoder.pth",
            DenoisingAutoencoder,
            {"window_length": 256, "in_channels": 1},
        ),
        "cnn_nowcaster": (
            "cnn_nowcaster.pth",
            CNNNowcaster,
            {"in_channels": 4, "window_length": 128},
        ),
        "bilstm_forecaster": (
            "bilstm_forecaster.pth",
            BiLSTMForecaster,
            {"in_channels": 3, "hidden_dim": 128, "num_classes": 4},
        ),
    }

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Inference device: %s", self.device)

        # Holds loaded model instances (or None when weights are absent)
        self._models: dict[str, torch.nn.Module | None] = {}

        for name, (filename, cls, kwargs) in self._MODEL_REGISTRY.items():
            self._models[name] = self._try_load(name, filename, cls, kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load(
        self,
        name: str,
        filename: str,
        cls: type,
        kwargs: dict[str, Any],
    ) -> torch.nn.Module | None:
        """Attempt to instantiate a model and load its weights.

        Returns ``None`` (instead of crashing) when the file is missing
        or incompatible, enabling graceful fallback.
        """
        weight_path = self.models_dir / filename
        if not weight_path.exists():
            logger.warning(
                "Weights not found for '%s' at %s — using simulated fallback.",
                name,
                weight_path,
            )
            return None

        try:
            model = cls(**kwargs)
            state_dict = torch.load(
                weight_path, map_location=self.device, weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            logger.info("Loaded '%s' from %s", name, weight_path)
            return model
        except Exception as exc:
            logger.error(
                "Failed to load '%s' from %s: %s — using simulated fallback.",
                name,
                weight_path,
                exc,
            )
            return None

    @staticmethod
    def _to_tensor(
        data: list | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert list / ndarray → torch.Tensor."""
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        return torch.tensor(data, dtype=dtype)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def models_loaded(self) -> dict[str, bool]:
        """Return a mapping of model name → ``True`` if weights were loaded."""
        return {name: (m is not None) for name, m in self._models.items()}

    # ------------------------------------------------------------------
    # 1. Denoising / Cleaning
    # ------------------------------------------------------------------

    def clean(
        self,
        raw_window: list[float],
        instrument: str = "helios",
    ) -> tuple[list[float], list[float]]:
        """Clean a raw telemetry window using the denoising autoencoder.

        Parameters
        ----------
        raw_window : list[float]
            1-D raw signal (arbitrary length; will be zero-padded / truncated
            to the model's ``window_length`` internally).
        instrument : str
            ``'helios'`` or ``'solexs'`` — reserved for future per-instrument
            scaling but currently handled identically.

        Returns
        -------
        cleaned : list[float]
            Denoised signal (same length as input).
        anomaly_scores : list[float]
            Per-window reconstruction error (single value wrapped in a list
            when the model is available; per-point absolute residual otherwise).
        """
        model = self._models.get("autoencoder")
        original_len = len(raw_window)

        if model is not None:
            # Prepare tensor: (1, 1, window_length)
            window_length = model.window_length  # type: ignore[attr-defined]
            arr = np.array(raw_window, dtype=np.float32)

            # Pad or truncate to match expected window
            if len(arr) < window_length:
                arr = np.pad(arr, (0, window_length - len(arr)), mode="edge")
            elif len(arr) > window_length:
                arr = arr[:window_length]

            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            x = x.to(self.device)

            with torch.no_grad():
                x_hat = model(x)
                mse = ((x - x_hat) ** 2).mean(dim=(1, 2)).cpu().numpy()

            cleaned = x_hat.squeeze().cpu().numpy()[:original_len].tolist()
            anomaly_scores = [round(float(mse[0]), 6)]
            return cleaned, anomaly_scores

        # ------ Fallback: return input with a simple smoothing ------
        arr = np.array(raw_window, dtype=np.float32)
        # Simple moving-average denoise (window=5)
        kernel_size = min(5, len(arr))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(arr, kernel, mode="same")
        residuals = np.abs(arr - smoothed)
        return smoothed.tolist(), residuals.tolist()

    # ------------------------------------------------------------------
    # 2. Nowcasting
    # ------------------------------------------------------------------

    def nowcast(
        self,
        hxr_features: list[list[float]] | np.ndarray,
    ) -> tuple[float, bool, float]:
        """Predict the imminent-flare probability from HXR features.

        Parameters
        ----------
        hxr_features : array-like, shape ``[128, 4]``
            Four-channel feature matrix: counts_clean, rolling_mean,
            rolling_std, derivative.

        Returns
        -------
        probability : float
            Flare probability ∈ [0, 1].
        is_critical : bool
            ``True`` when probability ≥ alert threshold.
        confidence : float
            Model confidence ∈ [0, 1].
        """
        model = self._models.get("cnn_nowcaster")

        if model is not None:
            # CNN expects (batch, channels, time) → transpose from (T, C)
            arr = self._to_tensor(hxr_features)        # (128, 4)
            x = arr.T.unsqueeze(0).to(self.device)     # (1, 4, 128)

            with torch.no_grad():
                prob_t = model(x)                      # (1, 1)

            prob = float(prob_t.squeeze().cpu())
            confidence = abs(prob - 0.5) * 2.0
            return (
                round(prob, 4),
                prob >= _NOWCAST_ALERT_THRESHOLD,
                round(confidence, 4),
            )

        # ------ Fallback: heuristic spike detector ------
        arr = np.array(hxr_features, dtype=np.float32)
        counts = arr[:, 0] if arr.ndim == 2 else arr
        mean_val = float(np.mean(counts))
        std_val = float(np.std(counts)) + 1e-8
        max_val = float(np.max(counts))

        # Simple z-score–based spike detection
        spike_score = (max_val - mean_val) / std_val
        prob = float(np.clip(1.0 / (1.0 + np.exp(-0.5 * (spike_score - 3.0))), 0, 1))
        # Add small random jitter for realism
        prob = float(np.clip(prob + np.random.normal(0, 0.02), 0, 1))
        confidence = abs(prob - 0.5) * 2.0
        return (
            round(prob, 4),
            prob >= _NOWCAST_ALERT_THRESHOLD,
            round(confidence, 4),
        )

    # ------------------------------------------------------------------
    # 3. Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        sxr_features: list[list[float]] | np.ndarray,
    ) -> tuple[str, dict[str, float], float, float]:
        """Forecast flare class and time-to-peak from SXR features.

        Parameters
        ----------
        sxr_features : array-like, shape ``[512, 3]``
            Three-channel feature matrix: counts_clean, rolling_mean,
            rolling_std.

        Returns
        -------
        predicted_class : str
            GOES flare class ('B', 'C', 'M', or 'X').
        class_probabilities : dict[str, float]
            Softmax probabilities for each class.
        time_to_peak_hours : float
            Estimated hours until flux peak.
        confidence : float
            Confidence in the predicted class ∈ [0, 1].
        """
        model = self._models.get("bilstm_forecaster")

        if model is not None:
            arr = self._to_tensor(sxr_features)          # (512, 3)
            x = arr.unsqueeze(0).to(self.device)          # (1, 512, 3)

            with torch.no_grad():
                logits, time_pred = model(x)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                time_hours = float(time_pred.squeeze().cpu())

            cls_idx = int(np.argmax(probs))
            predicted_class = _FLARE_CLASSES[cls_idx]
            class_probs = {
                c: round(float(probs[i]), 4) for i, c in enumerate(_FLARE_CLASSES)
            }
            confidence = float(probs[cls_idx])
            return predicted_class, class_probs, round(time_hours, 2), round(confidence, 4)

        # ------ Fallback: heuristic based on SXR mean level ------
        arr = np.array(sxr_features, dtype=np.float32)
        counts = arr[:, 0] if arr.ndim == 2 else arr
        mean_level = float(np.mean(np.abs(counts)))

        # Map mean level to rough flare class via thresholds
        if mean_level < 1e-7:
            base_probs = [0.70, 0.20, 0.08, 0.02]
        elif mean_level < 1e-5:
            base_probs = [0.30, 0.45, 0.18, 0.07]
        elif mean_level < 1e-3:
            base_probs = [0.05, 0.20, 0.50, 0.25]
        else:
            base_probs = [0.02, 0.08, 0.30, 0.60]

        # Add jitter
        noise = np.random.dirichlet(np.array(base_probs) * 50 + 1).tolist()
        probs_arr = np.array(noise, dtype=np.float32)
        probs_arr /= probs_arr.sum()

        cls_idx = int(np.argmax(probs_arr))
        predicted_class = _FLARE_CLASSES[cls_idx]
        class_probs = {
            c: round(float(probs_arr[i]), 4)
            for i, c in enumerate(_FLARE_CLASSES)
        }
        confidence = float(probs_arr[cls_idx])

        # Simulated time-to-peak (inverse relationship with intensity)
        time_hours = max(0.1, round(float(np.random.exponential(2.0 + 3.0 / (mean_level + 1))), 2))

        return predicted_class, class_probs, time_hours, round(confidence, 4)
