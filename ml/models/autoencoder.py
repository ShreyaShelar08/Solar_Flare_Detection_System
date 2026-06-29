"""
Denoising Autoencoder for unsupervised anomaly detection on solar X-ray
time series.

Architecture
------------
* **Encoder**: three Conv1D blocks (stride-2 down-sampling) with
  Batch Normalisation and ReLU.
* **Decoder**: mirror of the encoder using ConvTranspose1D, ending with
  a Sigmoid activation to keep outputs in [0, 1].

The model is trained to reconstruct *clean* windows from *noisy* inputs.
At inference time, the per-sample MSE reconstruction error serves as an
anomaly score — flares produce systematically higher error.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    """1-D convolutional denoising autoencoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for univariate signal).
    encoder_channels : list[int]
        Output channel sizes for each encoder conv layer.
    kernel_sizes : list[int]
        Kernel widths for each encoder conv layer.
    strides : list[int]
        Strides for each encoder conv layer (also used in decoder).
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: List[int] | None = None,
        kernel_sizes: List[int] | None = None,
        strides: List[int] | None = None,
    ) -> None:
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [16, 32, 64]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if strides is None:
            strides = [2, 2, 2]

        assert len(encoder_channels) == len(kernel_sizes) == len(strides), (
            "encoder_channels, kernel_sizes, and strides must have equal length."
        )

        # ── Encoder ─────────────────────────────────────────────────────
        enc_layers: List[nn.Module] = []
        ch_in = in_channels
        for ch_out, ks, s in zip(encoder_channels, kernel_sizes, strides):
            pad = (ks - 1) // 2
            enc_layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=ks, stride=s, padding=pad),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder (mirror) ───────────────────────────────────────────
        dec_layers: List[nn.Module] = []
        rev_channels = list(reversed(encoder_channels))
        rev_ks = list(reversed(kernel_sizes))
        rev_s = list(reversed(strides))

        for i, (ch_out_dec, ks, s) in enumerate(zip(
            rev_channels[1:] + [in_channels],
            rev_ks,
            rev_s,
        )):
            ch_in_dec = rev_channels[i]
            pad = (ks - 1) // 2
            out_pad = s - 1  # ensures exact size recovery with stride > 1
            dec_layers.append(
                nn.ConvTranspose1d(
                    ch_in_dec,
                    ch_out_dec,
                    kernel_size=ks,
                    stride=s,
                    padding=pad,
                    output_padding=out_pad,
                )
            )
            if i < len(rev_channels) - 1:
                dec_layers.extend([
                    nn.BatchNorm1d(ch_out_dec),
                    nn.ReLU(inplace=True),
                ])
            else:
                dec_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*dec_layers)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, T)`` — batch of single-channel windows.

        Returns
        -------
        torch.Tensor
            Reconstructed signal of the **same shape** as *x*.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        # Trim or pad to match input length (rounding from strided convs)
        T_in = x.size(-1)
        T_out = reconstructed.size(-1)
        if T_out > T_in:
            reconstructed = reconstructed[..., :T_in]
        elif T_out < T_in:
            reconstructed = F.pad(reconstructed, (0, T_in - T_out))

        return reconstructed

    # ------------------------------------------------------------------ #
    # Anomaly score
    # ------------------------------------------------------------------ #

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (anomaly score).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, T)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` — one scalar score per sample.
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            mse = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return mse

    # ------------------------------------------------------------------ #
    # Latent representation (for regularisation)
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent (bottleneck) representation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, T)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, C_latent, T_latent)``.
        """
        return self.encoder(x)
