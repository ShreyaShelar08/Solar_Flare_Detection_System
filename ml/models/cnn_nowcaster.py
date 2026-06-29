"""
CNN-based binary nowcaster for real-time flare detection on HEL1OS
hard X-ray data.

Architecture
------------
Four-channel input (``counts_clean``, ``rolling_mean``, ``rolling_std``,
``derivative``) → three conv-pool blocks → one final conv → global
average pooling → dropout → two dense layers → Sigmoid.

The model outputs a single probability indicating whether a flare is
currently in progress within the observation window.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNNowcaster(nn.Module):
    """1-D CNN binary classifier for flare nowcasting.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (default 4).
    conv_channels : list[int]
        Output channels for the four conv layers.
    kernel_size : int
        Kernel width for all conv layers.
    pool_size : int
        Max-pool kernel for the first three conv blocks.
    dropout : float
        Dropout rate before the first dense layer.
    fc_hidden : int
        Width of the hidden dense layer.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: list[int] | None = None,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.5,
        fc_hidden: int = 64,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128, 128]

        # ── Conv block 1: Conv -> BN -> ReLU -> MaxPool ─────────────────
        self.conv1 = nn.Conv1d(
            in_channels, conv_channels[0],
            kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.pool1 = nn.MaxPool1d(pool_size)

        # ── Conv block 2 ────────────────────────────────────────────────
        self.conv2 = nn.Conv1d(
            conv_channels[0], conv_channels[1],
            kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.pool2 = nn.MaxPool1d(pool_size)

        # ── Conv block 3 ────────────────────────────────────────────────
        self.conv3 = nn.Conv1d(
            conv_channels[1], conv_channels[2],
            kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.bn3 = nn.BatchNorm1d(conv_channels[2])
        self.pool3 = nn.MaxPool1d(pool_size)

        # ── Conv block 4 (no pooling) ──────────────────────────────────
        self.conv4 = nn.Conv1d(
            conv_channels[2], conv_channels[3],
            kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.bn4 = nn.BatchNorm1d(conv_channels[3])

        # ── Classifier head ────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_channels[3], fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 4, 128)`` — batch of 4-channel, 128-step windows.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)`` — flare probability per sample.
        """
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))

        # Global Average Pooling: (B, C, T') -> (B, C)
        x = x.mean(dim=-1)

        # Dense head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
