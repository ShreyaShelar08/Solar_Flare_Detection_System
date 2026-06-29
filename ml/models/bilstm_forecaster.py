"""
Bidirectional LSTM forecaster with learnable attention for multi-task
solar-flare prediction on SoLEXS soft X-ray data.

Architecture
------------
* **BiLSTM backbone**: 2-layer BiLSTM (hidden 128) with inter-layer dropout.
* **Attention**: a learnable query vector attends over all BiLSTM time
  steps to produce a fixed-length context vector (256-dim).
* **Classification head**: Dense(256 → 64 → 4) predicting B/C/M/X class.
* **Regression head**: Dense(256 → 64 → 1) + ReLU predicting
  time-to-peak in hours (non-negative).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Attention
# ═══════════════════════════════════════════════════════════════════════════


class Attention(nn.Module):
    """Learnable-query additive attention over a sequence.

    Given a sequence of hidden states ``H`` of shape ``(B, T, D)``,
    the module computes:

        score  = tanh(H @ W + b) @ v          → (B, T)
        alpha  = softmax(score, dim=-1)        → (B, T)
        context = alpha.unsqueeze(-1) * H      → (B, D)  (weighted sum)

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of each hidden vector.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted context vector.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Context vector of shape ``(B, D)``.
        """
        # (B, T, D)
        energy = torch.tanh(self.attn_proj(hidden_states))
        # (B, T)
        scores = torch.matmul(energy, self.query)
        alpha = F.softmax(scores, dim=-1)  # (B, T)
        # Weighted sum: (B, T, 1) * (B, T, D) -> sum -> (B, D)
        context = (alpha.unsqueeze(-1) * hidden_states).sum(dim=1)
        return context


# ═══════════════════════════════════════════════════════════════════════════
# BiLSTM Forecaster
# ═══════════════════════════════════════════════════════════════════════════


class BiLSTMForecaster(nn.Module):
    """Multi-task BiLSTM + Attention forecaster.

    Parameters
    ----------
    in_features : int
        Number of input features per time-step (default 3:
        counts_clean, rolling_mean, rolling_std).
    hidden_size : int
        LSTM hidden size per direction (total = 2 × hidden_size).
    num_layers : int
        Number of stacked LSTM layers.
    lstm_dropout : float
        Dropout between LSTM layers.
    attention_dropout : float
        Dropout after the attention context vector.
    num_classes : int
        Number of flare classes (default 4: B/C/M/X).
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        lstm_dropout: float = 0.3,
        attention_dropout: float = 0.4,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        context_dim = hidden_size * 2  # bidirectional → 2× hidden
        self.attention = Attention(context_dim)
        self.drop = nn.Dropout(attention_dropout)

        # ── Classification head: B / C / M / X ─────────────────────────
        self.cls_fc1 = nn.Linear(context_dim, 64)
        self.cls_fc2 = nn.Linear(64, num_classes)

        # ── Regression head: time-to-peak (hours) ──────────────────────
        self.reg_fc1 = nn.Linear(context_dim, 64)
        self.reg_fc2 = nn.Linear(64, 1)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, F)`` where *F* = ``in_features``.

        Returns
        -------
        class_logits : torch.Tensor
            Shape ``(B, num_classes)`` — raw logits (apply softmax
            externally for probabilities).
        time_pred : torch.Tensor
            Shape ``(B, 1)`` — predicted time-to-peak in hours (≥ 0).
        """
        # LSTM output: (B, T, 2*hidden)
        lstm_out, _ = self.lstm(x)

        # Attention context: (B, 2*hidden)
        context = self.attention(lstm_out)
        context = self.drop(context)

        # Classification head
        cls = F.relu(self.cls_fc1(context))
        class_logits = self.cls_fc2(cls)  # raw logits, no softmax

        # Regression head
        reg = F.relu(self.reg_fc1(context))
        time_pred = F.relu(self.reg_fc2(reg))  # ReLU ensures non-negative

        return class_logits, time_pred
