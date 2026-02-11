"""Nowcast model: Transformer encoder over embedding + weather sequences.

Outputs:
  - stress_index
  - growth_index
  - yield_shock_mean
  - yield_shock_log_sigma
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 200) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NowcastModel(nn.Module):
    """Transformer-based nowcast model.

    Input: concatenation of e_img, e_loc, delta_e, weather features per timestep.
    Output: 4 scalar predictions.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        weather_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        # Input: e_img + e_loc + delta_e + weather = 3*emb_dim + weather_dim
        input_dim = 3 * embedding_dim + weather_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.stress_head = nn.Linear(hidden_dim, 1)
        self.growth_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.yield_mean_head = nn.Linear(hidden_dim, 1)
        self.yield_log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        e_img: torch.Tensor,
        e_loc: torch.Tensor,
        delta_e: torch.Tensor,
        weather: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            e_img: (B, T, emb_dim)
            e_loc: (B, T, emb_dim)
            delta_e: (B, T, emb_dim)
            weather: (B, T, weather_dim)

        Returns:
            Dictionary with stress_index, growth_index, yield_shock_mean,
            yield_shock_log_sigma — each (B,).
        """
        # Concatenate inputs
        x = torch.cat([e_img, e_loc, delta_e, weather], dim=-1)  # (B, T, input_dim)
        x = self.input_proj(x)  # (B, T, hidden_dim)
        x = self.pos_enc(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, T, hidden_dim)

        # Use last timestep representation
        x_last = x[:, -1, :]  # (B, hidden_dim)
        x_last = self.dropout(x_last)

        stress = self.stress_head(x_last).squeeze(-1)
        growth = self.growth_head(x_last).squeeze(-1)
        yield_mean = self.yield_mean_head(x_last).squeeze(-1)
        yield_log_sigma = self.yield_log_sigma_head(x_last).squeeze(-1)

        return {
            "stress_index": stress,
            "growth_index": growth,
            "yield_shock_mean": yield_mean,
            "yield_shock_log_sigma": yield_log_sigma,
        }


def build_model(
    embedding_dim: int = 512,
    weather_dim: int = 8,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.15,
) -> NowcastModel:
    """Factory to create a NowcastModel with given hyperparameters."""
    return NowcastModel(
        embedding_dim=embedding_dim,
        weather_dim=weather_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
