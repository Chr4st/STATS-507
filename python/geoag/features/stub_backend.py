"""Stub embedding backend for demo mode.

Produces deterministic embeddings based on image hash and location coordinates.
Same interface as SatCLIP but no real model weights required.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from geoag.features.embedding_provider import EmbeddingProvider


class StubBackend(EmbeddingProvider):
    """Deterministic pseudo-random embeddings for demo/testing."""

    def __init__(self, dim: int = 512, seed: int = 42) -> None:
        self._dim = dim
        self._seed = seed
        # Fixed random projection matrices (deterministic)
        rng = np.random.RandomState(seed)
        self._img_proj = torch.from_numpy(
            rng.randn(256, dim).astype(np.float32)
        )
        self._loc_proj = torch.from_numpy(
            rng.randn(64, dim).astype(np.float32)
        )
        self._img_bias = torch.from_numpy(
            rng.randn(dim).astype(np.float32) * 0.1
        )
        self._loc_bias = torch.from_numpy(
            rng.randn(dim).astype(np.float32) * 0.1
        )

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_image(self, tile: np.ndarray) -> torch.Tensor:
        """Deterministic embedding from image content.

        Approach: compute channel histograms (256 bins) → project to dim.
        """
        flat = tile.reshape(-1) if tile.ndim == 3 else tile.flatten()

        # Use MD5 of raw bytes for deterministic seed
        h = hashlib.md5(flat.tobytes()[:4096]).hexdigest()
        rng = np.random.RandomState(int(h[:8], 16))

        # Histogram features (256-dim, normalized)
        hist = np.histogram(flat, bins=256, range=(0, 255))[0].astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)

        # Add small noise for variation
        hist += rng.randn(256).astype(np.float32) * 0.01

        feat = torch.from_numpy(hist)
        emb = feat @ self._img_proj + self._img_bias
        # L2 normalize
        emb = emb / (emb.norm() + 1e-8)
        return emb

    def encode_location(self, lat: float, lon: float) -> torch.Tensor:
        """Deterministic location embedding using Fourier features."""
        # Normalize to [-1, 1]
        lat_n = lat / 90.0
        lon_n = lon / 180.0

        # Fourier features at multiple frequencies
        freqs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        features: list[float] = []
        for f in freqs:
            features.extend(
                [
                    np.sin(f * lat_n * np.pi),
                    np.cos(f * lat_n * np.pi),
                    np.sin(f * lon_n * np.pi),
                    np.cos(f * lon_n * np.pi),
                ]
            )
        # Pad to 64
        while len(features) < 64:
            features.append(0.0)
        features = features[:64]

        feat = torch.tensor(features, dtype=torch.float32)
        emb = feat @ self._loc_proj + self._loc_bias
        emb = emb / (emb.norm() + 1e-8)
        return emb
