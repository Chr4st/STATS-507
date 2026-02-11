"""Abstract embedding provider interface for SatCLIP-style encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class EmbeddingProvider(ABC):
    """Interface for image + location encoders."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of output embeddings."""
        ...

    @abstractmethod
    def encode_image(self, tile: np.ndarray) -> torch.Tensor:
        """Encode an image tile (H, W, C) to an embedding vector.

        Args:
            tile: uint8 array of shape (H, W, C).

        Returns:
            1-D tensor of shape (embedding_dim,).
        """
        ...

    @abstractmethod
    def encode_location(self, lat: float, lon: float) -> torch.Tensor:
        """Encode a geographic location to an embedding vector.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            1-D tensor of shape (embedding_dim,).
        """
        ...

    def encode_image_batch(self, tiles: list[np.ndarray]) -> torch.Tensor:
        """Encode a batch of tiles. Default: loop over encode_image.

        Returns:
            Tensor of shape (N, embedding_dim).
        """
        return torch.stack([self.encode_image(t) for t in tiles])

    def encode_location_batch(
        self, coords: list[tuple[float, float]]
    ) -> torch.Tensor:
        """Encode a batch of (lat, lon) pairs. Default: loop.

        Returns:
            Tensor of shape (N, embedding_dim).
        """
        return torch.stack([self.encode_location(lat, lon) for lat, lon in coords])
