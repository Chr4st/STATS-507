"""Tests for the stub embedding backend."""

from __future__ import annotations

import numpy as np
import torch

from geoag.features.stub_backend import StubBackend


class TestStubBackend:
    """Test suite for the deterministic stub embedding backend."""

    def test_embedding_dim(self) -> None:
        backend = StubBackend(dim=512, seed=42)
        assert backend.embedding_dim == 512

    def test_encode_image_shape(self) -> None:
        backend = StubBackend(dim=256, seed=42)
        tile = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        emb = backend.encode_image(tile)
        assert emb.shape == (256,)
        assert emb.dtype == torch.float32

    def test_encode_image_deterministic(self) -> None:
        """Same tile → same embedding every time."""
        backend = StubBackend(dim=512, seed=42)
        tile = np.random.RandomState(99).randint(0, 255, (64, 64, 3)).astype(np.uint8)
        emb1 = backend.encode_image(tile)
        emb2 = backend.encode_image(tile)
        assert torch.allclose(emb1, emb2, atol=1e-6)

    def test_encode_image_normalized(self) -> None:
        """Embeddings should be L2-normalized."""
        backend = StubBackend(dim=512, seed=42)
        tile = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        emb = backend.encode_image(tile)
        norm = emb.norm().item()
        assert abs(norm - 1.0) < 0.01

    def test_different_tiles_different_embeddings(self) -> None:
        """Different tiles should produce different embeddings."""
        backend = StubBackend(dim=512, seed=42)
        tile_a = np.zeros((64, 64, 3), dtype=np.uint8)
        tile_b = np.full((64, 64, 3), 255, dtype=np.uint8)
        emb_a = backend.encode_image(tile_a)
        emb_b = backend.encode_image(tile_b)
        # Should not be identical
        assert not torch.allclose(emb_a, emb_b, atol=1e-3)

    def test_encode_location_shape(self) -> None:
        backend = StubBackend(dim=512, seed=42)
        emb = backend.encode_location(41.5, -89.0)
        assert emb.shape == (512,)
        assert emb.dtype == torch.float32

    def test_encode_location_deterministic(self) -> None:
        """Same coordinates → same embedding."""
        backend = StubBackend(dim=512, seed=42)
        emb1 = backend.encode_location(41.5, -89.0)
        emb2 = backend.encode_location(41.5, -89.0)
        assert torch.allclose(emb1, emb2, atol=1e-6)

    def test_encode_location_normalized(self) -> None:
        backend = StubBackend(dim=512, seed=42)
        emb = backend.encode_location(41.5, -89.0)
        norm = emb.norm().item()
        assert abs(norm - 1.0) < 0.01

    def test_different_locations_different_embeddings(self) -> None:
        backend = StubBackend(dim=512, seed=42)
        emb_us = backend.encode_location(41.5, -89.0)
        emb_br = backend.encode_location(-12.5, -55.5)
        assert not torch.allclose(emb_us, emb_br, atol=1e-3)

    def test_encode_image_batch(self) -> None:
        backend = StubBackend(dim=256, seed=42)
        tiles = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
        ]
        batch = backend.encode_image_batch(tiles)
        assert batch.shape == (3, 256)

    def test_encode_location_batch(self) -> None:
        backend = StubBackend(dim=256, seed=42)
        coords = [(41.5, -89.0), (-12.5, -55.5), (48.1, 1.5)]
        batch = backend.encode_location_batch(coords)
        assert batch.shape == (3, 256)

    def test_seed_reproducibility(self) -> None:
        """Same seed → same projection matrices → same embeddings."""
        b1 = StubBackend(dim=128, seed=123)
        b2 = StubBackend(dim=128, seed=123)
        tile = np.ones((32, 32, 3), dtype=np.uint8) * 128
        assert torch.allclose(b1.encode_image(tile), b2.encode_image(tile))
        assert torch.allclose(
            b1.encode_location(0, 0), b2.encode_location(0, 0)
        )

    def test_different_seeds_different_projections(self) -> None:
        """Different seeds should produce different embeddings."""
        b1 = StubBackend(dim=128, seed=1)
        b2 = StubBackend(dim=128, seed=2)
        tile = np.ones((32, 32, 3), dtype=np.uint8) * 128
        e1 = b1.encode_image(tile)
        e2 = b2.encode_image(tile)
        assert not torch.allclose(e1, e2, atol=1e-3)
