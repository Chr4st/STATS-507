"""Training loop for the nowcast model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geoag.common.config import DATA_LAKE_DIR, get_config
from geoag.common.logging import get_logger
from geoag.model.dataset import build_dataset
from geoag.model.model import NowcastModel, build_model

logger = get_logger("model.train")

MODELS_DIR = DATA_LAKE_DIR / "models"


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom collate: stack tensors, skip region_id."""
    keys = [k for k in batch[0] if k != "region_id"]
    collated: dict[str, torch.Tensor] = {}
    for k in keys:
        collated[k] = torch.stack([b[k] for b in batch])
    return collated


def train_model(
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 8,
    save_path: Path | None = None,
) -> NowcastModel:
    """Train the nowcast model on available feature data.

    Returns the trained model.
    """
    config = get_config()
    mc = config.settings.model

    dataset = build_dataset()
    logger.info("Dataset size: %d samples", len(dataset))

    if len(dataset) == 0:
        logger.warning("Empty dataset — returning untrained model")
        return build_model(
            embedding_dim=config.settings.embedding.dim,
            weather_dim=8,
            hidden_dim=mc.hidden_dim,
            num_layers=mc.num_layers,
            dropout=mc.dropout,
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = build_model(
        embedding_dim=config.settings.embedding.dim,
        weather_dim=8,
        hidden_dim=mc.hidden_dim,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            optimizer.zero_grad()

            out = model(
                batch["e_img"],
                batch["e_loc"],
                batch["delta_e"],
                batch["weather"],
            )

            loss = (
                mse(out["stress_index"], batch["stress_target"])
                + mse(out["growth_index"], batch["growth_target"])
                + mse(out["yield_shock_mean"], batch["yield_shock_mean_target"])
                + mse(out["yield_shock_log_sigma"], batch["yield_shock_sigma_target"].log().clamp(-5, 5))
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("Epoch %d/%d — loss: %.5f", epoch + 1, epochs, avg_loss)

    # Save model
    save_to = save_path or (MODELS_DIR / "nowcast_model.pt")
    save_to.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_to)
    logger.info("Saved model to %s", save_to)

    return model


if __name__ == "__main__":
    from geoag.common.logging import setup_logging

    setup_logging("INFO")
    train_model()
