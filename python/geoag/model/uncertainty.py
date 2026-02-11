"""Uncertainty estimation via MC Dropout."""

from __future__ import annotations

import torch

from geoag.model.model import NowcastModel


def enable_mc_dropout(model: NowcastModel) -> None:
    """Enable dropout layers during inference for MC sampling."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_predict(
    model: NowcastModel,
    e_img: torch.Tensor,
    e_loc: torch.Tensor,
    delta_e: torch.Tensor,
    weather: torch.Tensor,
    n_samples: int = 10,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run MC Dropout inference and return mean + std for each output.

    Args:
        model: Trained NowcastModel.
        e_img, e_loc, delta_e, weather: Input tensors (B, T, D).
        n_samples: Number of MC forward passes.

    Returns:
        Dict mapping output name -> {"mean": Tensor, "std": Tensor} each (B,).
    """
    model.eval()
    enable_mc_dropout(model)

    predictions: dict[str, list[torch.Tensor]] = {
        "stress_index": [],
        "growth_index": [],
        "yield_shock_mean": [],
        "yield_shock_log_sigma": [],
    }

    with torch.no_grad():
        for _ in range(n_samples):
            out = model(e_img, e_loc, delta_e, weather)
            for key in predictions:
                predictions[key].append(out[key])

    results: dict[str, dict[str, torch.Tensor]] = {}
    for key, preds in predictions.items():
        stacked = torch.stack(preds, dim=0)  # (n_samples, B)
        results[key] = {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
        }

    return results


def compute_confidence(
    sigma: torch.Tensor,
    missingness: torch.Tensor,
    cloudiness: torch.Tensor,
    a: float = 3.0,
    b: float = 2.0,
    c: float = 1.5,
    d: float = 1.0,
) -> torch.Tensor:
    """Compute confidence score from uncertainty and data quality.

    confidence = sigmoid(a - b * sigma - c * missingness - d * cloudiness)

    Args:
        sigma: Model uncertainty (std of MC predictions). (B,)
        missingness: Fraction of missing data. (B,)
        cloudiness: Fraction cloud cover. (B,)
        a, b, c, d: Coefficients from config.

    Returns:
        Confidence scores in [0, 1]. (B,)
    """
    logit = a - b * sigma - c * missingness - d * cloudiness
    return torch.sigmoid(logit)
