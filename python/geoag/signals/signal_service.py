"""Signal service: orchestrates nowcasts → macro + trade ideas."""

from __future__ import annotations

import json

import pandas as pd

from geoag.common.config import DATA_LAKE_DIR, ConfigStore, get_config
from geoag.common.logging import get_logger
from geoag.common.schemas import MacroIndicators, RegionNowcast, TradeIdea
from geoag.model.infer import InferenceService
from geoag.signals.arb_logic import ArbEngine
from geoag.signals.macro import compute_macro

logger = get_logger("signals.service")


class SignalService:
    """Full signal pipeline: features → nowcasts → macro + trade ideas."""

    def __init__(self, config: ConfigStore | None = None) -> None:
        self.config = config or get_config()
        self.inference = InferenceService(self.config)
        self.arb_engine = ArbEngine(self.config)
        self.lake_dir = DATA_LAKE_DIR

    def run(self) -> dict[str, object]:
        """Run the full signal pipeline.

        Returns dict with:
          - nowcasts: list[RegionNowcast]
          - macro: MacroIndicators
          - trade_ideas: list[TradeIdea]
        """
        # Load features
        features_path = self.lake_dir / "features.parquet"
        if not features_path.exists():
            logger.error("features.parquet not found — run feature service first")
            return {"nowcasts": [], "macro": None, "trade_ideas": []}

        features_df = pd.read_parquet(features_path)

        # Run inference
        nowcasts = self.inference.infer(features_df)

        # Compute macro
        macro = compute_macro(nowcasts, self.config)

        # Generate trade ideas
        trade_ideas = self.arb_engine.generate_ideas(nowcasts)

        # Persist outputs
        self._save_outputs(nowcasts, macro, trade_ideas)

        return {
            "nowcasts": nowcasts,
            "macro": macro,
            "trade_ideas": trade_ideas,
        }

    def _save_outputs(
        self,
        nowcasts: list[RegionNowcast],
        macro: MacroIndicators,
        trade_ideas: list[TradeIdea],
    ) -> None:
        """Save outputs to data lake."""
        self.lake_dir.mkdir(parents=True, exist_ok=True)

        # Nowcasts
        nc_records = [nc.model_dump() for nc in nowcasts]
        nc_df = pd.DataFrame(nc_records)
        nc_df.to_parquet(self.lake_dir / "nowcasts.parquet", index=False)

        # Macro
        macro_dict = macro.model_dump()
        macro_df = pd.DataFrame([macro_dict])
        macro_df.to_parquet(self.lake_dir / "macro.parquet", index=False)

        # Trade ideas
        ideas_data = [ti.model_dump() for ti in trade_ideas]
        with open(self.lake_dir / "trade_ideas.json", "w") as f:
            json.dump(ideas_data, f, indent=2, default=str)

        logger.info(
            "Saved %d nowcasts, macro indicators, %d trade ideas",
            len(nowcasts),
            len(trade_ideas),
        )


def run_signals() -> dict[str, object]:
    """Convenience entry point."""
    svc = SignalService()
    return svc.run()


if __name__ == "__main__":
    from geoag.common.logging import setup_logging

    setup_logging("INFO")
    result = run_signals()
    print(f"Produced {len(result['nowcasts'])} nowcasts, {len(result['trade_ideas'])} trade ideas")
