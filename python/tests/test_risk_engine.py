"""Tests for the risk engine."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from geoag.common.config import ConfigStore
from geoag.common.schemas import RiskTag
from geoag.signals.risk import RiskEngine
from geoag.signals.sessions import SessionChecker


class TestSessionChecker:
    """Test session checking logic."""

    def test_cme_weekday_pit_session(self, config: ConfigStore) -> None:
        """CME pit session should be open during pit hours on a weekday."""
        checker = SessionChecker(config)
        # Wednesday 10:00 AM Chicago time → pit session
        dt = datetime(2026, 2, 11, 16, 0, 0, tzinfo=ZoneInfo("UTC"))  # 10 AM Chicago
        result = checker.check("ZC", at=dt)
        assert result["tradable_now"] is True
        assert result["session_type"] in ("PIT", "ELECTRONIC")

    def test_cme_weekend_closed(self, config: ConfigStore) -> None:
        """CME should be closed on Saturday."""
        checker = SessionChecker(config)
        # Saturday noon UTC
        dt = datetime(2026, 2, 14, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = checker.check("ZC", at=dt)
        assert result["tradable_now"] is False
        assert result["session_type"] == "WEEKEND"
        assert result["best_window"] is not None

    def test_cme_electronic_session(self, config: ConfigStore) -> None:
        """CME electronic session should be open late evening Chicago time."""
        checker = SessionChecker(config)
        # Tuesday 9 PM Chicago = Wednesday 3 AM UTC
        dt = datetime(2026, 2, 11, 3, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = checker.check("ZC", at=dt)
        assert result["tradable_now"] is True
        assert result["session_type"] == "ELECTRONIC"

    def test_unknown_instrument(self, config: ConfigStore) -> None:
        checker = SessionChecker(config)
        result = checker.check("FAKE", at=datetime.now(ZoneInfo("UTC")))
        assert result["tradable_now"] is False
        assert result["session_type"] == "UNKNOWN_INSTRUMENT"

    def test_check_all(self, config: ConfigStore) -> None:
        checker = SessionChecker(config)
        results = checker.check_all()
        assert "ZC" in results
        assert "ZW" in results
        assert "ZS" in results


class TestRiskEngine:
    """Test risk evaluation."""

    def test_pin_risk_near_strike(self, config: ConfigStore) -> None:
        """When spot is near an OI peak, pin risk should be elevated."""
        engine = RiskEngine(config)
        # ZC OI peaks: 420, 440, 460, 480, 500
        # Spot at 460 = right on the biggest OI peak
        result = engine.evaluate(["ZC"], spot_prices={"ZC": 460.0})
        assert result["pin_risk_score"] > 0.3

    def test_pin_risk_far_from_strike(self, config: ConfigStore) -> None:
        """When spot is far from OI peaks, pin risk should be low."""
        engine = RiskEngine(config)
        result = engine.evaluate(["ZC"], spot_prices={"ZC": 510.0})
        assert result["pin_risk_score"] < 0.3

    def test_no_pin_risk_without_spot(self, config: ConfigStore) -> None:
        """Without spot price, pin risk should be 0."""
        engine = RiskEngine(config)
        result = engine.evaluate(["ZC"])
        assert result["pin_risk_score"] == 0.0

    def test_spec_mismatch_detection(self, config: ConfigStore) -> None:
        """Mixed futures + ETF should flag spec mismatch."""
        engine = RiskEngine(config)
        result = engine.evaluate(["ZC", "VEGI"])
        tags = result["risk_tags"]
        # Should detect settlement and/or timezone mismatch
        assert RiskTag.SPEC_MISMATCH in tags or RiskTag.LOW_LIQUIDITY in tags

    def test_low_liquidity_etf(self, config: ConfigStore) -> None:
        """ETFs should be flagged as potentially low liquidity for ag."""
        engine = RiskEngine(config)
        result = engine.evaluate(["VEGI"])
        assert RiskTag.LOW_LIQUIDITY in result["risk_tags"]

    def test_risk_tags_are_risk_tag_enum(self, config: ConfigStore) -> None:
        """All risk tags should be valid RiskTag enum values."""
        engine = RiskEngine(config)
        result = engine.evaluate(["ZC", "ZW", "VEGI"])
        for tag in result["risk_tags"]:
            assert isinstance(tag, RiskTag)

    def test_evaluate_empty_symbols(self, config: ConfigStore) -> None:
        """Empty symbol list should return clean result."""
        engine = RiskEngine(config)
        result = engine.evaluate([])
        assert result["tradable_now"] is True
        assert result["pin_risk_score"] == 0.0
