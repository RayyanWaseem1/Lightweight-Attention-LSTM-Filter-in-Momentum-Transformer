"""Portfolio construction, rebalancing and cost tests.

These pin the Tier-0 defects: the static-snapshot portfolio, zero realised
turnover being charged as if it were full turnover, and the extended-hours
renormalisation that let one illiquid print become the whole book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Models.config import BacktestConfig
from Utils.Market_data import tradeable_returns
from Utils.Portfolio import (
    build_weight_matrix,
    construct_weights,
    rebalance_timestamps,
    run_portfolio_backtest,
)


@pytest.fixture
def toy_market():
    """Three symbols on a 7-bar/day calendar over 40 sessions."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2022-01-03", periods=40, freq="B")
    stamps = [d + pd.Timedelta(hours=13 + b) for d in days for b in range(7)]
    calendar = pd.DatetimeIndex(stamps)

    rows = []
    for symbol, drift in [("AAA", 0.0004), ("BBB", 0.0), ("CCC", -0.0003)]:
        close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.006, len(calendar))))
        rows.append(pd.DataFrame({"timestamp": calendar, "symbol": symbol,
                                  "close": close}))
    prices = pd.concat(rows, ignore_index=True)
    returns = tradeable_returns(prices)
    return prices, returns, calendar


def test_rebalance_schedule_counts_are_sane(toy_market):
    _, _, calendar = toy_market

    daily = rebalance_timestamps(calendar, "daily")
    weekly = rebalance_timestamps(calendar, "weekly")
    monthly = rebalance_timestamps(calendar, "monthly")

    assert len(daily) == 40                      # one per session
    assert 7 <= len(weekly) <= 9                 # 40 business days ~ 8 ISO weeks
    assert len(monthly) == 2                     # Jan and Feb
    assert len(daily) > len(weekly) > len(monthly)

    # Every rebalance stamp must be on the calendar.
    assert set(weekly).issubset(set(calendar))


def test_rebalance_schedule_is_derived_not_hardcoded():
    """The old code matched ``hour == 10`` and silently found zero rebalances
    when the data's clock differed."""
    odd_hours = pd.DatetimeIndex(
        [pd.Timestamp("2022-03-01") + pd.Timedelta(hours=3 + i) for i in range(50)]
    )
    schedule = rebalance_timestamps(odd_hours, "daily")
    assert len(schedule) > 0


def test_construct_weights_normalises_gross_exposure():
    signals = pd.Series({"A": 0.9, "B": -0.6, "C": 0.3, "D": 0.05})
    weights = construct_weights(signals, max_positions=3, max_weight=0.5)

    assert len(weights) == 3
    assert "D" not in weights.index                       # weakest signal dropped
    assert weights.abs().sum() == pytest.approx(1.0)
    assert weights["B"] < 0                               # sign preserved
    assert weights.abs().max() <= 0.5 + 1e-9


def test_dynamic_weights_produce_nonzero_turnover(toy_market):
    """The headline Tier-0 fix.

    The previous pipeline kept only the LAST prediction per symbol and applied
    those weights across the whole test period, so realised turnover was exactly
    zero while the cost model charged for full turnover at every 'rebalance'.
    """
    prices, returns, calendar = toy_market
    rng = np.random.default_rng(1)

    # Genuinely time-varying predictions.
    predictions = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, "prediction": rng.normal()}
            for ts in calendar
            for sym in ("AAA", "BBB", "CCC")
        ]
    )

    cfg = BacktestConfig(rebalance_frequency="weekly", max_positions=2)
    result = run_portfolio_backtest(predictions, returns, calendar, cfg)

    assert result["total_turnover"] > 0.0
    assert result["mean_turnover"] > 0.0
    assert result["n_rebalances"] == len(rebalance_timestamps(calendar, "weekly"))
    assert result["total_cost"] > 0.0

    # Costs must be exactly turnover * rate -- charged on trades that happened.
    assert result["total_cost"] == pytest.approx(
        result["total_turnover"] * cfg.cost_rate, rel=1e-9
    )


def test_static_predictions_incur_no_ongoing_cost(toy_market):
    """A book that never trades must be charged only for its initial entry."""
    prices, returns, calendar = toy_market
    fixed = {"AAA": 0.9, "BBB": 0.1, "CCC": -0.5}
    predictions = pd.DataFrame(
        [{"timestamp": ts, "symbol": s, "prediction": v}
         for ts in calendar for s, v in fixed.items()]
    )

    cfg = BacktestConfig(rebalance_frequency="weekly", max_positions=3)
    result = run_portfolio_backtest(predictions, returns, calendar, cfg)

    # Entry turnover is 1.0 (gross), and nothing after that.
    assert result["total_turnover"] == pytest.approx(1.0, abs=1e-6)
    assert result["turnover"].iloc[1:].sum() == pytest.approx(0.0, abs=1e-9)


def test_weights_are_held_between_rebalances(toy_market):
    prices, returns, calendar = toy_market
    rng = np.random.default_rng(2)
    predictions = pd.DataFrame(
        [{"timestamp": ts, "symbol": s, "prediction": rng.normal()}
         for ts in calendar for s in ("AAA", "BBB", "CCC")]
    )
    schedule = rebalance_timestamps(calendar, "weekly")
    weights = build_weight_matrix(predictions, calendar, schedule, 2, 0.6)

    # Between two consecutive rebalances the weights must not move.
    first, second = schedule[0], schedule[1]
    first_row = weights.loc[[first]].iloc[0]
    between = weights.loc[(weights.index > first) & (weights.index < second)]
    for _, row in between.iterrows():
        pd.testing.assert_series_equal(row, first_row, check_names=False)


def test_missing_bar_earns_zero_not_a_renormalised_full_weight(toy_market):
    """The extended-hours pathology from audit 0.5.

    When a held name does not print, the old code renormalised by ``total_weight``
    so the remaining name became the entire portfolio return at 100% gross.
    """
    prices, returns, calendar = toy_market
    predictions = pd.DataFrame(
        [{"timestamp": ts, "symbol": s, "prediction": v}
         for ts in calendar for s, v in {"AAA": 1.0, "BBB": 1.0}.items()]
    )

    # Drop every BBB return: it is held but never prints.
    holed = returns[returns["symbol"] != "BBB"]

    cfg = BacktestConfig(rebalance_frequency="weekly", max_positions=2)
    result = run_portfolio_backtest(predictions, holed, calendar, cfg)

    full = run_portfolio_backtest(predictions, returns, calendar, cfg)

    # With half the book not printing, returns must be roughly halved -- not
    # renormalised back up to the full-gross magnitude.
    assert result["gross_returns"].std() < 0.75 * full["gross_returns"].std()


def test_net_returns_are_gross_minus_costs(toy_market):
    prices, returns, calendar = toy_market
    rng = np.random.default_rng(3)
    predictions = pd.DataFrame(
        [{"timestamp": ts, "symbol": s, "prediction": rng.normal()}
         for ts in calendar for s in ("AAA", "BBB", "CCC")]
    )
    cfg = BacktestConfig(rebalance_frequency="daily", max_positions=2)
    result = run_portfolio_backtest(predictions, returns, calendar, cfg)

    expected = result["gross_returns"] - result["costs"]
    pd.testing.assert_series_equal(result["net_returns"], expected, check_names=False)
    assert (result["net_returns"] <= result["gross_returns"] + 1e-12).all()


def test_returns_are_not_winsorised(toy_market):
    """Realised P&L must pass through untouched -- no 1st/99th clipping."""
    prices, returns, calendar = toy_market

    # Plant a large single-bar move in a name we will hold outright.
    spike_ts = calendar[100]
    returns = returns.copy()
    mask = (returns["symbol"] == "AAA") & (returns["timestamp"] == spike_ts)
    returns.loc[mask, "tradeable_return"] = -0.35

    predictions = pd.DataFrame(
        [{"timestamp": ts, "symbol": "AAA", "prediction": 1.0} for ts in calendar]
    )
    cfg = BacktestConfig(rebalance_frequency="weekly", max_positions=1)
    result = run_portfolio_backtest(predictions, returns, calendar, cfg)

    # The full -35% bar must survive into the P&L.
    assert result["gross_returns"].min() == pytest.approx(-0.35, abs=1e-9)
