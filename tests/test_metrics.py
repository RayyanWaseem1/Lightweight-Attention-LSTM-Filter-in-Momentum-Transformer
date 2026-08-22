"""Metric correctness tests.

Each test here pins a specific defect found in the audit so it cannot return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Models.config import PERIODS_PER_YEAR, TRADING_DAYS_PER_YEAR
from Utils import Metrics


def test_longest_streak_counts_only_the_requested_value():
    """The old implementation counted runs of *either* boolean value.

    ``longest_streak(returns > 0)`` could therefore report the longest LOSING
    streak, which is what ``win_streak`` was doing.
    """
    # 2 wins, then 5 losses, then 3 wins.
    wins = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    assert Metrics.longest_streak(wins, True) == 3
    assert Metrics.longest_streak(wins, False) == 5

    # The buggy "runs of identical values" logic would return 5 for both.
    assert Metrics.longest_streak(wins, True) != Metrics.longest_streak(wins, False)


def test_longest_streak_edges():
    assert Metrics.longest_streak([], True) == 0
    assert Metrics.longest_streak([False, False], True) == 0
    assert Metrics.longest_streak([True, True, True], True) == 3


def test_sortino_is_annualised_exactly_once():
    """The old code applied sqrt(ppy) twice, giving a ~64x error.

    That is why the repo reported Sortino 0.029 next to Sharpe 1.87 -- a
    combination that cannot occur.
    """
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, 5000)

    sortino = Metrics.sortino_ratio(r, PERIODS_PER_YEAR)

    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    expected = r.mean() / downside * np.sqrt(PERIODS_PER_YEAR)
    assert sortino == pytest.approx(expected, rel=1e-9)

    # Sanity: for a positive-mean series Sortino should exceed Sharpe, and both
    # should be the same order of magnitude.
    sharpe = Metrics.sharpe_ratio(r, PERIODS_PER_YEAR)
    assert sortino > sharpe
    assert sortino < 10 * sharpe


def test_sortino_uses_full_sample_downside_deviation():
    """Downside deviation is about the target over the FULL sample.

    Not the standard deviation of the losing subset about its own mean, which
    is a different (and larger) quantity.
    """
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    downside_full = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    downside_subset = np.std(r[r < 0], ddof=1)
    assert downside_full != pytest.approx(downside_subset)

    got = Metrics.sortino_ratio(r, 1.0)
    assert got == pytest.approx(r.mean() / downside_full, rel=1e-9)


def test_max_drawdown_is_not_floored():
    """A strategy that loses more than 100% must report that it did."""
    # Equity: 1.1 -> 0.11 -> 0.011 -> 0.0011 -> 0.00011, i.e. -99.99% from peak.
    r = np.array([0.1, -0.9, -0.9, -0.9, -0.9])
    dd = Metrics.max_drawdown(r)

    assert dd < -0.999, f"drawdown {dd} was floored"
    # The old code clamped with max(dd, -0.999), so it could never go below it.
    assert dd != pytest.approx(-0.999, abs=1e-6)


def test_annualized_return_compounds_over_the_real_horizon():
    """CAGR, not ``mean * periods_per_year``."""
    ppy = 252.0
    r = np.full(504, 0.001)  # two years of a constant daily return

    cagr = Metrics.annualized_return(r, ppy)
    expected = (1.001 ** 504) ** (1 / 2) - 1
    assert cagr == pytest.approx(expected, rel=1e-9)

    naive = r.mean() * ppy
    assert cagr != pytest.approx(naive, rel=1e-3)


def test_daily_sharpe_is_independent_of_bars_per_day():
    """Aggregating to daily removes the bars-per-year assumption entirely.

    Two series with the same daily P&L but different intraday bar counts must
    produce the same daily Sharpe.
    """
    rng = np.random.default_rng(1)
    days = pd.date_range("2022-01-03", periods=200, freq="B")

    def build(bars_per_day: int) -> pd.Series:
        rng_local = np.random.default_rng(42)
        index, values = [], []
        for day in days:
            daily = rng_local.normal(0.0005, 0.01)
            per_bar = (1 + daily) ** (1 / bars_per_day) - 1
            for b in range(bars_per_day):
                # Spread the bars WITHIN the session. Using `hours=9+b` pushed
                # the 24-bar case past midnight into the next calendar date,
                # which changed the daily grouping rather than the bar count.
                offset = pd.Timedelta(seconds=int(b * 86400 / bars_per_day))
                index.append(day + offset)
                values.append(per_bar)
        return pd.Series(values, index=pd.DatetimeIndex(index))

    s7 = Metrics.daily_sharpe(build(7))
    s24 = Metrics.daily_sharpe(build(24))
    assert s7 == pytest.approx(s24, rel=1e-6)


def test_wrong_annualisation_inflates_sharpe_by_a_known_factor():
    """Pins the specific magnitude of the 252*24 error."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0002, 0.005, 10_000)

    honest = Metrics.sharpe_ratio(r, PERIODS_PER_YEAR)      # 252 * 7
    inflated = Metrics.sharpe_ratio(r, 252 * 24)            # the old constant

    assert inflated / honest == pytest.approx(np.sqrt(24 / 7), rel=1e-9)
    assert inflated / honest == pytest.approx(1.852, abs=0.01)


def test_cross_sectional_ic_matches_hand_computation():
    frame = pd.DataFrame(
        {
            "timestamp": ["t1"] * 4 + ["t2"] * 4,
            "symbol": list("abcd") * 2,
            "prediction": [1, 2, 3, 4, 4, 3, 2, 1],
            "target": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )
    result = Metrics.cross_sectional_ic(frame)
    # Perfect agreement then perfect disagreement -> mean IC of zero.
    assert result["mean_ic"] == pytest.approx(0.0, abs=1e-12)
    assert result["n_timestamps"] == 2


def test_alpha_beta_recovers_planted_coefficients():
    rng = np.random.default_rng(4)
    n = 5000
    benchmark = pd.Series(rng.normal(0.0003, 0.01, n))
    alpha_per_bar, beta = 0.0001, 0.45
    strategy = alpha_per_bar + beta * benchmark + rng.normal(0, 0.002, n)

    got = Metrics.alpha_beta(strategy, benchmark, periods_per_year=1.0)
    assert got["beta"] == pytest.approx(beta, abs=0.02)
    assert got["alpha_annual"] == pytest.approx(alpha_per_bar, abs=5e-5)
    assert got["alpha_tstat"] > 2


def test_deflated_sharpe_falls_as_trials_rise():
    """More configurations tried => the same Sharpe is less convincing."""
    rng = np.random.default_rng(6)
    r = rng.normal(0.0004, 0.01, 4000)

    one = Metrics.deflated_sharpe_ratio(r, n_trials=1)
    many = Metrics.deflated_sharpe_ratio(r, n_trials=100)

    assert many["deflated_sharpe"] < one["deflated_sharpe"]
    assert many["expected_max_sharpe"] > one["expected_max_sharpe"]


def test_turnover_of_static_weights_is_zero_after_entry():
    """The specific pathology behind the fictional '48 rebalances, 30% cost drag'."""
    index = pd.date_range("2022-01-03", periods=10, freq="D")
    static = pd.DataFrame(
        np.tile([0.5, 0.5], (10, 1)), index=index, columns=["A", "B"]
    )
    turnover = Metrics.turnover(static)
    assert turnover.iloc[0] == pytest.approx(1.0)   # entry from flat
    assert turnover.iloc[1:].sum() == pytest.approx(0.0)


def test_performance_summary_has_the_headline_fields():
    rng = np.random.default_rng(8)
    index = pd.date_range("2022-01-03", periods=1000, freq="h")
    s = pd.Series(rng.normal(0.0001, 0.004, 1000), index=index)

    summary = Metrics.performance_summary(s)
    for key in ("sharpe_ratio", "daily_sharpe", "sortino_ratio", "max_drawdown",
                "annualized_return", "volatility", "n_periods", "periods_per_year"):
        assert key in summary
    assert summary["n_periods"] == 1000