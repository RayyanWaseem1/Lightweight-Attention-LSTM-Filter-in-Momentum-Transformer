""" Portfolio construction, rebalancing, transaction costs and baselines. 

Weights are rebuilt at each scheduled rebalance from the cross-section of 
predictions **available at that timestamp**, held between rebalances, and 
charged ``sum_i |w_{i,t} - w_{i, t-1} * cost_rate`` on realized turnover

Weights formed from predictions at ``t`` are multiplied by 
``tradeable_return[t] = close[t]/close[t-1] - 1``. Feature row ``t`` is built
from ``close.shift(1)``, so its information cutoff is the end of the bar ``t-1``;
a position established there earns bar ``t``'s return, which uses ``close[t]``
-- a price no feature at row ``t`` has seen. See ``Utils.Market_data.tradeable_returns`` for the full derivation
and ``tests/test_causality.py`` for the assertion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple 

import numpy as np 
import pandas as pd 

from Models.config import BacktestConfig, PERIODS_PER_YEAR, bars
from Utils import Metrics 

EPS = 1e-12
GROSS_EXPOSURE_TOL = 1e-6

### Rebalance Schedule ###

def rebalance_timestamps(
    calendar: pd.DatetimeIndex, frequency: str = "weekly"
) -> pd.DatetimeIndex:
    """ First calendar bar of each day / ISO week / month
    
    Derived from the actual calendar rather than by matching a hardcoded hour, 
    so it cannot slightly produce zero rebalances when the data's clock differs
    from the one assumed.
    """
    calendar = pd.DatetimeIndex(sorted(calendar))
    if len(calendar) == 0:
        return calendar 

    frame = pd.DataFrame(index = calendar)
    if frequency == "daily":
        key = calendar.date
    elif frequency == "weekly":
        iso = calendar.isocalendar()
        key = list(zip(iso.year, iso.week))
    elif frequency == "monthly":
        key = list(zip(calendar.year, calendar.month))
    else:
        raise ValueError(f"Invalid rebalance frequency: {frequency!r}")

    frame["key"] = key 
    firsts = frame.groupby("key", sort = False).head(1)
    return pd.DatetimeIndex(sorted(firsts.index))

### Cross sectional weight construction ###

def construct_weights(
    signals: pd.Series,
    max_positions: int = 10,
    max_weight: float = 0.2,
) -> pd.Series:
    """ Turn one timestamp's cross-section of signals into portfolio weights.
    
    Signals are already bounded positions in [-1, 1] (the model head ends in 
    ``tanh``) and volatility-targeted, so no ``min-signal`` threshold or ``std * 2`` 
    rescaling is applied -- those were patches for the missing activation, not portfolio logic
    
    Gross exposure is normalized to 1.0
    """

    signals = signals.dropna()
    if signals.empty:
        return pd.Series(dtype = float)

    top = signals.reindex(signals.abs().sort_values(ascending = False).index)
    top = top.iloc[:max_positions]

    gross = top.abs().sum()
    if gross < EPS:
        return pd.Series(0.0, index = top.index)

    weights = (top / gross).clip(-max_weight, max_weight)

    gross = weights.abs().sum()
    if gross < EPS:
        return pd.Series(0.0, index = top.index)
    return weights / gross 

def build_weight_matrix(
    predictions: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    max_positions: int = 10,
    max_weight: float = 0.2,
) -> pd.DataFrame:
    """ [time x symbol] weights, rebuilt on schedule and held in between
    
    ``predictions`` needs columns ``timestamp``, ``symbol``, ``prediction``.
    """
    if predictions.empty:
        return pd.DataFrame(index = calendar)

    pred_matrix = predictions.pivot_table(
        index = "timestamp", columns = "symbol", values = "prediction", aggfunc = "last"
    ).reindex(calendar)

    rebalance_set = set(rebalance_dates)
    rows: Dict[pd.Timestamp, pd.Series] = {}

    for ts in calendar:
        if ts not in rebalance_set:
            continue 
        cross_section = pred_matrix.loc[[ts]].iloc[0].dropna()
        if cross_section.empty:
            continue 
        rows[ts] = construct_weights(cross_section, max_positions, max_weight)

    if not rows:
        return pd.DataFrame(index = calendar)

    weights = pd.DataFrame(rows).T.reindex(columns = pred_matrix.columns).fillna(0.0)
    # Hold positions between rebalances instead of renormalizing gross exposure
    # on bars where only some names printed 
    weights = weights.reindex(calendar).ffill().fillna(0.0)
    return weights 

### Backtest ###

def run_portfolio_backtest(
    predictions: pd.DataFrame,
    tradeable_returns: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: Optional[BacktestConfig] = None,
    cost_rate: Optional[float] = None,
) -> Dict:
    """ Run the scheduled-rebalance backetest.
    
    Returns a dict with the gross/net return series, the weight matrix, the
    realized turnover series and the aggregate cost figures
    """

    config = config or BacktestConfig()
    cost_rate = config.cost_rate if cost_rate is None else cost_rate 

    calendar = pd.DatetimeIndex(sorted(calendar))
    schedule = rebalance_timestamps(calendar, config.rebalance_frequency)

    weights = build_weight_matrix(
        predictions, calendar, schedule, config.max_positions, config.max_weight
    )
    if weights.empty or weights.shape[1] == 0:
        empty = pd.Series(dtype = float, index = pd.DatetimeIndex([]))
        exposure = {
            "mean_gross": 0.0,
            "median_gross": 0.0,
            "p95_gross": 0.0,
            "max_gross": 0.0,
            "mean_net": 0.0,
            "median_net": 0.0,
            "p95_abs_net": 0.0,
            "max_abs_net": 0.0,
            "mean_position_count": 0.0,
            "max_position_count": 0,
        }
        return {"gross_returns": empty, "net_returns": empty,
                 "weights": weights, "gross_exposure": empty, "net_exposure": empty,
                 "position_count": empty, "exposure": exposure, "turnover": empty,
                 "n_rebalances": 0, "total_turnover": 0.0, "mean_turnover": 0.0,
                 "total_cost": 0.0, "cost_rate": float(cost_rate),
                 "rebalance_frequency": config.rebalance_frequency}

    ret_matrix = tradeable_returns.pivot_table(
        index = "timestamp", columns = "symbol", values = "tradeable_return", aggfunc = "last"
    ).reindex(index = calendar, columns = weights.columns)

    ret_matrix = ret_matrix.fillna(0.0)

    gross_returns = (weights * ret_matrix).sum(axis = 1)

    gross_exposure = weights.abs().sum(axis = 1)
    net_exposure = weights.sum(axis = 1)
    position_count = (weights.abs() > EPS).sum(axis = 1)

    max_observed_gross = float(gross_exposure.max()) if len(gross_exposure) else 0.0
    if max_observed_gross > config.max_gross + GROSS_EXPOSURE_TOL:
        raise AssertionError(
            f"Gross exposure exceeded max_gross: observed {max_observed_gross:.12f}, "
            f"limit {config.max_gross:.12f}. Check portfolio normalization and "
            "volatility targeting before trusting the run."
        )

    # Realized turnover: the first row counts as entering from flat
    weight_changes = weights.diff()
    weight_changes.iloc[0] = weights.iloc[0]
    turnover = weight_changes.abs().sum(axis = 1)

    costs = turnover * cost_rate 
    net_returns = gross_returns - costs 

    # Only bars where capital is actually deployed count as observations
    active = weights.abs().sum(axis = 1) > EPS
    gross_returns, net_returns = gross_returns[active], net_returns[active]
    turnover, costs = turnover[active], costs[active]
    gross_exposure, net_exposure = gross_exposure[active], net_exposure[active]
    position_count = position_count[active]

    exposure = {
        "mean_gross": float(gross_exposure.mean()) if len(gross_exposure) else 0.0,
        "median_gross": float(gross_exposure.median()) if len(gross_exposure) else 0.0,
        "p95_gross": float(gross_exposure.quantile(0.95)) if len(gross_exposure) else 0.0,
        "max_gross": float(gross_exposure.max()) if len(gross_exposure) else 0.0,
        "mean_net": float(net_exposure.mean()) if len(net_exposure) else 0.0,
        "median_net": float(net_exposure.median()) if len(net_exposure) else 0.0,
        "p95_abs_net": float(net_exposure.abs().quantile(0.95)) if len(net_exposure) else 0.0,
        "max_abs_net": float(net_exposure.abs().max()) if len(net_exposure) else 0.0,
        "mean_position_count": float(position_count.mean()) if len(position_count) else 0.0,
        "max_position_count": int(position_count.max()) if len(position_count) else 0,
    }

    return {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "weights": weights.loc[active],
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "position_count": position_count,
        "exposure": exposure,
        "turnover": turnover,
        "costs": costs,
        "n_rebalances": int(len(schedule)),
        "total_turnover": float(turnover.sum()),
        "mean_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "total_cost": float(costs.sum()),
        "cost_rate": float(cost_rate),
        "rebalance_frequency": config.rebalance_frequency,
    }

### Baselines ###

def buy_and_hold(
    tradeable_returns: pd.DataFrame, symbol: str, calendar: pd.DatetimeIndex
) -> pd.Series:
    """ Buy and hold return series for a symbol present in the tradeable universe"""
    series = tradeable_returns[tradeable_returns["symbol"] == symbol]
    return (
        series.set_index("timestamp")["tradeable_return"]
        .reindex(calendar)
        .fillna(0.0)
        .rename(f"{symbol}_buy_hold")
    )

def benchmark_returns(market: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """ Benchmark buy and hold, taken from the market OHLCV frame
    
    The benchmark is loaded outside the universe filters, so it is not 
    guaranteed to appear in the tradeable-symbol returns -- with a small 
    ``--max-symbols`` it usually does not. Deriving it from the market frame 
    means the SPY comparison and the alpha/beta regression are always available,
    rather than silently degrading to an all-zero series
    """
    close = market["close"].reindex(calendar).ffill()
    return close.pct_change().fillna(0.0).rename("spy")

def equal_weight_baseline(
    tradeable_returns: pd.DataFrame,
    symbols: Sequence[str],
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Equal-weight, always-long basket of ``symbols``."""
    matrix = tradeable_returns[tradeable_returns["symbol"].isin(list(symbols))].pivot_table(
        index = "timestamp", columns = "symbol", values = "tradeable_return", aggfunc = "last"
    ).reindex(calendar)
    return matrix.mean(axis = 1).fillna(0.0).rename("equal_weight")

def tsmom_baseline(
    prices: pd.DataFrame,
    tradeable_returns: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    lookback_days: int = 21,
    max_positions: int = 10,
    rebalance_frequency: str = "weekly",
) -> pd.Series:
    """ Naive time-series momentum: long/short the sign of trailing returns"""

    lookback = bars(lookback_days)

    close = prices.pivot_table(
        index = "timestamp", columns = "symbol", values = "close", aggfunc = "last"
    ).reindex(calendar).ffill() 

    # Trailing return, shifted so the signal uses only past information
    trailing = close.pct_change(lookback).shift(1)

    schedule = set(rebalance_timestamps(calendar, rebalance_frequency))
    rows = {}
    for ts in calendar:
        if ts not in schedule:
            continue 
        cross = trailing.loc[[ts]].iloc[0].dropna()
        if cross.empty:
            continue
        signal = pd.Series(np.sign(cross.to_numpy(dtype = float)), index = cross.index)
        picked = signal.reindex(cross.abs().sort_values(ascending = False).index)
        picked = picked.iloc[:max_positions]
        gross = picked.abs().sum()
        rows[ts] = picked / gross if gross > EPS else picked * 0.0

    if not rows:
        return pd.Series(0.0, index = calendar, name = "tsmom")

    weights = pd.DataFrame(rows).T.reindex(columns = close.columns)
    weights = weights.reindex(calendar).ffill().fillna(0.0)

    ret_matrix = tradeable_returns.pivot_table(
        index = "timestamp", columns = "symbol", values = "tradeable_return", aggfunc = "last"
    ).reindex(index = calendar, columns = weights.columns).fillna(0.0)

    return (weights * ret_matrix).sum(axis = 1).rename("tsmom")

def compare_to_baseline(
    strategy_returns: pd.Series,
    baselines: Dict[str, pd.Series],
    periods_per_year: float = PERIODS_PER_YEAR,
    benchmark_key: str = "spy",
) -> Dict: 
    """ Summarize the strategy against each baseline, plus alpha/beta vs SPY"""
    report: Dict[str, Dict] = {
        "strategy": Metrics.performance_summary(strategy_returns, periods_per_year)
    }
    for name, series in baselines.items():
        aligned = series.reindex(strategy_returns.index).fillna(0.0)
        report[name] = Metrics.performance_summary(aligned, periods_per_year)

    if benchmark_key in baselines:
        bench = baselines[benchmark_key].reindex(strategy_returns.index).fillna(0.0)
        report["alpha_beta_vs_" + benchmark_key] = Metrics.alpha_beta(
            strategy_returns, bench, periods_per_year
        )

    return report 
