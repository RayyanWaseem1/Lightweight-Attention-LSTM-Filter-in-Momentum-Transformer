""" Performance and statistical metrics

This is the **only** place in the repo where a Sharpe, Sortino, Calmar,
drawdown or information coefficient is defined. 

Conventions:
* Per-bar series are annualized with ``PERIODS_PER_YEAR`` from 
    ``Models.config`` (7 bars/day x 252 days = 1764), or with an empirically
    derived value when one is supplied.
*``annualized_return`` compounds over the true horizon. It is never
    ``mean * periods_per_year``
* Drawdown is reported as a negative fraction and is **not** floored
* Sortino uses downside deviation about the target,
    ``sqrt(mean(min(r - target, 0) **2)) ``, computed over the **full** sample --
    not the standard deviation of the losing subset about its own mean.
* The headline figure reported by the backtest is ``daily_sharpe``: PnL is 
    aggregated to daily and annualized with sqrt(252)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np 
import pandas as pd 
from scipy import stats 

from Models.config import PERIODS_PER_YEAR, TRADING_DAYS_PER_YEAR

EPS = 1e-12

__all__ = [
    "to_array",
    "sharpe_ratio",
    "daily_sharpe",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "annualized_return",
    "annualized_volatility",
    "longest_streak",
    "win_rate",
    "profit_factor",
    "turnover",
    "information_coefficient",
    "cross_sectional_ic",
    "block_bootstrap_ic",
    "deflated_sharpe_ratio",
    "alpha_beta",
    "performance_summary",
    "directional_accuracy",
]

def to_array(returns) -> np.ndarray:
    """Coerce a Series/list/array of returns to a finite 1-D float array"""
    if isinstance(returns, pd.Series):
        values = returns.to_numpy(dtype = float)
    else:
        values = np.asarray(returns, dtype = float)
    values = values.reshape(-1)
    return values[np.isfinite(values)]

### Risk-adjusted return ratios ###

def sharpe_ratio(returns, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe of a per-bar return series"""
    r = to_array(returns)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof = 1)
    if sd < EPS:
        return 0.0 
    return float(r.mean() / sd * np.sqrt(periods_per_year))

def daily_sharpe(returns: pd.Series) -> float:
    """Sharpe of the daily-aggregated PnL, annualized with sqrt(252)
    
    ``returns`` must carry a DatetimeIndex. Intraday bars are compounded
    within each sesion before the ratio is taken, which removes any 
    dependence on the assumed number of bars per day 
    """

    if not isinstance(returns, pd.Series):
        raise TypeError("daily_sharpe requires a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("daily_sharpe requires a DatetimeIndex")

    daily = (1.0 + returns).groupby(returns.index.date).prod() - 1.0
    return sharpe_ratio(daily, periods_per_year = TRADING_DAYS_PER_YEAR)

def sortino_ratio(
    returns,
    periods_per_year: float = PERIODS_PER_YEAR,
    target: float = 0.0,
) -> float:
    """Annualized Sortino ratio 
    
    Downside deviation is the RMS of ``min(r - target, 0)`` over the **full**
    sample, and the ratio is annualized exactly once
    """
    r = to_array(returns)
    if r.size < 2:
        return 0.0
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd < EPS:
        return 0.0
    return float((r.mean() - target) / dd * np.sqrt(periods_per_year))

def max_drawdown(returns) -> float:
    """ Maximum drawdown of the compounded equity curve, as a negative fraction
    
    Not floored: a strategy that loses more than 100% reports that it did
    """
    r = to_array(returns)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.where(
        np.abs(running_max) < EPS, np.nan, running_max
    )
    if np.all(np.isnan(drawdown)):
        return 0.0
    return float(np.nanmin(drawdown))

def annualized_return(returns, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    """ True CAGR, compounded over the realized horizon"""
    r = to_array(returns)
    if r.size == 0:
        return 0.0
    total_growth = float(np.prod(1.0 + r))
    n_years = r.size / periods_per_year
    if n_years <= 0:
        return 0.0
    if total_growth <= 0:
        # capital wiped out: CAGR is undefined, report -100%
        return -1.0
    return float(total_growth ** (1.0 / n_years) - 1.0)

def annualized_volatility(
        returns, periods_per_year: float = PERIODS_PER_YEAR
) -> float:
    r = to_array(returns)
    if r.size < 2:
        return 0.0
    return float(r.std(ddof = 1) * np.sqrt(periods_per_year))

def calmar_ratio(returns, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    dd = abs(max_drawdown(returns))
    if dd < EPS:
        return 0.0
    return float(annualized_return(returns, periods_per_year) / dd)

### Trade statistics ###

def longest_streak(arr: Any, value = True) -> int:
    """ Longest run of consecutive elements equal to ``value``,
    """
    values = np.asarray(arr)
    if values.size == 0:
        return 0

    best = current = 0
    for item in values:
        if item == value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)

def win_rate(returns) -> float:
    r = to_array(returns)
    if r.size == 0:
        return 0.0
    return float(np.mean(r>0))

def profit_factor(returns) -> float:
    r = to_array(returns)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses < EPS:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)

def turnover(weights: pd.DataFrame) -> pd.Series:
    """ Per-rebalance turnover, ``sum_i |w_{i,t} - w_{i, t-1}|``
    
    ``weights`` is [time x symbol]; the first row counts as turnover from flat
    """
    if weights.empty:
        return pd.Series(dtype = float)

    filled = weights.fillna(0.0)
    changes = filled.diff()
    # diff() leaves the first row NaN, and .sum(axis = 1) would silently treat
    # that as 0.0 -- reporting no turnover for the trade that establishes the 
    # book. Set it explicitly to the initial gross exposure
    changes.iloc[0] = filled.iloc[0]
    return changes.abs().sum(axis = 1)

### Predictive - power statistics ###

def directional_accuracy(predictions, targets) -> float:
    """ Fraction of bars where sign(prediction) == sign(target),
    
    Computed on **raw** predictions. Rescaling predictions to the target's moments --
    shifts them by ``target_mean - pred_mean`` and therefore changes their sign, so the
    resulting number is not the model's directional accuracy at all
    """
    p = np.asarray(predictions, dtype = float).reshape(-1)
    t = np.asarray(targets, dtype = float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(t)
    if mask.sum() == 0:
        return 0.0 
    return float(np.mean(np.sign(p[mask]) == np.sign(t[mask])))

def information_coefficient(predictions, targets) -> tuple:
    """ Pooled Spearman rank IC and its nominal p-value.
    
    The nominal p-value assumes i.i.d observations and is badly optimistic 
    for overlapping windows across correlated names -- use ``block_bootstrap_ic``
    or ``cross_sectional_ic`` for an honest standard error
    """
    p = np.asarray(predictions, dtype = float).reshape(-1)
    t = np.asarray(targets, dtype = float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(t)
    if mask.sum() < 3:
        return 0.0, 1.0
    ic_raw, pvalue_raw = stats.spearmanr(p[mask], t[mask])
    ic = float(np.asarray(ic_raw).item())
    pvalue = float(np.asarray(pvalue_raw).item())
    if not np.isfinite(ic):
        return 0.0, 1.0
    return ic, pvalue

def cross_sectional_ic(df: pd.DataFrame) -> Dict[str, float]:
    """ Mean cross-sectional IC per timestamp with a Newey-West style t-stat.
    
    This is the standard construction: rank-correlate predictions against 
    realized returns *within* each timestamp, then test whether the resulting
    time series of ICs has a mean different from zero. It sidesteps the 
    overlapping window problem that makes the pooled p-value meaningless
    
    ``df`` needs columns `` timestamp``, ``prediction`` and ``target``.
    """
    required = {"timestamp", "prediction", "target"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"cross_sectional_ic missing columns: {sorted(missing)}")

    per_ts = []
    for ts, group in df.groupby("timestamp"):
        if len(group) < 3:
            continue 
        # A timestamp where every prediction (or target) ties has an undefined
        # rank correlation. Skp it rather than emitting a warning per bar 
        if group["prediction"].nunique() < 2 or group["target"].nunique() < 2:
            continue 
        ic_raw, _ = stats.spearmanr(group["prediction"], group["target"])
        ic = float(np.asarray(ic_raw).item())
        if np.isfinite(ic):
            per_ts.append(ic)

    if len(per_ts) < 2:
        return {"mean_ic": 0.0, "ic_std": 0.0, "ic_tstat": 0.0, "ic_pvalue": 1.0,
                "n_timestamps": len(per_ts)}

    ics = np.asarray(per_ts, dtype = float)
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof = 1))
    tstat = mean_ic / (std_ic / np.sqrt(len(ics)) + EPS)
    pvalue = float(2 * (1 - stats.t.cdf(abs(tstat), df = len(ics) - 1)))

    return {
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "ic_tstat": float(tstat),
        "ic_pvalue": pvalue,
        "n_timestamps": int(len(ics)),
    }

def block_bootstrap_ic(
        predictions, 
        targets,
        block_size: int = 252,
        n_boot: int = 1000,
        seed: int = 0,
) -> Dict[str, float]:
    """ Block-bootstrap standard error and CI for the pooled IC.
    
    Moving-block bootstrap preserves the serial dependence induced by
    overlapping feature windows, which the analytic Spearman p-value ignores
    """

    p = np.asarray(predictions, dtype = float).reshape(-1)
    t = np.asarray(targets, dtype = float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]

    n = p.size
    if n < block_size * 2:
        block_size = max(2, n // 10)
    if n < 10:
        return {"ic": 0.0, "ic_se": 0.0, "ic_ci_low": 0.0, "ic_ci_high": 0.0}

    point, _ = information_coefficient(p, t)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size 

    samples = np.empty(n_boot, dtype = float)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size = n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        ic, _ = information_coefficient(p[idx], t[idx])
        samples[b] = ic 

    return {
        "ic": float(point),
        "ic_se": float(samples.std(ddof = 1)),
        "ic_ci_low": float(np.percentile(samples, 2.5)),
        "ic_ci_high": float(np.percentile(samples, 97.5)),
    }

def deflated_sharpe_ratio(
    returns,
    n_trials: int,
    periods_per_year: float = PERIODS_PER_YEAR,
    benchmark_sharpe: float = 0.0,
) -> Dict[str, float]:
    """ Deflated Sharpe Ratio (Baily & Lopez de Prado, 2014).
    
    Adjusts the observed Sharpe for (a) the number of configurations tried and 
    (b) the non-normality of the return distribution. ``n_trials`` should be 
    the honest count of everything searched: tuning trials x architectures x seeds.
    
    Returns the probability that the true Sharpe exceeds ``benchmark_sharpe``
    """

    r = to_array(returns)
    n = r.size
    if n < 4 or n_trials < 1:
        return {"sharpe": 0.0, "expected_max_sharpe": 0.0, "deflated_sharpe": 0.0,
                 "n_trials": int(n_trials)}

    # work in per-bar (non-annualized) units
    sr = r.mean() / (r.std(ddof = 1) + EPS)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher = False))

    # Expected maximum Sharpe under the null of zero true skill across trials,
    # expressed as a normal quantile. It becomes a Sharpe threshold only after
    # multiplying by the estimator standard error below.
    euler = 0.5772156649015329
    if n_trials > 1:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max_z = (1 - euler) * z1 + euler * z2
    else:
        expected_max_z = 0.0

    # Variance of the Sharpe estimator under non-normal returns 
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        denom = EPS 
    sr_std = np.sqrt(denom / (n-1))

    threshold = benchmark_sharpe / np.sqrt(periods_per_year) + expected_max_z * sr_std
    dsr = float(stats.norm.cdf((sr - threshold) / (sr_std + EPS)))

    return {
        "sharpe": float(sr * np.sqrt(periods_per_year)),
        "expected_max_sharpe": float(expected_max_z * sr_std * np.sqrt(periods_per_year)),
        "deflated_sharpe": dsr,
        "n_trials": int(n_trials),
        "skew": skew,
        "excess_kurtosis": kurt - 3.0,
        "sharpe_standard_error": float(sr_std * np.sqrt(periods_per_year)),
    }

def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """ OLS regression of strategy on benchmark returns.
    
    Reports annualized alpha with a t-stat, beta, and the correlation -- the 
    "what's your alpha net of beta?" """

    joined = pd.concat(
            [pd.Series(strategy_returns).rename("s"),
             pd.Series(benchmark_returns).rename("b")],
            axis=1,
    ).dropna()

    degenerate = {"alpha_annual": 0.0, "alpha_tstat": 0.0, "beta": 0.0,
                  "correlation": 0.0, "r_squared": 0.0, "n_obs": len(joined)}
    if len(joined) < 3:
        return degenerate 

    # A constant benchmark carries no information; scipy raises rather than 
    # returning NaN, so guard explicitly. This happens when the benchmark is 
    # absent from the data and its series is all zeros 
    if joined["b"].std(ddof = 0) < EPS:
        degenerate["note"] = "benchmark series is constnat; regression skipped"
        return degenerate 

    b = joined["b"].to_numpy(dtype = float)
    s = joined["s"].to_numpy(dtype = float)
    b_mean = float(b.mean())
    s_mean = float(s.mean())
    b_centered = b - b_mean
    s_centered = s - s_mean
    b_var = float(np.sum(b_centered ** 2))
    slope = float(np.sum(b_centered * s_centered) / b_var)
    intercept = float(s_mean - slope * b_mean)
    rvalue = float(np.corrcoef(b, s)[0, 1])
    n = len(joined)

    # Standard error of the intercept
    resid = s - (intercept + slope * b)
    dof = n - 2
    s_err = np.sqrt((resid ** 2).sum() / dof) if dof > 0 else np.nan
    if b_var > EPS and np.isfinite(s_err):
        se_intercept = s_err * np.sqrt(1.0 / n + b_mean ** 2 / b_var)
    else:
        se_intercept = np.nan

    tstat = intercept / se_intercept if se_intercept and se_intercept > 0 else 0.0

    return {
        "alpha_annual": float(intercept * periods_per_year),
        "alpha_tstat": float(tstat) if np.isfinite(tstat) else 0.0,
        "beta": slope,
        "correlation": rvalue,
        "r_squared": float(rvalue**2),
        "n_obs": int(n),
    }

### Aggregate Summary ###

def performance_summary(
    returns: pd.Series,
    periods_per_year: float = PERIODS_PER_YEAR,
    turnover_series: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """ Full metric set for a per-bar return series.
    
    ``returns`` should carry a DatetimeIndex so the daily-aggregated headline
    Sharpe can be computed
    """
    r = to_array(returns)
    total_return = float(np.prod(1.0 + r) - 1.0) if r.size else 0.0

    summary = {
        "total_return": total_return,
        "annualized_return": annualized_return(r, periods_per_year),
        "volatility": annualized_volatility(r, periods_per_year),
        "sharpe_ratio": sharpe_ratio(r, periods_per_year),
        "sortino_ratio": sortino_ratio(r, periods_per_year),
        "calmar_ratio": calmar_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "win_rate": win_rate(r),
        "profit_factor": profit_factor(r),
        "win_streak": longest_streak(r > 0, True),
        "loss_streak": longest_streak(r < 0, True),
        "n_periods": int(r.size),
        "periods_per_year": float(periods_per_year),
    }

    if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        summary["daily_sharpe"] = daily_sharpe(returns)
        summary["n_days"] = int(len(set(returns.index.date)))

    if turnover_series is not None and len(turnover_series):
        summary["mean_turnover"] = float(np.mean(to_array(turnover_series)))
        summary["total_turnover"] = float(np.sum(to_array(turnover_series)))

    return summary 
