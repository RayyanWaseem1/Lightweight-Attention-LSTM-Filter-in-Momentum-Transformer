from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np 
import pandas as pd 

from Models.config import BacktestConfig, PERIODS_PER_YEAR
from Utils import Metrics
from Utils.Portfolio import run_portfolio_backtest 

@dataclass
class WalkForwardWindow:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp 
    test_start: pd.Timestamp 
    test_end: pd.Timestamp
    metrics: Dict = field(default_factory = dict)
    returns: pd.Series = field(default_factory = lambda: pd.Series(dtype = float))

class WalkForwardAnalyzer:
    """ Rolling train/test evaluation over the trading calendar"""

    def __init__(
        self,
        train_bars: int, 
        test_bars: int,
        step_bars: int, 
        backtest_config: Optional[BacktestConfig] = None,
        periods_per_year: float = PERIODS_PER_YEAR,
    ):
        if min(train_bars, test_bars, step_bars) <= 0:
            raise ValueError("train/test/step bars must all be positive")

        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars 
        self.backtest_config = backtest_config or BacktestConfig()
        self.periods_per_year = periods_per_year

    def plan_windows(self, calendar: pd.DatetimeIndex) -> List[Dict]:
        """ Enumerate the (train, test) index ranges without running anything"""
        calendar = pd.DatetimeIndex(sorted(calendar))
        n = len(calendar)
        windows = []

        start = 0
        while start + self.train_bars + self.test_bars <= n:
            train_end = start + self.train_bars
            test_end = min(train_end + self.test_bars, n)
            windows.append(
                {
                    "index": len(windows),
                    "train_slice": (start, train_end),
                    "test_slice": (train_end, test_end),
                    "train_start": calendar[start],
                    "train_end": calendar[train_end - 1],
                    "test_start": calendar[train_end],
                    "test_end": calendar[test_end - 1],
                }
            )
            start += self.step_bars
        return windows 

    def run(
        self,
        features_df: pd.DataFrame,
        tradeable_returns: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        train_fn: Callable[[pd.DataFrame, pd.DataFrame], object],
        predict_fn: Callable[[object, pd.DataFrame], pd.DataFrame],
        verbose: bool = True,
    ) -> Dict:
        """ Run the full analysis
        
        ``train_fn(train_features, val_features) -> model``
        ``predict_fn(model, test_features) -> DataFrame[timestamp, symbol, prediction]``
        """
        calendar = pd.DatetimeIndex(sorted(calendar))
        plans = self.plan_windows(calendar)

        if not plans:
            raise ValueError(
                f"No walk-forward windows fit: calendar has {len(calendar)} bars but "
                f"train({self.train_bars}) + test({self.test_bars}) = "
                f"{self.train_bars + self.test_bars} are required."
            )

        if verbose:
            print(f"\n [Walk-forward] {len(plans)} windows | "
            f"train {self.train_bars} bars, test {self.test_bars} bars, "
            f"step {self.step_bars} bars")

        results: List[WalkForwardWindow] = []

        for plan in plans:
            train_mask = (
                (features_df["timestamp"] >= plan["train_start"])
                & (features_df["timestamp"] <= plan["train_end"])
            )
            test_mask = (
                (features_df["timestamp"] >= plan["test_start"])
                & (features_df["timestamp"] <= plan["test_end"])
            )

            train_features = features_df[train_mask]
            test_features = features_df[test_mask]

            if train_features.empty or test_features.empty:
                continue 

            # Hold out the tail of the training window for model selection, so 
            # nothing in the test window influences the fitted model
            split = train_features["timestamp"].quantile(0.85)
            fit_features = train_features[train_features["timestamp"] <= split]
            val_features = train_features[train_features["timestamp"] > split]

            if verbose:
                print(f" Window {plan['index'] + 1}/{len(plans)}: "
                      f"train {plan['train_start'].date()} .. {plan['train_end'].date()} | "
                      f"test {plan['test_start'].date()} .. {plan['test_end'].date()}")

            model = train_fn(fit_features, val_features)
            predictions = predict_fn(model, test_features)

            if predictions.empty:
                continue

            test_calendar = calendar[
                (calendar >= plan["test_strat"]) & (calendar <= plan["test_end"])
            ]
            backtest = run_portfolio_backtest(
                predictions, tradeable_returns, test_calendar, self.backtest_config
            )

            net = backtest["net_returns"]
            if net.empty:
                continue 

            window = WalkForwardWindow(
                index = plan["index"],
                train_start = plan["train_start"],
                train_end = plan["train_end"],
                test_start = plan["test_start"],
                test_end = plan["test_end"],
                metrics = Metrics.performance_summary(net, self.periods_per_year),
                returns = net,
            )
            results.append(window)

            if verbose:
                print(f" Sharpe {window.metrics['sharpe_ratio']:.6.3f} | "
                      f"return {window.metrics['total_return']:7.2%} | "
                      f"turnover {backtest['mean_turnover']:.3f}")
                
        if not results:
            raise RuntimeError("Walk-foward produced no usable windows")

        return self._aggregate(results)

    def _aggregate(self, windows: List[WalkForwardWindow]) -> Dict:
        """ Pool the out-of-sample returns and summarize the window distribution"""
        # Chain the windows into one continous out-of-sample series. Windows
        # can overlap in time when step < test; keep the first occurrence.    
        pooled = pd.concat([w.returns for w in windows])
        pooled = pooled[~pooled.index.duplicated(keep = "first")].sort_index()

        sharpes = np.array([w.metrics["sharpe_ratio"] for w in windows], dtype = float)
        returns = np.array([w.metrics["total_return"] for w in windows], dtype = float)
        drawdowns = np.array([w.metrics["max_drawdown"] for w in windows], dtype = float)

        n = len(sharpes)
        se = sharpes.std(ddof = 1) / np.sqrt(n) if n > 1 else 0.0 

        return {
            "n_windows": n,
            # The Sharpe of the pooled out-of-sample series -- the honest headline
            # mean_sharpe across windows is reported alongside, not instead of it
            "pooled": Metrics.performance_summary(pooled, self.periods_per_year),
            "pooled_returns": pooled,
            "window_distribution": {
                "mean_sharpe": float(sharpes.mean()),
                "std_sharpe": float(sharpes.std(ddof = 1)) if n > 1 else 0.0,
                "min_sharpe": float(sharpes.min()),
                "max_sharpe": float(sharpes.max()),
                "median_sharpe": float(np.median(sharpes)),
                # Normal-approximation CI on the mean across windows
                "sharpe_ci_low": float(sharpes.mean() - 1.96 * se),
                "sharpe_ci_high": float(sharpes.mean() + 1.96 * se),
                "positive_windows": int((sharpes > 0).sum()),
                "mean_total_return": float(returns.mean()),
                "worst_drawdown": float(drawdowns.min()),
            },
            "windows": [
                {
                    "index": w.index,
                    "train_start": str(w.train_start),
                    "train_end": str(w.train_end),
                    "test_start": str(w.test_start),
                    "test_end": str(w.test_end),
                    "sharpe_ratio": w.metrics["sharpe_ratio"],
                    "total_return": w.metrics["total_return"],
                    "max_drawdown": w.metrics["max_drawdown"],
                    "n_periods": w.metrics["n_periods"],
                }
                for w in windows
            ],
        }

def print_walk_forward_results(results: Dict) -> None:
    dist = results["window_distribution"]
    pooled = results["pooled"]

    print("\n" + "=" * 78)
    print("WALK-FORWARD RESULTS")
    print("=" * 78)
    print(f"Windows: {results['n_windows']}")
    print(f"Positive windows: {dist['positive_windows']}/{results['n_windows']}")
    print("\n Pooled out-of-sample (the headline):")
    print(f" Sharpe (per-bar ann.) {pooled['sharpe_ratio']:.8.3f}")
    if "daily_sharpe" in pooled:
        print(f" Sharpe (daily agg.) {pooled['daily_sharpe']:8.3f}")
    print(f" Total return {pooled['total_return']:8.2%}")
    print(f" Max drawdown {pooled['max_drawdown']:8.2%}")
    print("\n Across-window distribution:")
    print(f" Mean Sharpe {dist['mean_sharpe']:8.3f}")
    print(f" Std Sharpe {dist['std_sharpe']:8.3f}")
    print(f" 95% CI on mean [{dist['sharpe_ci_low']:.3f}, {dist['sharpe_ci_high']:.3f}]")
    print(f" Min / Max {dist['min_sharpe']:.3f} / {dist['max_sharpe']:.3f}")
    print("=" * 78)
