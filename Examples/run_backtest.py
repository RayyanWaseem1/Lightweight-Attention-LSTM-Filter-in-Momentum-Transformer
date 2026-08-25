"""Single backtest harness for all three model variants.

This is the canonical end-to-end entry point for vanilla, attention, and
ensemble runs.

Usage:
    python Examples/run_backtest.py --model ensemble
    python Examples/run_backtest.py --model vanilla --max-symbols 8 --epochs 2
    python Examples/run_backtest.py --model ensemble --walk-forward 
    
Every run writes its resolved configuration next to its outputs, so any result
can be traced to the parameters that produced it 
"""

from __future__ import annotations 

import argparse
import json 
import random 
import sys
import time 
from dataclasses import asdict 
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, cast 

import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import DataLoader 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.config import ( # noqa: E402
    PERIODS_PER_YEAR,
    Config,
    derive_periods_per_year,
    get_production_config,
    )
from Models.Ensemble_model import EnsembleMomentumTransformer, get_ensemble_model # noqa: E402
from Models.Momentum_transformer import (   # noqa: E402
    count_parameters,
    get_momentum_transformer,
)
from Utils import Metrics # noqa: E402
from Utils.Backtesting import WalkForwardAnalyzer, print_walk_forward_results # noqa: E402
from Utils.Feature_engineering import ( #noqa: E402
    FeatureConfig,
    FeatureNormalizer,
    check_collinearity,
    check_feature_scales,
    create_features_from_ohlcv,
)
from Utils.Losses import SharpeRatioLoss # noqa: E402
from Utils.Market_data import prepare_market_data, tradeable_returns # noqa: E402
from Utils.Portfolio import ( # noqa: E402
    benchmark_returns,
    compare_to_baseline,
    equal_weight_baseline,
    run_portfolio_backtest,
    tsmom_baseline,
)
from Utils.Regime_detector import StatisticalRegimeDetector, label_regimes # noqa: E402
from Utils.Training import EarlyStopping, train_model # noqa: E402
from data.Dataset import MultiAssetWindowDataset # noqa: E402

METADATA_COLUMNS = ["symbol", "timestamp", "target", "ex_ante_vol"]

### Feature Frame ###

def build_feature_frame(
    prices: pd.DataFrame,
    market: pd.DataFrame,
    symbols: List[str],
    returns_df: pd.DataFrame,
    feature_config: FeatureConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """ Per symbol features joined to the target and an ex-ante volatility"""

    if verbose: 
        print("\n[Features] Engineering stock-agnostic features ...")

    target_lookup = returns_df.set_index(["symbol", "timestamp"])["tradeable_return"]
    frames = []

    for symbol in symbols:
        symbol_prices = prices[prices["symbol"] == symbol].set_index("timestamp")
        if len(symbol_prices) <= feature_config.max_lookback + 10:
            continue 

        features = create_features_from_ohlcv(
            symbol_prices[["open", "high", "low", "close", "volume"]],
            market_df = market,
            symbol = symbol,
            config = feature_config,
        )
        if features.empty:
            continue 

        features = features.reset_index()
        features["symbol"] = symbol 

        keys = list(zip(features["symbol"], features["timestamp"]))
        features["target"] = target_lookup.reindex(keys).to_numpy()

        # Ex-ante annualized volatility for the volatility-targeting layer
        # Uses only the trailing-volatility feature, which is already causal 
        vol_col = f"volatility_{feature_config.window(3)}"
        if vol_col in features.columns:
            features["ex_ante_vol"] = (
                features[vol_col].abs() * np.sqrt(PERIODS_PER_YEAR)
            )
        else:
            features["ex_ante_vol"] = 0.0

        frames.append(features.dropna(subset = ["target"]))

    if not frames:
        raise RuntimeError("No symbol produced a usable feature frame")

    combined = pd.concat(frames, ignore_index = True)
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop = True)

    feature_cols = [c for c in combined.columns if c not in METADATA_COLUMNS]
    if verbose:
        print(f" {len(combined):,} rows | {combined['symbol'].nunique()} symbols "
              f"| {len(feature_cols)} features")

    return combined 

def run_feature_quality_checks(features_df: pd.DataFrame, verbose: bool = True) -> Dict:
    """Collinearity and scale checks -- guards against 3.1 and 3.2 regressing"""
    feature_cols = [c for c in features_df.columns if c not in METADATA_COLUMNS]
    sample = features_df[feature_cols].sample(
        min(20_000, len(features_df)), random_state = 0
    )

    collinear = check_collinearity(sample, threshold = 0.99)
    scales = check_feature_scales(sample)

    if verbose:
        print("\n[Features] Quality checks")
        if collinear:
            print(f" WARNING: {len(collinear)} near-duplicate feature pairs "
                  f"(|rho| > 0.99):")
            for a, b, rho in collinear[:5]:
                print(f" {a} ~ {b}: {rho:.4f}")
        else:
            print(" No feature pair with |rho| > 0.99")
        print(f" Std spread across features: {scales['ratio']:.1f}x "
              f"({scales.get('smallest')} .. {scales.get('largest')})")

    return {"collinear_pairs": collinear, "scales": scales}

### Model ###

def build_model(model_type: str, config: Config, feature_names: List[str]):
    if model_type == "vanilla":
        return get_momentum_transformer(config.model, "simple", config.training)
    if model_type == "attention":
        config.model.use_lstm_attention = True 
        return get_momentum_transformer(config.model, "enhanced", config.training)
    if model_type == "ensemble":
        return get_ensemble_model(config, feature_names)
    raise ValueError(f"Unknown model type: {model_type!r}")

def make_validation_fn(
    val_dataset: MultiAssetWindowDataset,
    val_returns: pd.DataFrame,
    val_calendar: pd.DatetimeIndex,
    config: Config,
    batch_size: int,
    periods_per_year: float,
):
    """ Validation score computed the same way the backtest is. 
    
    The previous validation metric scored a long-only top-quintile rule on the 
    pooled validation set (with a comment saying "top 50%" over code using the 
    80th percentile) while training optimized continous-position Sharpe and 
    the backtest evaluated a 10-name long/short book. three different 
    objectives, so "best val Sharpe" selected a checkpoint for a strategy that 
    was never run
    """

    index_frame = val_dataset.index_frame

    def validation_fn(model) -> float:
        predictions = predict_dataset(model, val_dataset, index_frame, batch_size)
        if predictions.empty:
            return float("-inf")
        result = run_portfolio_backtest(
            predictions, val_returns, val_calendar, config.backtest
        )
        net = result["net_returns"]
        if net.empty:
            return float("-inf")
        return Metrics.sharpe_ratio(net, periods_per_year)

    return validation_fn 

@torch.no_grad()
def predict_dataset(
    model,
    dataset: MultiAssetWindowDataset,
    index_frame: pd.DataFrame,
    batch_size: int = 256,
) -> pd.DataFrame: 
    """ Predict every window, returning ``[timestamp, symbol, prediction]``
    
    The full prediction path is kept. The previous implementation generated a 
    prediction at every bar and then discarded all but the last one per symbol
    """

    model.eval()
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    outputs = []
    for x, _, vol in loader:
        positions, _ = model(x, ex_ante_vol = vol)
        outputs.append(positions.reshape(-1).cpu().numpy())

    if not outputs:
        return pd.DataFrame(columns = ["timestamp", "symbol", "prediction"])

    predictions = np.concatenate(outputs)
    frame = index_frame.copy()
    frame["prediction"] = predictions[: len(frame)]
    return frame 

def scale_to_volatility(
    returns: pd.Series,
    target_volatility: float,
    periods_per_year: float,
) -> pd.Series:
    """Scale a return series to a target annualized volatility."""
    vol = Metrics.annualized_volatility(returns, periods_per_year)
    if vol <= 0 or target_volatility <= 0:
        return pd.Series(0.0, index=returns.index, name=returns.name)
    return (returns * (target_volatility / vol)).rename(returns.name)

def train_variant(
    model_type: str,
    config: Config,
    train_dataset: MultiAssetWindowDataset,
    val_dataset: MultiAssetWindowDataset,
    feature_names: List[str],
    validation_fn,
    verbose: bool = True,
):
    """ Train one variant.
    
    For this ensemble this is a genuine two-stage schedule -- each arm trained
    independently, then frozen, then the blend network fitted
    """
    criterion = SharpeRatioLoss()
    batch_size = config.training.batch_size

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    def fit(model, epochs, params = None, tag = ""):
        optimizer = torch.optim.AdamW(
            params if params is not None else model.parameters(),
            lr = config.training.learning_rate,
            weight_decay = config.training.weight_decay,
        )
        early = EarlyStopping(
            patience = config.training.early_stopping_patience,
            min_delta = config.training.early_stopping_min_delta,
            mode = "max",
        )
        if verbose and tag:
            print(f"\n -- {tag} --")
        return train_model(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            criterion = criterion,
            optimizer = optimizer,
            num_epochs = epochs,
            early_stopping = early,
            gradient_clip_norm = config.training.gradient_clip_norm,
            accumulation_steps= config.training.sharpe_accumulation_steps,
            validation_fn = validation_fn,
            verbose = verbose,
        )

    if model_type != "ensemble" or not config.ensemble.pretrain_submodels:
        model = build_model(model_type, config, feature_names)
        fit_regime_thresholds(model, train_dataset, feature_names)
        if verbose:
            print(f" Parameters: {count_parameters(model):,}")
        fit(model, config.training.num_epochs)
        return model 

    model = cast(EnsembleMomentumTransformer, build_model("ensemble", config, feature_names))
    fit_regime_thresholds(model, train_dataset, feature_names)
    if verbose:
        print(f" Parameters: {count_parameters(model):,} "
              f" (vanilla {count_parameters(model.vanilla_model):,} | "
              f"attention {count_parameters(model.attention_model):,})")

    # Stage 1: each arm independently, on the training split
    fit(model.vanilla_model, config.training.num_epochs, tag = "Stage 1a: vanilla arm")
    fit(model.attention_model, config.training.num_epochs, tag = "Stage 1b: attention arm")

    # Stage 2: freeze the arms, fit only the blend network
    model.freeze_submodels()
    if verbose:
        print(f"\n -- Stage 2: blend networks (sub-models frozen) -- ")
    fit(
        model,
        config.ensemble.weight_network_epochs,
        params = model.weight_network.parameters(),
    )
    return model 

def fit_regime_thresholds(model, train_dataset, feature_names, batch_size = 512):
    """ Fit the extractor's volatility terciles on the training split only"""
    extractor = getattr(model, "regime_extractor", None)
    if extractor is None:
        return 

    return_idx = feature_names.index("return_1")
    loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)

    chunks = []
    for x, _, _ in loader:
        chunks.append(x[:, :, return_idx])
        if sum(c.shape[0] for c in chunks) >= 20_000:
            break 

    if chunks:
        extractor.fit_thresholds(torch.cat(chunks))

### Reporting ###

def build_report(
    backtest: Dict,
    predictions: pd.DataFrame,
    returns_df: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    symbols: List[str],
    config: Config,
    periods_per_year: float,
    n_trials: int,
    burn_in_bars: int,
) -> Dict:
    net = backtest["net_returns"]
    gross = backtest["gross_returns"]

    baselines = {
        "spy": benchmark_returns(market, net.index),
        "equal_weight": equal_weight_baseline(returns_df, symbols, net.index),
        "tsmom": tsmom_baseline(prices, returns_df, net.index),
    }
    comparison = compare_to_baseline(net, baselines, periods_per_year)

    spy_aligned = baselines["spy"].reindex(net.index).fillna(0.0)
    common_vol = Metrics.annualized_volatility(spy_aligned, periods_per_year)
    if common_vol <= 0:
        common_vol = config.training.target_volatility

    vol_matched = {
        "target_volatility": common_vol,
        "strategy": Metrics.performance_summary(
            scale_to_volatility(net, common_vol, periods_per_year), periods_per_year
        ),
    }
    for name, series in baselines.items():
        aligned = series.reindex(net.index).fillna(0.0)
        vol_matched[name] = Metrics.performance_summary(
            scale_to_volatility(aligned, common_vol, periods_per_year),
            periods_per_year,
        )

    cost_scenarios = {}
    for rate in config.backtest.cost_scenarios:
        scenario = gross - backtest["turnover"] * rate
        cost_scenarios[f"{int(round(rate * 10000))}bps"] = {
            "cost_rate": float(rate),
            **Metrics.performance_summary(scenario, periods_per_year,
                                          turnover_series=backtest["turnover"]),
            "total_cost": float((backtest["turnover"] * rate).sum()),
        }

    # Predictive power, on RAW predictions
    merged = predictions.merge(
        returns_df.rename(columns = {"tradeable_return": "target"}),
        on = ["timestamp", "symbol"],
        how = "inner",
    ).dropna(subset = ["prediction", "target"])

    ic_pooled, ic_p = Metrics.information_coefficient(merged["prediction"], merged["target"])
    ic_cs = Metrics.cross_sectional_ic(merged)
    ic_boot = Metrics.block_bootstrap_ic(
        merged["prediction"].to_numpy(), merged["target"].to_numpy(), n_boot = 500
    )
    dir_acc = Metrics.directional_accuracy(merged["prediction"], merged["target"])
    pred_mean = float(merged["prediction"].mean()) if len(merged) else 0.0
    pred_median = float(merged["prediction"].median()) if len(merged) else 0.0
    target_mean = float(merged["target"].mean()) if len(merged) else 0.0
    target_median = float(merged["target"].median()) if len(merged) else 0.0
    demeaned_dir_acc = Metrics.directional_accuracy(
        merged["prediction"] - pred_mean,
        merged["target"] - target_mean,
    )

    dsr = Metrics.deflated_sharpe_ratio(net, n_trials = n_trials,
                                        periods_per_year=periods_per_year)

    return {
        "test_period": {
            "start": str(net.index.min()),
            "end": str(net.index.max()),
            "symbol_count": int(len(symbols)),
            "burn_in_bars": int(burn_in_bars),
        },
        "gross": Metrics.performance_summary(gross, periods_per_year),
        "net": Metrics.performance_summary(net, periods_per_year,
                                           turnover_series = backtest["turnover"]),
        "exposure": backtest["exposure"],
        "costs": {
            "cost_rate": backtest["cost_rate"],
            "n_rebalances": backtest["n_rebalances"],
            "rebalance_frequency": backtest["rebalance_frequency"],
            "total_turnover": backtest["total_turnover"],
            "mean_turnover": backtest["mean_turnover"],
            "total_cost": backtest["total_cost"],
        },
        "cost_scenarios": cost_scenarios,
        "baselines": comparison,
        "vol_matched": vol_matched,
        "prediction_quality": {
            "n_observations": int(len(merged)),
            "pooled_ic": ic_pooled,
            "pooled_ic_pvalue_NOMINAL": ic_p,
            "pooled_ic_pvalue_note": (
                "Nominal p-value assumes i.i.d observations. These are "
                "overlapping windows across correlated names, so it is badly "
                "optimistic. Use the cross-sectional t-stat and the bootstrap CI"
            ),
            "cross_sectional_ic": ic_cs,
            "block_bootstrap_ic": ic_boot,
            "directional_accuracy": dir_acc,
            "demeaned_directional_accuracy": demeaned_dir_acc,
            "prediction_mean": pred_mean,
            "prediction_median": pred_median,
            "target_mean": target_mean,
            "target_median": target_median,
        },
        "multiple_testing": dsr,
    }

def make_plots(net: pd.Series, baselines: Dict[str, pd.Series], regimes, regime_names,
               output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-darkgrid")

    # Equity curves vs baselines.
    fig, ax = plt.subplots(figsize=(12, 6))
    (1 + net).cumprod().plot(ax=ax, label="Strategy (net)", linewidth=2)
    for name, series in baselines.items():
        (1 + series.reindex(net.index).fillna(0)).cumprod().plot(
            ax=ax, label=name, alpha=0.75, linewidth=1.2
        )
    ax.set_title("Equity curves vs baselines")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curves.png", dpi=130)
    plt.close(fig)

    # Underwater.
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.to_numpy(), 0, color="crimson", alpha=0.4)
    ax.set_title("Underwater (net)")
    fig.tight_layout()
    fig.savefig(output_dir / "underwater_chart.png", dpi=130)
    plt.close(fig)

    # Rolling Sharpe on daily-aggregated returns.
    net_dates = pd.DatetimeIndex(net.index).date
    daily = (1 + net).groupby(net_dates).prod() - 1
    daily.index = pd.to_datetime(daily.index)
    window = min(63, max(10, len(daily) // 4))
    rolling = daily.rolling(window).mean() / daily.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 4))
    rolling.plot(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling {window}-day Sharpe (daily aggregation)")
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_sharpe.png", dpi=130)
    plt.close(fig)

    # Return distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(net.to_numpy(), bins=100, alpha=0.8)
    ax.set_title("Per-bar net return distribution (no winsorisation)")
    fig.tight_layout()
    fig.savefig(output_dir / "returns_distribution.png", dpi=130)
    plt.close(fig)

    # Regime attribution.
    if regimes is not None and len(regimes) == len(net):
        frame = pd.DataFrame({"ret": net.to_numpy(), "regime": regimes})
        stats = frame.groupby("regime")["ret"].agg(["mean", "std", "count"])
        stats["sharpe"] = stats["mean"] / stats["std"] * np.sqrt(PERIODS_PER_YEAR)
        labels = [regime_names.get(str(i), str(i)) for i in stats.index]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, stats["sharpe"].to_numpy())
        ax.set_title("Sharpe by detected regime")
        ax.set_ylabel("Sharpe")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / "regime_performance.png", dpi=130)
        plt.close(fig)


def write_summary(report: Dict, output_dir: Path, model_type: str) -> None:
    lines = []
    add = lines.append

    add("=" * 78)
    add(f"MOMENTUM TRANSFORMER BACKTEST -- model: {model_type}")
    add("=" * 78)

    net, gross, costs = report["net"], report["gross"], report["costs"]
    exposure = report["exposure"]
    test_period = report["test_period"]

    add("\nHEADLINE (net of costs)")
    add("-" * 78)
    if "daily_sharpe" in net:
        add(f"  Sharpe (daily aggregation, sqrt(252)) : {net['daily_sharpe']:8.3f}   <-- headline")
    add(f"  Sharpe (per-bar annualised)           : {net['sharpe_ratio']:8.3f}")
    add(f"  Sortino                               : {net['sortino_ratio']:8.3f}")
    add(f"  Calmar                                : {net['calmar_ratio']:8.3f}")
    add(f"  CAGR                                  : {net['annualized_return']:8.2%}")
    add(f"  Volatility                            : {net['volatility']:8.2%}")
    add(f"  Max drawdown                          : {net['max_drawdown']:8.2%}")
    add(f"  Total return                          : {net['total_return']:8.2%}")
    add(f"  Win rate                              : {net['win_rate']:8.2%}")
    add(f"  Periods / days                        : {net['n_periods']:,} / {net.get('n_days', 0):,}")
    add(f"  Bars per year used                    : {net['periods_per_year']:.0f}")
    add(f"  Test period                           : {test_period['start']} -> {test_period['end']}")
    add(f"  Symbols / burn-in bars                : {test_period['symbol_count']} / {test_period['burn_in_bars']}")

    add("\nEXPOSURE")
    add("-" * 78)
    add(f"  Gross exposure mean/median/p95/max    : "
        f"{exposure['mean_gross']:6.3f} / {exposure['median_gross']:6.3f} / "
        f"{exposure['p95_gross']:6.3f} / {exposure['max_gross']:6.3f}")
    add(f"  Net exposure mean/median/p95abs/maxabs: "
        f"{exposure['mean_net']:6.3f} / {exposure['median_net']:6.3f} / "
        f"{exposure['p95_abs_net']:6.3f} / {exposure['max_abs_net']:6.3f}")
    add(f"  Position count mean/max               : "
        f"{exposure['mean_position_count']:6.2f} / {exposure['max_position_count']}")

    add("\nTRADING ACTIVITY")
    add("-" * 78)
    add(f"  Rebalance frequency                   : {costs['rebalance_frequency']}")
    add(f"  Rebalances                            : {costs['n_rebalances']:,}")
    add(f"  Mean turnover per bar                 : {costs['mean_turnover']:8.4f}")
    add(f"  Total turnover                        : {costs['total_turnover']:8.2f}")
    add(f"  Cost rate (one-way)                   : {costs['cost_rate']:8.5f}")
    add(f"  Total cost drag                       : {costs['total_cost']:8.4f}")
    add(f"  Gross total return                    : {gross['total_return']:8.2%}")

    add("\nCOST SCENARIOS")
    add("-" * 78)
    for name, stats in report["cost_scenarios"].items():
        add(f"  {name:>5s} one-way  Sharpe {stats['sharpe_ratio']:7.3f} | "
            f"return {stats['total_return']:8.2%} | total cost {stats['total_cost']:7.4f}")

    add("\nBASELINES")
    add("-" * 78)
    for name, stats in report["baselines"].items():
        if name.startswith("alpha_beta"):
            continue
        add(f"  {name:14s} Sharpe {stats['sharpe_ratio']:7.3f} | "
            f"return {stats['total_return']:8.2%} | DD {stats['max_drawdown']:7.2%}")

    for key, value in report["baselines"].items():
        if key.startswith("alpha_beta"):
            add(f"\n  Regression vs {key.replace('alpha_beta_vs_', '').upper()}:")
            add(f"    Alpha (annualised) : {value['alpha_annual']:8.2%}  "
                f"(t = {value['alpha_tstat']:.2f})")
            add(f"    Beta               : {value['beta']:8.3f}")
            add(f"    Correlation        : {value['correlation']:8.3f}")

    vm = report["vol_matched"]
    add(f"\nVOL-MATCHED TO SPY VOL ({vm['target_volatility']:.2%})")
    add("-" * 78)
    for name, stats in vm.items():
        if name == "target_volatility":
            continue
        add(f"  {name:14s} Sharpe {stats['sharpe_ratio']:7.3f} | "
            f"CAGR {stats['annualized_return']:8.2%} | DD {stats['max_drawdown']:7.2%}")

    pq = report["prediction_quality"]
    cs = pq["cross_sectional_ic"]
    boot = pq["block_bootstrap_ic"]
    add("\nPREDICTIVE POWER (raw predictions, no rescaling)")
    add("-" * 78)
    add(f"  Observations                          : {pq['n_observations']:,}")
    add(f"  Pooled IC                             : {pq['pooled_ic']:8.4f}")
    add(f"  Mean cross-sectional IC               : {cs['mean_ic']:8.4f}")
    add(f"    t-stat / p-value                    : {cs['ic_tstat']:8.2f} / {cs['ic_pvalue']:.4f}")
    add(f"    timestamps                          : {cs['n_timestamps']:,}")
    add(f"  Block-bootstrap IC 95% CI             : "
        f"[{boot['ic_ci_low']:.4f}, {boot['ic_ci_high']:.4f}]")
    add(f"  Directional accuracy                  : {pq['directional_accuracy']:8.2%}")
    add(f"  Demeaned directional accuracy         : {pq['demeaned_directional_accuracy']:8.2%}")
    add(f"  Prediction mean / median              : "
        f"{pq['prediction_mean']:9.5f} / {pq['prediction_median']:9.5f}")
    add(f"  Target mean / median                  : "
        f"{pq['target_mean']:9.5f} / {pq['target_median']:9.5f}")
    add(f"\n  {pq['pooled_ic_pvalue_note']}")

    mt = report["multiple_testing"]
    add("\nMULTIPLE-TESTING ADJUSTMENT")
    add("-" * 78)
    add(f"  Configurations counted                : {mt['n_trials']}")
    add(f"  Expected max Sharpe under null        : {mt['expected_max_sharpe']:8.3f}")
    add(f"  Sharpe standard error                 : {mt['sharpe_standard_error']:8.3f}")
    add(f"  Deflated Sharpe (P[true SR > 0])      : {mt['deflated_sharpe']:8.4f}")
    add("\n" + "=" * 78)

    text = "\n".join(lines)
    (output_dir / "performance_summary.txt").write_text(text)
    print("\n" + text)

### Main ###

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["vanilla", "attention", "ensemble"],
                   default="ensemble")
    p.add_argument("--csv", default=str(PROJECT_ROOT / "OHLCV-1HR" / "OHLCV.csv"))
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default=None)
    p.add_argument("--walk-forward", action="store_true",
                   help="Run walk-forward analysis instead of a single split")
    p.add_argument("--n-trials", type=int, default=93,
                   help="Configurations tried, for the deflated Sharpe "
                        "(30 tuning trials x 3 architectures + 3 seeds)")
    p.add_argument("--accumulation-steps", type=int, default=None,
                   help="Mini-batches pooled per Sharpe estimate (0 = whole epoch)")
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    started = time.time()

    config = get_production_config()
    config.seed = args.seed
    config.selection_strategy = args.model
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.rebalance is not None:
        config.backtest.rebalance_frequency = args.rebalance
    if args.accumulation_steps is not None:
        config.training.sharpe_accumulation_steps = (
            None if args.accumulation_steps == 0 else args.accumulation_steps
        )

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT / "outputs" / f"run_{pd.Timestamp.now():%Y-%m-%d_%H%M}_{args.model}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"MOMENTUM TRANSFORMER -- {args.model.upper()}")
    print(f"Output: {output_dir}")
    print("=" * 78)

    # 1. Data 
    bundle = prepare_market_data(
        args.csv, config.data, max_symbols=args.max_symbols,
        start=args.start, end=args.end,
    )
    prices, symbols = bundle["prices"], bundle["symbols"]
    market, calendar = bundle["market"], bundle["calendar"]

    ppy = derive_periods_per_year(calendar)
    print(f"\n[Calendar] Empirical bars/year: {ppy:.0f} "
          f"(constant PERIODS_PER_YEAR = {PERIODS_PER_YEAR})")

    returns_df = tradeable_returns(prices)

    # 2. Features
    feature_config = FeatureConfig()
    features_df = build_feature_frame(prices, market, symbols, returns_df, feature_config)
    quality = run_feature_quality_checks(features_df)

    # 3. Splits 
    train_end = pd.Timestamp(config.data.train_end)
    val_end = pd.Timestamp(config.data.val_end)

    train_df = features_df[features_df["timestamp"] <= train_end]
    val_df = features_df[(features_df["timestamp"] > train_end)
                         & (features_df["timestamp"] <= val_end)]
    test_df = features_df[features_df["timestamp"] > val_end]

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        print("\nERROR: one of the splits is empty. Check --start/--end against "
              f"train_end={config.data.train_end}, val_end={config.data.val_end}.")
        return 1

    print(f"\n[Splits] train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    # 4. Normalisation, fitted on TRAIN ONLY 
    feature_cols = [c for c in features_df.columns if c not in METADATA_COLUMNS]
    normalizer = FeatureNormalizer().fit(train_df[feature_cols])

    def normalise(df):
        out = df.copy()
        out[feature_cols] = normalizer.transform(df[feature_cols])
        return out

    train_df, val_df, test_df = normalise(train_df), normalise(val_df), normalise(test_df)
    (output_dir / "feature_normalizer.json").write_text(
        json.dumps(normalizer.to_dict(), indent=2, default=float)
    )

    config.model.input_dim = len(feature_cols)

    # 5. Datasets 
    print("\n[Datasets]")
    seq_len = config.model.sequence_length
    train_ds = MultiAssetWindowDataset(train_df, seq_len, feature_cols, calendar)
    val_ds = MultiAssetWindowDataset(val_df, seq_len, feature_cols, calendar)
    test_ds = MultiAssetWindowDataset(test_df, seq_len, feature_cols, calendar)

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        print("\nERROR: a split produced zero windows. Reduce --max-symbols or "
              "widen the date range.")
        return 1

    val_calendar = calendar[(calendar > train_end) & (calendar <= val_end)]
    test_calendar = calendar[calendar > val_end]

    validation_fn = make_validation_fn(
        val_ds, returns_df, val_calendar, config, config.training.batch_size, ppy
    )

    # 6. Train 
    print(f"\n[Training] {args.model}")
    model = train_variant(
        args.model, config, train_ds, val_ds, feature_cols, validation_fn
    )

    # 7. Walk-forward (optional) 
    if args.walk_forward:
        analyzer = WalkForwardAnalyzer(
            train_bars=config.backtest.wf_train_bars,
            test_bars=config.backtest.wf_test_bars,
            step_bars=config.backtest.wf_step_bars,
            backtest_config=config.backtest,
            periods_per_year=ppy,
        )

        def train_fn(fit_features, val_features):
            wf_normalizer = FeatureNormalizer().fit(fit_features[feature_cols])

            def wf_normalise(df):
                out = df.copy()
                out[feature_cols] = wf_normalizer.transform(df[feature_cols])
                return out

            fit_features_n = wf_normalise(fit_features)
            val_features_n = wf_normalise(val_features)

            fit_ds = MultiAssetWindowDataset(fit_features_n, seq_len, feature_cols,
                                             calendar, verbose=False)
            v_ds = MultiAssetWindowDataset(val_features_n, seq_len, feature_cols,
                                           calendar, verbose=False)
            if len(fit_ds) == 0 or len(v_ds) == 0:
                raise RuntimeError("Walk-forward window produced no windows")
            m = train_variant(args.model, config, fit_ds, v_ds, feature_cols,
                              None, verbose=False)
            setattr(m, "_walk_forward_normalizer", wf_normalizer)
            return m

        def predict_fn(m, test_features):
            wf_normalizer = getattr(m, "_walk_forward_normalizer", None)
            if wf_normalizer is not None:
                test_features = test_features.copy()
                test_features[feature_cols] = wf_normalizer.transform(
                    test_features[feature_cols]
                )
            ds = MultiAssetWindowDataset(test_features, seq_len, feature_cols,
                                         calendar, verbose=False)
            if len(ds) == 0:
                return pd.DataFrame(columns=["timestamp", "symbol", "prediction"])
            return predict_dataset(m, ds, ds.index_frame, config.training.batch_size)

        wf = analyzer.run(features_df, returns_df, calendar, train_fn, predict_fn)
        print_walk_forward_results(wf)
        (output_dir / "walk_forward.json").write_text(
            json.dumps({k: v for k, v in wf.items() if k != "pooled_returns"},
                       indent=2, default=float)
        )
        wf["pooled_returns"].to_csv(output_dir / "walk_forward_returns.csv")

    # 8. Test-period backtest 
    print("\n[Backtest] Generating predictions across the full test period ...")
    predictions = predict_dataset(model, test_ds, test_ds.index_frame,
                                  config.training.batch_size)
    print(f"  {len(predictions):,} predictions "
          f"({predictions['symbol'].nunique()} symbols, "
          f"{predictions['timestamp'].nunique():,} timestamps)")

    backtest = run_portfolio_backtest(predictions, returns_df, test_calendar,
                                      config.backtest)
    net = backtest["net_returns"]
    if net.empty:
        print("\nERROR: backtest produced no returns.")
        return 1

    print(f"  Rebalances: {backtest['n_rebalances']} | "
          f"mean turnover/bar: {backtest['mean_turnover']:.4f} | "
          f"total cost: {backtest['total_cost']:.4f}")

    # 9. Report 
    report = build_report(backtest, predictions, returns_df, prices, market,
                          test_calendar, symbols, config, ppy, args.n_trials,
                          feature_config.max_lookback)

    detector = StatisticalRegimeDetector().fit(
        returns_df["tradeable_return"].to_numpy()
    )
    regimes, regime_names = label_regimes(net.to_numpy(), lookback=min(252, len(net) // 4),
                                          detector=detector)

    baselines = {
        "spy": benchmark_returns(market, net.index),
        "equal_weight": equal_weight_baseline(returns_df, symbols, net.index),
        "tsmom": tsmom_baseline(prices, returns_df, net.index),
    }
    make_plots(net, baselines, regimes, regime_names, output_dir)
    write_summary(report, output_dir, args.model)

    # 10. Persist 
    pd.DataFrame(
        {"timestamp": net.index, "net_return": net.to_numpy(),
         "gross_return": backtest["gross_returns"].to_numpy(),
         "turnover": backtest["turnover"].to_numpy()}
    ).to_csv(output_dir / "portfolio_returns.csv", index=False)

    backtest["weights"].to_csv(output_dir / "weights.csv")
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    if isinstance(model, EnsembleMomentumTransformer):
        model.record_weights(True)
        model.reset_weight_history()
        predict_dataset(model, test_ds, test_ds.index_frame, config.training.batch_size)
        stats = model.get_weight_statistics()
        model.record_weights(False)
        report["ensemble_weights_TEST_ONLY"] = stats

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "argv": vars(args),
                "config": asdict(config),
                "empirical_periods_per_year": ppy,
                "data_diagnostics": bundle["diagnostics"],
                "feature_count": len(feature_cols),
                "feature_columns": feature_cols,
                "collinear_pairs": quality["collinear_pairs"],
                "n_parameters": count_parameters(model),
                "runtime_seconds": time.time() - started,
            },
            indent=2, default=str,
        )
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"\nDone in {time.time() - started:.1f}s. Artifacts in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
