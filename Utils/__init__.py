"""Utilities: metrics, data preparation, features, portfolio, training, losses

Import order note: ''Utils.Metrics'' and ''Utils.Portfolio'' import
''Models.config'' for the calendar constants. ''Models'' does not import 
''Utils'', so the dependency runs one way only.
"""

from . import Metrics

from .Losses import (
    LOSS_REGISTRY,
    CalmarRatioLoss,
    CombinedLoss,
    InformationRatioLoss,
    MaximumDrawdownLoss,
    NormalizedDrawdownLoss,
    SharpeRatioLoss,
    SharpeWithTurnoverPenalty,
    SoftDirectionalLoss,
    SortinoRatioLoss,
    get_loss_function,
)

from .Training import (
    EarlyStopping,
    GradientClipper,
    PerformanceTracker,
    SequentialBlockSampler,
    TrainingMetrics,
    compute_metrics,
    compute_sharpe_ratio,
    evaluate_model,
    print_metrics,
    train_epoch,
    train_model,
)

from .Feature_engineering import (
    FeatureConfig,
    FeatureNormalizer,
    StockAgnosticFeatureEngineer,
    assert_contiguous,
    check_collinearity,
    check_feature_scales,
    create_features_from_ohlcv,
)

from .Market_data import (
    back_adjust_splits,
    build_trading_calendar,
    causal_bad_tick_mask,
    detect_splits,
    infer_regular_hours,
    load_raw_ohlcv,
    prepare_market_data,
    tradeable_returns,
)

from .Portfolio import (
    build_weight_matrix,
    compare_to_baseline,
    construct_weights,
    equal_weight_baseline,
    benchmark_returns,
    buy_and_hold,
    rebalance_timestamps,
    run_portfolio_backtest,
    tsmom_baseline,
)

from .Regime_detector import (
    MarketRegime,
    RegimeMetrics,
    RollingRegimeDetector,
    StatisticalRegimeDetector,
    label_regimes,
)

_all__ = [
    "Metrics",
    # Losses
    "SharpeRatioLoss", "SortinoRatioLoss", "CalmarRatioLoss",
    "MaximumDrawdownLoss", "NormalizedDrawdownLoss",
    "SharpeWithTurnoverPenalty", "CombinedLoss", "SoftDirectionalLoss",
    "InformationRatioLoss", "get_loss_function", "LOSS_REGISTRY",
    # Training
    "EarlyStopping", "GradientClipper", "PerformanceTracker", 
    "SequentialBlockSampler", "TrainingMetrics", "train_model", "train_epoch",
    "evaluate_model", "compute_sharpe_ratio", "compute_metrics", "print_metrics",
    # Features
    "FeatureConfig", "FeatureNormalizer", "StockAgnosticFeatureEngineer",
    "create_features_from_ohlcv", "check_collinearity", "check_feature_scales",
    "assert_contiguous",
    # Data
    "load_raw_ohlcv", "prepare_market_data", "tradeable_returns",
    "back_adjust_splits", "detect_splits", "build_trading_calendar",
    "infer_regular_hours", "causal_bad_tick_mask",
    # Portfolio
    "rebalance_timestamps", "construct_weights", "build_weight_matrix",
    "run_portfolio_backtest", "buy_and_hold", "benchmark_returns",
    "equal_weight_baseline",
    "tsmom_baseline", "compare_to_baseline",
    # Regime 
    "MarketRegime", "RegimeMetrics", "StatisticalRegimeDetector",
    "RollingRegimeDetector", "label_regimes",
]
