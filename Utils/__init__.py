"""
Utils Package for Momentum Transformer

Training utilities, loss functions, feature engineering, and regime detection
"""

from .Losses import (
    SharpeRatioLoss,
    SortinoRatioLoss,
    CalmarRatioLoss,
    MaximumDrawdownLoss,
    SharpeWithTurnoverPenalty,
    CombinedLoss,
    DirectionalAccuracyLoss,
    InformationRatioLoss,
    get_loss_function,
    sharpe_ratio_loss,
    sortino_ratio_loss,
    calmar_ratio_loss,
    maximum_drawdown_loss
)

from .Training import (
    EarlyStopping,
    GradientClipper,
    PerformanceTracker,
    TrainingMetrics,
    train_model,
    train_epoch,
    evaluate_model,
    compute_sharpe_ratio,
    compute_metrics,
    print_metrics
)

from .Feature_engineering import (
    create_features_from_ohlcv,
    normalize_features
)

from .Regime_detector import (
    RegimeFeatureExtractor,
    StatisticalRegimeDetector,
    HiddenMarkovRegimeDetector,
    RollingRegimeDetector,
    MarketRegime,
    RegimeMetrics,
    extract_regime_features_numpy
)

from .Backtesting import (
    Backtester,
    BacktestConfig,
    BacktestResults,
    WalkForwardAnalyzer,
    print_backtest_results
)

__all__ = [
    # Losses
    'SharpeRatioLoss',
    'SortinoRatioLoss',
    'CalmarRatioLoss',
    'MaximumDrawdownLoss',
    'SharpeWithTurnoverPenalty',
    'CombinedLoss',
    'DirectionalAccuracyLoss',
    'InformationRatioLoss',
    'get_loss_function',
    'sharpe_ratio_loss',
    'sortino_ratio_loss',
    'calmar_ratio_loss',
    'maximum_drawdown_loss',
    
    # Training
    'EarlyStopping',
    'GradientClipper',
    'PerformanceTracker',
    'TrainingMetrics',
    'train_model',
    'train_epoch',
    'evaluate_model',
    'compute_sharpe_ratio',
    'compute_metrics',
    'print_metrics',
    
    # Feature Engineering
    'create_features_from_ohlcv',
    'normalize_features',
    
    # Regime Detection
    'RegimeFeatureExtractor',
    'StatisticalRegimeDetector',
    'HiddenMarkovRegimeDetector',
    'RollingRegimeDetector',
    'MarketRegime',
    'RegimeMetrics',
    'extract_regime_features_numpy',
    
    # Backtesting
    'Backtester',
    'BacktestConfig',
    'BacktestResults',
    'WalkForwardAnalyzer',
    'print_backtest_results',
]
