""" Momentum Transformer model package"""

from .config import (
    ANNUALIZATION_SQRT,
    BARS_PER_DAY,
    PERIODS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    BacktestConfig,
    Config,
    DataConfig,
    EnsembleConfig,
    ModelConfig,
    RegimeDetectorConfig,
    TrainingConfig,
    apply_tuned_hyperparameters,
    bars,
    derive_periods_per_year,
    get_default_config,
    get_enhanced_config,
    get_paper_config,
    get_production_config,
)

from .LSTM import (
    LSTMMomentum,
    LSTMMomentumDualPath,
    LSTMMomentumEncoder,
    get_lstm_encoder,
)

from .Transformer_layers import (
    InterpretableMultiheadAttention,
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderBlock,
    build_causal_mask,
    get_transformer_encoder,
)

from .Momentum_transformer import (
    MomentumTransformer,
    MomentumTransformerDualPath,
    MomentumTransformerSimple,
    apply_volatility_target,
    count_parameters,
    get_momentum_transformer,
    model_kwargs_from_config,
)

from .Ensemble_model import (
    REGIME_FEATURE_NAMES,
    DynamicWeightNetwork,
    EnsembleMomentumTransformer,
    RegimeFeatureExtractor,
    get_ensemble_model,
)

__version__ = "2.0.0"

__all__ = [
    # Constants
    "BARS_PER_DAY", "TRADING_DAYS_PER_YEAR", "PERIODS_PER_YEAR",
    "ANNUALIZATION_SQRT", "bars", "derive_periods_per_year",

    # Config
    "Config", "ModelConfig", "TrainingConfig", "EnsembleConfig",
    "RegimeDetectorConfig", "BacktestConfig", "DataConfig",
    "get_default_config", "get_paper_config", "get_enhanced_config",
    "get_production_config", "apply_tuned_hyperparameters",

    # LSTM
    "LSTMMomentumEncoder", "LSTMMomentum", "LSTMMomentumDualPath",
    "get_lstm_encoder",

    # Transformer
    "PositionalEncoding", "TransformerEncoderBlock", "TransformerEncoder",
    "InterpretableMultiheadAttention", "build_causal_mask", 
    "get_transformer_encoder",

    # Models
    "MomentumTransformer", "MomentumTransformerSimple",
    "MomentumTransformerDualPath", "get_momentum_transformer",
    "model_kwargs_from_config", "apply_volatility_target", "count_parameters",

    # Ensemble
    "RegimeFeatureExtractor", "DynamicWeightNetwork",
    "EnsembleMomentumTransformer", "get_ensemble_model",
    "REGIME_FEATURE_NAMES",
]
