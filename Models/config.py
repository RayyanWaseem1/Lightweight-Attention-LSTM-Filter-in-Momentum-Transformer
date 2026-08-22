""" Configuration for the Momentum Transformer.

This module is the single source of truth for:

* the bar/calendar constants used for every annualization in the repo
    (``BARS_PER_DAY``, ``PERIODS_PER_YEAR``), and 
* the dataclasses holding model / training / backtest hyperparameters
"""

from dataclasses import dataclass, field 
from typing import List, Literal, Optional 

import numpy as np 
import pandas as pd

### Calendear constants ###

# The OHLCV feed is US equity hourly data. Regular trading hours produce a
# median of 7 bars per session (13:00 - 19:00 UTC); the remainig hours in the 
# file are sparse extended-hours prints and are excluded by the trading 
# calendar in ``Utils.Market_data``

# Everything in the repo annualizations with PERIODS_PER_YEAR. Reported headline
# figures additionally aggregate PnL to daily and use sqrt(252) 

BARS_PER_DAY: int = 7
TRADING_DAYS_PER_YEAR: int = 252
PERIODS_PER_YEAR: int = BARS_PER_DAY * TRADING_DAYS_PER_YEAR #1764
ANNUALIZATION_SQRT: float = float(np.sqrt(PERIODS_PER_YEAR))

def bars(days: float) -> int:
    """ Convert a horizon expressed in trading days into a number of bars"""
    return max(1, int(round(days * BARS_PER_DAY)))

def derive_periods_per_year(timestamps) -> float:
    """ Empirically derive bars/year from an observed timestamp index
    
    Prefer this over the ``PERIODS_PER_YEAR`` constant whenever a real series is
    in hand: it reflects the calendar the data actually has rather than the 
    calendar we assume it has. Falls back to the constant for degenrate input
    """

    ts = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(timestamps)))
    if len(ts) < 2:
        return float(PERIODS_PER_YEAR)

    span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
    if span_days <= 0:
        return float(PERIODS_PER_YEAR)


    years = span_days / 365.25
    return float(len(ts) / years)

@dataclass
class ModelConfig:
    """ Model architecture configuration"""

    # LSTM Encoder 
    input_dim: int = 32
    hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    # Local window the LSTM filter operates over: 9 trading days
    short_window: int = bars(9) # 63

    # Transformer
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    transformer_dropout: float = 0.2
    feedforward_dim: int = 256

    # Sequence
    #Full model lookback: 36 trading days. Chosen so that it is an exact 
    # multiple of ``short_window`` (252 = 4 * 63), which the strided encoder
    # relies on 
    sequence_length: int = bars(36) # 252

    # Architecture choices
    use_lstm_attention: bool = True
    use_positional_encoding: bool = True
    use_causal_mask: bool = True 

    # How the LSTM filter relates to the Transformer's receptive field 
    #   "strided" - run the LSTM over consecutive ``short_window11 blocks and 
    #               let the Transformer attend across the resulting local 
    #               embeddings. Implements "local filter + global attention".
    #   "full" -    run the LSTM across the whole sequence (closes to 
    #               Wood et al. 2112.08534)
    #   "truncate" - legacy behavior: keep only the last ``short_window`` bars 

    lstm_transformer_mode: Literal["strided", "full", "truncate"] = "strided"

    # Position sizing 
    # The head emits tanh(z) in [-1, 1]. When volatility targeting is on, the 
    # traded position is tanh(z) * target_vol / ex_ante_vol 

    use_volatility_targeting: bool = True 

@dataclass 
class TrainingConfig:
    """ Training configuration"""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    num_epochs: int = 100

    # Gradient management
    gradient_clip_norm: float = 1.0

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Early stopping
    # Selection criterion is validation Sharpe, so mode is always "max"
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0

    # Loss 
    # Sharpe is estimated over `sharpe_accumulation_steps` mini-batches pooled
    # together, not per mini-batch. At batch_size = 128 and 16 steps each estimate 
    # sees 2,048 observations instead of 128, cutting the standard error of the ratio by 4x

    # Set to None for the true whole-epoch Sharpe. That is the ideal, but it 
    # requires holding the entire epoch's autograd graph: on this dataset 
    # (~17k windows * 252 bars * 54 features per split) it does not fit in 
    # memory or finish in reasonable time on CPU. 16 is the practical setting 

    sharpe_accumulation_steps: Optional[int] = 16
    target_volatility: float = 0.15 # 15% annualized, used by vol targeting 

    # Device
    device: str = "cpu"

@dataclass
class EnsembleConfig:
    """ Ensemble model configuration"""

    regime_feature_dim: int = 21
    weight_hidden_dim: int = 32
    weight_dropout: float = 0.2 

    # Train the two sub-models to convergence first, freeze them, then fit the
    # weight network on the validation split
    pretrain_submodels: bool = True
    weight_network_epochs: int = 10

@dataclass
class RegimeDetectorConfig:
    """ Market regime detection configuration"""

    lookback_short: int = bars(3) # 21 bars ~ 3 trading days
    lookback_long: int = bars(9) # 63 bars ~ 9 trading days 

    # Volatility terciles are fitted on the training split at runtime (see
    # ``Models.Ensemble_model.RegimeFeatureExtractor.fit_thresholds``). These
    # are only the fallbacks used before fitting, calibrated for hourly bars
    volatility_threshold_low: float = 0.004
    volatility_threshold_high: float = 0.010

@dataclass
class BacktestConfig:
    """ Backtest / portfolio construction configuration"""

    # Costs, charged on realized turnover: sum |w_t - w_{t-1}| * cost_rate
    spread_cost: float = 0.0005 # half-spread, 5 bps
    fee: float = 0.0001 # 1 bp
    impact: float = 0.0 # reserved for a size-dependent term 

    # Portfolio construction 
    max_positions: int = 10 
    max_weight: float = 0.2 
    rebalance_frequency: str = "weekly" # daily | weekly | monthly 

    # Walk-forward windows, expressed in trading days and converted to bars 
    wf_train_days: int = 504 # 2 years
    wf_test_days: int = 63 # 1 quarter
    wf_step_days: int = 63

    initial_capital: float = 1.0 

    @property 
    def cost_rate(self) -> float:
        """ One-way cost charged per unit of turnover"""
        return self.spread_cost + self.fee + self.impact 

    @property
    def wf_train_bars(self) -> int:
        return bars(self.wf_train_days)

    @property
    def wf_test_bars(self) -> int:
        return bars(self.wf_test_days)

    @property
    def wf_step_bars(self) -> int:
        return bars(self.wf_step_days)

@dataclass
class DataConfig:
    """ Data processing configuration"""

    data_end: Optional[str] = None # None means "use everythign in the file"

    # Split boundaries (inclusive upper bounds)
    train_end: str = "2020-12-31"
    val_end: str = "2021-12-31"

    # Universe filters
    min_price: float = 2.0
    min_bars: int = 3000
    max_abs_return: float = 10.0 # drop symbols with any |return| > 1000%

    # Causal bad-tick filter: drop bars where |r| > k * trailing vol 
    bad_tick_k: float = 12.0
    bad_tick_vol_window: int = field(default_factory = lambda: bars(21))

    # The benchmark is loaded outside the tradeable universe and must be present
    market_symbol: str = "SPY"

@dataclass
class Config:
    """ Master configuration"""

    model: ModelConfig = field(default_factory = ModelConfig)
    training: TrainingConfig = field(default_factory = TrainingConfig)
    ensemble: EnsembleConfig = field(default_factory = EnsembleConfig)
    regime_detector: RegimeDetectorConfig = field(default_factory = RegimeDetectorConfig)
    backtest: BacktestConfig = field(default_factory = BacktestConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment tracking 
    experiment_name: str = "momentum_transformer"
    save_dir: str = "./checkpoints"
    seed: int = 123 

    selection_strategy: Literal["vanilla", "attention", "ensemble"] = "ensemble"

    def to_dict(self) -> dict:
        """ Flatten to a JSON serializable dict for run provenance"""
        from dataclasses import asdict

        return asdict(self)

def get_default_config() -> Config:
    return Config() 

def get_paper_config() -> Config:
    """ Configuration matching the research paper's setup (no LSTM attention)"""
    config = Config() 
    config.model.use_lstm_attention = False 
    config.selection_strategy = "vanilla"
    return config 

def get_enhanced_config() -> Config:
    """ Attention-enhanced LSTM filter"""
    config = Config()
    config.model.use_lstm_attention = True 
    config.selection_strategy = "attention"
    return config 

def get_production_config() -> Config: 
    """ Ensemble configuration 
    
    Hyperparameters here are defaults only. The resolved configuration for a 
    run is loaded from the tuning output via
    ``outputs.Hyperparameter_tuning.load_best_config`` and logged alongside the 
    results, so any result can be traced to the parameters that produced it
    """
    config = Config()
    config.selection_strategy = "ensemble"
    return config 

def apply_tuned_hyperparameters(config: Config, tuned: dict) -> Config:
    """ Overlay a Ray Tune ``best_config.json`` payload onto ``config``.
    
    Unknown keys raise, so a stale tuning artifact fails loudly instead of 
    silently leaving the searched dimension at its default
    """

    mapping = {
        "hidden_dim": ("model", "hidden_dim"),
        "num_transformer_layers": ("model", "num_transformer_layers"),
        "num_attention_heads": ("model", "num_attention_heads"),
        "dropout": ("model", "transformer_dropout"),
        "lstm_num_layers": ("model", "lstm_num_layers"),
        "lstm_dropout": ("model", "lstm_dropout"),
        "feedforward_dim": ("model", "feedforward_dim"),
        "short_window": ("model", "short_window"),
        "learning_rate": ("training", "learning_rate"),
        "weight_decay": ("training", "weight_decay"),
        "batch_size": ("training", "batch_size"),
        "num_epochs": ("training", "num_epochs"),
        "weight_hidden_dim": ("ensemble", "weight_hidden_dim"),
        "weight_dropout": ("ensemble", "weight_dropout"),
    }

    unknown = sorted(set(tuned) - set(mapping))
    if unknown:
        raise KeyError(
            f"Tuned config contains keys with no home in Config: {unknown}. "
            "Add them to apply_tuned_hyperparameters or remove them from the "
            "tuning search space."
        )

    for key,value in tuned.items():
        section, attr = mapping[key]
        setattr(getattr(config, section), attr, value)

    return config
