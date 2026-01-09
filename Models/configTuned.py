#config file for the Momentum Transformer
#Contains the hyperparameters

from dataclasses import dataclass, field
from typing import Literal 

@dataclass
class ModelConfig: 
    #Model architecture configurations
    
    #LSTM Encoder
    input_dim: int = 32 
    hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    short_window: int = 63 #LSTM lookback window (1 quarter)

    #Transformer
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    transformer_dropout: float = 0.2
    feedforward_dim: int = 256 

    #Sequence
    sequence_length: int = 252 #full look back (1 year)

    #Architecture choices
    use_lstm_attention: bool = True #Whether or not to use the attention in the LSTM
    use_positional_encoding: bool = True 


@dataclass
class TrainingConfig:
    #Training configurations

    #Optimizations
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    num_epochs: int = 100

    #Gradient management
    gradient_clip_norm: float = 1.0

    #Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    #Data splits
    train_ratio: float = 0.90
    val_ratio: float = 0.10

    #Loss Function
    target_volatility: float = 0.15 #15% annualized

    #Device
    device: str = "cuda" #or cpu


@dataclass
class EnsembleConfig:
    #Ensemble model configurations

    #Regime feature dimensions
    regime_feature_dim: int = 10

    #weight network architecture
    weight_hidden_dim: int = 32
    weight_dropout: float = 0.2


@dataclass
class OnlineSelectorConfig:
    #Online performance monitoring configurations
    lookback_window: int = 50
    switch_threshold: float = 0.2 #Sharpe difference used to trigger a switch in model selection
    min_observations: int = 20
    switch_cooldown_period: int = 10 #Days before allowing another model switch

    #Monte Carlo dropout for any uncertainty estimation
    num_mc_samples: int = 10
    uncertainty_threshold: float = 0.15


@dataclass
class RegimeDetectorConfig:
    #Market regime detection configurations
    lookback_short: int = 21 #1 month
    lookback_long: int = 63 #1 quarter

    #Volatility thresholds (annualized)
    volatility_threshold_low: float = 0.01
    volatility_threshold_high: float = 0.03

    #Attention scoring weights
    volatility_weight: float = 0.3
    trend_weight: float = 0.2
    changepoint_weight: float = 0.3
    confidence_weight: float = 0.2


@dataclass
class DataConfig:
    #Data processing configurations


    #Feature horizons
    return_horizons: dict = None
    macd_pairs: list = None 
    cpd_lookbacks: list = None 

    #Data source
    data_start: str = "1990-01-01"
    data_end: str = None #None means today/current date

    #Asset universe
    num_assets: int = 50

    def __post_init__(self):
        if self.return_horizons is None:
            self.return_horizons = {
                "1d": 1,
                "1w": 5,
                "1m": 21,
                "3m": 63,
                "6m": 126,
                "12m": 252
            }

        if self.macd_pairs is None:
            self.macd_pairs = [(8,24), (16,48), (32,96)]

        if self.cpd_lookbacks is None:
            self.cpd_lookbacks = [21,126]


@dataclass
class Config:
    #Master configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig) 
    online_selector: OnlineSelectorConfig = field(default_factory=OnlineSelectorConfig)
    regime_detector: RegimeDetectorConfig = field(default_factory=RegimeDetectorConfig)
    data: DataConfig = field(default_factory=DataConfig) 

    #Experiment tracking
    experiment_name: str = "momentum_transformer"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    #Selection strategy
    selection_strategy: Literal["vanilla", "attention", "ensemble", "online"] = "ensemble"


#Default configs
def get_default_config() -> Config:
    #Get default configuration
    return Config() 

#Configuration presets
def get_paper_config() -> Config:
    #Configuration matching the research paper's setup
    config = Config()
    config.model.use_lstm_attention = False  #research paper uses a simple LSTM as filtering 
    config.model.hidden_dim = 64
    config.model.num_transformer_layers = 2
    config.training.batch_size = 128
    return config

def get_enhanced_config() -> Config:
    #Enhanced configuration with LSTM attention
    config = Config()
    config.model.use_lstm_attention = True #We are using attention in our LSTM filter
    config.model.hidden_dim = 64
    config.model.num_transformer_layers = 2
    config.training.batch_size = 128
    return config 

def get_production_config() -> Config:
    """
    Production configuration with ensemble
    *** TUNED HYPERPARAMETERS FROM RAY TUNE (Val Sharpe: 0.6213) ***
    """
    config = Config() 
    config.selection_strategy = "ensemble"
    
    # ===== TUNED MODEL ARCHITECTURE =====
    config.model.hidden_dim = 64  # Optimal size
    config.model.num_transformer_layers = 1  # Simpler is better (was 2)
    config.model.num_attention_heads = 2  # Fewer heads (was 4)
    config.model.transformer_dropout = 0.226  # ~0.226 (was 0.2)
    config.model.lstm_dropout = 0.183  # ~0.183 (was 0.2)
    config.model.lstm_num_layers = 2  # Keep at 2
    
    # ===== TUNED TRAINING PARAMETERS =====
    config.training.learning_rate = 0.000222  # Lower LR (was 0.001)
    config.training.weight_decay = 0.00003116  # ~3.1e-5 (was 1e-5)
    config.training.batch_size = 256  # Larger batches (was 128)
    
    # ===== TUNED ENSEMBLE PARAMETERS =====
    config.ensemble.weight_hidden_dim = 32  # Keep at 32
    config.ensemble.weight_dropout = 0.237  # ~0.237 (was 0.2)
    
    return config