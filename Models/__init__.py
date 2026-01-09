"""
Momentum Transformer Package
Easy imports for all components
"""

# Configuration
from .config import (
    Config,
    ModelConfig,
    TrainingConfig,
    EnsembleConfig,
    OnlineSelectorConfig,
    get_default_config,
    get_paper_config,
    get_enhanced_config,
    get_production_config
)

# LSTM Encoders
from .LSTM import (
    LSTMMomentum,
    LSTMMMomentumDualPath,
    LSTMMomentumWithAttention,
    get_lstm_encoder
)

# Transformer Components
from .Transformer_layers import (
    PositionalEncoding,
    TransformerEncoderBlock,
    TransformerEncoder,
    InterpretableMultiheadAttention,
    get_transformer_encoder
)

# Full Models
from .Momentum_transformer import (
    MomentumTransformer,
    MomentumTransformerSimple,
    MomentumTransformerDualPath,
    get_momentum_transformer
)

# Ensemble (Strategy 2)
from .Ensemble_model import (
    RegimeFeatureExtractor,
    DynamicWeightNetwork,
    EnsembleMomentumTransformer,
    get_ensemble_model
)

# Online Selector (Strategy 4)
from .Online_Selector import (
    PerformanceMetrics,
    PerformanceTracker,
    OnlineModelSelector,
    HybridSelector
)

# Loss Functions
from Utils.Losses import (
    sharpe_ratio_loss,
    sortino_ratio_loss,
    calmar_ratio_loss,
    sharpe_with_turnover_penalty,
    maximum_drawdown_loss,
    CombinedLoss,
    get_loss_function
)

__version__ = "1.0.0"

__all__ = [
    # Config
    'Config',
    'ModelConfig',
    'TrainingConfig',
    'EnsembleConfig',
    'OnlineSelectorConfig',
    'get_default_config',
    'get_paper_config',
    'get_enhanced_config',
    'get_production_config',
    
    # LSTM
    'LSTMMomentum',
    'LSTMMomentumDualPath',
    'LSTMMomentumWithAttention',
    'get_lstm_encoder',
    
    # Transformer
    'PositionalEncoding',
    'TransformerEncoderBlock',
    'TransformerEncoder',
    'InterpretableMultiheadAttention',
    'get_transformer_encoder',
    
    # Full Models
    'MomentumTransformer',
    'MomentumTransformerSimple',
    'MomentumTransformerDualPath',
    'get_momentum_transformer',
    'get_standalone_lstm',
    
    # Ensemble
    'RegimeFeatureExtractor',
    'DynamicWeightNetwork',
    'EnsembleMomentumTransformer',
    'get_ensemble_model',
    
    # Online Selector
    'PerformanceMetrics',
    'PerformanceTracker',
    'OnlineModelSelector',
    'HybridSelector',
    
    # Losses
    'sharpe_ratio_loss',
    'sortino_ratio_loss',
    'calmar_ratio_loss',
    'sharpe_with_turnover_penalty',
    'maximum_drawdown_loss',
    'combined_loss',
    'get_loss_function',
]


def quick_start_vanilla():
    """
    Quick start: Create vanilla model (paper's approach)
    """
    config = get_paper_config()
    config.model.input_dim = 32  # Set to your number of features
    return get_momentum_transformer(config.model, model_type='simple')


def quick_start_enhanced():
    """
    Quick start: Create enhanced model with LSTM attention
    """
    config = get_enhanced_config()
    config.model.input_dim = 32  # Set to your number of features
    return get_momentum_transformer(config.model, model_type='enhanced')


def quick_start_ensemble():
    """
    Quick start: Create ensemble model (recommended for production)
    """
    config = get_production_config()
    config.model.input_dim = 32  # Set to your number of features
    return get_ensemble_model(config)


def quick_start_hybrid(ensemble, vanilla, attention):
    """
    Quick start: Create hybrid selector with ensemble + monitoring
    
    Args:
        ensemble: Trained ensemble model
        vanilla: Trained vanilla model
        attention: Trained attention model
    
    Returns:
        HybridSelector instance
    """
    return HybridSelector(
        ensemble_model=ensemble,
        vanilla_model=vanilla,
        attention_model=attention,
        lookback_window=50,
        switch_threshold=0.3
    )


# Convenience functions
__all__.extend([
    'quick_start_vanilla',
    'quick_start_enhanced',
    'quick_start_ensemble',
    'quick_start_hybrid'
])


if __name__ == "__main__":
    print("Momentum Transformer Package")
    print(f"Version: {__version__}")
    print("\nQuick Start Examples:")
    print("1. Vanilla model: model = quick_start_vanilla()")
    print("2. Enhanced model: model = quick_start_enhanced()")
    print("3. Ensemble model: model = quick_start_ensemble()")
    print("4. See examples.py for complete usage")
