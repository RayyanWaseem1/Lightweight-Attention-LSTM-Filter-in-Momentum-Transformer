"""
Integration Test - Complete Pipeline

Tests all components working together:
- Feature engineering
- Dataset creation
- Model training
- Regime detection
- Ensemble model
- Performance monitoring
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 80)
print("MOMENTUM TRANSFORMER - COMPLETE INTEGRATION TEST")
print("=" * 80)

# ============================================================================
# 1. FEATURE ENGINEERING
# ============================================================================
print("\n[1/7] Testing Feature Engineering...")

from Utils.Feature_engineering import create_features_from_ohlcv, normalize_features

# Create synthetic OHLCV data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=2000, freq='D')
returns = np.random.randn(2000) * 0.02 + 0.0005
prices = 100 * np.exp(np.cumsum(returns))

df = pd.DataFrame({
    'close': prices,
    'open': prices * (1 + np.random.randn(2000) * 0.005),
    'high': prices * (1 + np.abs(np.random.randn(2000)) * 0.01),
    'low': prices * (1 - np.abs(np.random.randn(2000)) * 0.01),
    'volume': np.random.randint(1000000, 10000000, 2000)
}, index=dates)

# Engineer features
features_df = create_features_from_ohlcv(df, minimal=True)
features_df = normalize_features(features_df, method='zscore', window=252)

print(f"✓ Created {features_df.shape[1]} features from OHLCV data")
print(f"✓ Feature names: {list(features_df.columns)}")

# ============================================================================
# 2. DATASET CREATION
# ============================================================================
print("\n[2/7] Testing Dataset Creation...")

from data.Dataset import create_train_val_test_split, create_dataloaders

features = features_df.values
returns = features_df['return'].values

train_ds, val_ds, test_ds = create_train_val_test_split(
    features=features,
    returns=returns,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    sequence_length=252
)

train_loader, val_loader, test_loader = create_dataloaders(
    train_ds, val_ds, test_ds,
    batch_size=32,
    num_workers=0
)

print(f"✓ Train samples: {len(train_ds)}")
print(f"✓ Val samples: {len(val_ds)}")
print(f"✓ Test samples: {len(test_ds)}")

# ============================================================================
# 3. MODEL CREATION
# ============================================================================
print("\n[3/7] Testing Model Creation...")

from Models.config import get_production_config
from Models.Momentum_transformer import get_momentum_transformer

config = get_production_config()
config.model.input_dim = features.shape[1]
config.model.hidden_dim = 32  # Small for fast test
config.training.num_epochs = 5  # Few epochs for test

# Create all model types
vanilla = get_momentum_transformer(config.model, 'simple')
dual_path = get_momentum_transformer(config.model, 'enhanced')

print(f"✓ Created vanilla model: {sum(p.numel() for p in vanilla.parameters()):,} parameters")
print(f"✓ Created dual-path model: {sum(p.numel() for p in dual_path.parameters()):,} parameters")

# ============================================================================
# 4. TRAINING
# ============================================================================
print("\n[4/7] Testing Training Pipeline...")

from Utils.Losses import SharpeRatioLoss
from Utils.Training import train_model, EarlyStopping, compute_metrics

criterion = SharpeRatioLoss()
optimizer = torch.optim.AdamW(vanilla.parameters(), lr=config.training.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
early_stopping = EarlyStopping(patience=3, mode='min')

print("Training vanilla model...")
tracker = train_model(
    model=vanilla,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=config.training.num_epochs,
    early_stopping=early_stopping,
    gradient_clip_norm=1.0,
    verbose=False
)

summary = tracker.get_summary()
print(f"✓ Training complete: {summary['total_epochs']} epochs")
print(f"✓ Best val Sharpe: {summary['best_val_sharpe']:.4f}")

# ============================================================================
# 5. REGIME DETECTION
# ============================================================================
print("\n[5/7] Testing Regime Detection...")

from Utils.Regime_detector import StatisticalRegimeDetector, RegimeFeatureExtractor

# Test statistical detector
detector = StatisticalRegimeDetector()
regime_metrics = detector.detect_regime(returns[-100:])

print(f"✓ Current regime: {regime_metrics.regime.value}")
print(f"✓ Confidence: {regime_metrics.confidence:.2f}")
print(f"✓ Volatility: {regime_metrics.volatility:.4f}")

# Test PyTorch feature extractor
regime_extractor = RegimeFeatureExtractor()
x_sample = torch.randn(4, 252, features.shape[1])
regime_features = regime_extractor(x_sample)
print(f"✓ Extracted {regime_features.shape[1]} regime features")

# ============================================================================
# 6. ENSEMBLE MODEL
# ============================================================================
print("\n[6/7] Testing Ensemble Model...")

from Models.Ensemble_model import get_ensemble_model

ensemble = get_ensemble_model(config)

# Quick forward pass
x_test = torch.randn(16, 252, config.model.input_dim)
position, metadata = ensemble(x_test, return_components=True)

print(f"✓ Ensemble prediction shape: {position.shape}")
print(f"✓ Mean vanilla weight: {metadata['mean_vanilla_weight']:.4f}")
print(f"✓ Mean attention weight: {metadata['mean_attention_weight']:.4f}")

# ============================================================================
# 7. ONLINE MONITORING
# ============================================================================
print("\n[7/7] Testing Online Performance Monitoring...")

from Models.Online_Selector import OnlineModelSelector

# Create selector
selector = OnlineModelSelector(
    vanilla_model=vanilla,
    attention_model=dual_path,
    lookback_window=20,
    switch_threshold=0.2
)

# Simulate live trading
for i in range(30):
    x, y = test_ds[i]
    x = x.unsqueeze(0)
    
    # Get prediction
    position, metadata = selector.predict(x)
    
    # Update with realized return
    selector.update_performance(y.item())

performance = selector.get_performance_summary()
print(f"✓ Vanilla Sharpe: {performance['vanilla_sharpe']:.4f}")
print(f"✓ Attention Sharpe: {performance['attention_sharpe']:.4f}")
print(f"✓ Active model: {performance['current_model']}")

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("COMPLETE PIPELINE EVALUATION")
print("=" * 80)

# Evaluate on test set
vanilla.eval()
all_positions = []
all_returns = []

with torch.no_grad():
    for x, y in test_loader:
        positions = vanilla(x)
        all_positions.append(positions)
        all_returns.append(y)

all_positions = torch.cat(all_positions)
all_returns = torch.cat(all_returns)

metrics = compute_metrics(all_positions, all_returns)

print("\nTest Set Performance:")
print("-" * 80)
print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:8.4f}")
print(f"Sortino Ratio:       {metrics['sortino_ratio']:8.4f}")
print(f"Max Drawdown:        {metrics['max_drawdown']:8.4f}")
print(f"Win Rate:            {metrics['win_rate']:8.4f}")
print(f"Annualized Return:   {metrics['annualized_return']:8.4f}")
print(f"Volatility:          {metrics['volatility']:8.4f}")
print("-" * 80)

# ============================================================================
# SUCCESS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)

print("\nComponents Tested:")
print("  ✓ Feature Engineering      (utils/feature_engineering.py)")
print("  ✓ Dataset Creation         (data/dataset.py)")
print("  ✓ Model Architecture       (momentum_transformer.py)")
print("  ✓ Training Pipeline        (utils/training.py)")
print("  ✓ Loss Functions           (utils/losses.py)")
print("  ✓ Regime Detection         (utils/regime_detector.py)")
print("  ✓ Ensemble Model           (ensemble_model.py)")
print("  ✓ Online Monitoring        (online_selector.py)")

print("\n" + "=" * 80)
print("INTEGRATION TEST COMPLETE - All Components Working Together!")
print("=" * 80)

print("\nNext Steps:")
print("1. Replace synthetic data with your actual data")
print("2. Tune hyperparameters on validation set")
print("3. Backtest on multiple time periods")
print("4. Deploy with online monitoring")

print("\nFor more information:")
print("  - See COMPLETE_README.md for full documentation")
print("  - See examples.py for detailed usage examples")
print("  - See ARCHITECTURE_GUIDE.md for design decisions")

print("\n" + "=" * 80)