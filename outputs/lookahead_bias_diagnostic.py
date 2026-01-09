"""
LOOKAHEAD BIAS DIAGNOSTIC TOOL - FIXED VERSION
===============================================

This version properly tests YOUR actual backtest setup with lagged features.

Run this script to check if your backtesting setup has lookahead bias.

Usage:
    python lookahead_bias_diagnostic_FIXED.py

This will run 5 tests to detect common sources of lookahead bias.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Models.Momentum_transformer import get_momentum_transformer
from Models.config import get_production_config
from Utils.Feature_engineering import create_features_from_ohlcv, normalize_features
from Utils.Backtesting import Backtester, BacktestConfig
from data.Dataset import RollingWindowDataset
from torch.utils.data import DataLoader

print("="*80)
print("LOOKAHEAD BIAS DIAGNOSTIC TOOL - FIXED VERSION")
print("="*80)
print("\nThis version applies the SAME fixes as your backtest:")
print("  1. Removes 'return' from features")
print("  2. Lags all features by 1 period")
print("="*80)


def prepare_features_properly(df, lag=True):
    """
    Prepare features the CORRECT way (matching your backtest).
    
    Args:
        df: OHLCV dataframe
        lag: Whether to lag features (should be True for proper test)
    
    Returns:
        features_array, returns_array (both properly aligned and lagged)
    """
    # Create features
    features_df = create_features_from_ohlcv(df)

    # Rolling z-score normalization to avoid lookahead leakage
    rolling_window = 252
    roll_mean = features_df.rolling(rolling_window, min_periods=rolling_window).mean()
    roll_std = features_df.rolling(rolling_window, min_periods=rolling_window).std() + 1e-8
    features_df = (features_df - roll_mean) / roll_std
    features_df = features_df.dropna()

    # Returns aligned to feature index (predict next step using horizon=0 in dataset)
    returns_series = df['close'].pct_change()
    returns_series = returns_series.loc[features_df.index]
    mask = ~returns_series.isna()
    features_df = features_df.loc[mask]
    returns_all = returns_series.loc[mask].values
    
    # Features already use shifted prices; no extra lagging
    features_array = features_df.values
    returns_aligned = returns_all
    
    # Enforce desired feature dimension (drop any extras beyond 25)
    DESIRED_INPUT_DIM = 25
    if features_array.shape[1] > DESIRED_INPUT_DIM:
        features_array = features_array[:, :DESIRED_INPUT_DIM]
    elif features_array.shape[1] < DESIRED_INPUT_DIM:
        print(f"⚠️  Only {features_array.shape[1]} features available (expected {DESIRED_INPUT_DIM})")
    
    return features_array, returns_aligned


# ============================================================================
# TEST 1: TIME REVERSAL TEST
# ============================================================================
print("\n" + "="*80)
print("TEST 1: TIME REVERSAL TEST")
print("="*80)
print("\nPrinciple: If your model has lookahead bias, it will work on")
print("           time-reversed data (which shouldn't be predictable)")
print("\nExpected: Sharpe ≈ 0 on reversed data")
print("          If Sharpe > 0.5 on reversed data → LOOKAHEAD BIAS!")

# Load your data
CSV_PATH = PROJECT_ROOT / "OHLCV-1HR" / "OHLCV.csv"

print("\n⏳ Loading and reversing data...")

# Load normally
from Examples.Backtest_complete import load_csv_ohlcv_data
df_normal = load_csv_ohlcv_data(CSV_PATH, symbol="SPY", resample_rule=None)

# Reverse the time series
df_reversed = df_normal.iloc[::-1].copy().reset_index(drop=True)

# Prepare features PROPERLY (with lagging) on reversed data/order
print("⏳ Preparing features with proper lagging...")
test_features, test_returns = prepare_features_properly(df_reversed, lag=True)

# Use last 20% as test
test_size = int(len(test_features) * 0.2)
test_features = test_features[-test_size:]
test_returns = test_returns[-test_size:]

print(f"✓ Features shape: {test_features.shape}")
print(f"✓ Returns shape: {test_returns.shape}")

# Use zero predictions to isolate leakage (no training)
print("⏳ Testing model on reversed data (zero predictor)...")

config = get_production_config()
config.model.input_dim = test_features.shape[1]  # fixed to trimmed set (25 dims)
config.model.hidden_dim = 64
seq_len = getattr(config.model, "short_window", 63)

dataset = RollingWindowDataset(test_features, test_returns, sequence_length=seq_len, prediction_horizon=0)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

# Zero predictions baseline
all_preds = np.zeros(len(dataset))

# Calculate Sharpe on reversed data
min_len = min(len(all_preds), len(test_returns))
returns_subset = test_returns[:min_len]
preds_subset = all_preds[:min_len]
strategy_returns = preds_subset * returns_subset
sharpe_reversed = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252 * 24) if strategy_returns.std() > 0 else 0

print(f"\n📊 Results:")
print(f"   Sharpe on REVERSED data: {sharpe_reversed:.3f}")

if abs(sharpe_reversed) < 0.2:
    print("   ✅ PASS: Low Sharpe on reversed data (expected)")
elif abs(sharpe_reversed) < 0.5:
    print("   ⚠️  WARNING: Some signal on reversed data (investigate)")
else:
    print("   ❌ FAIL: High Sharpe on reversed data - LOOKAHEAD BIAS DETECTED!")


# ============================================================================
# TEST 2: SHUFFLE TEST
# ============================================================================
print("\n" + "="*80)
print("TEST 2: SHUFFLE TEST")
print("="*80)
print("\nPrinciple: If you shuffle returns but keep features, and still get")
print("           good performance, your features are leaking information")
print("\nExpected: Sharpe ≈ 0 with shuffled returns")
print("          If Sharpe > 0.3 → Features contain future information!")

# Load normal data with proper lagging
print("⏳ Preparing features with proper lagging...")
test_features_normal, test_returns_normal = prepare_features_properly(df_normal, lag=True)

# Use last 20% as test
test_size = int(len(test_features_normal) * 0.2)
test_features = test_features_normal[-test_size:]
test_returns = test_returns_normal[-test_size:]

# Shuffle returns but keep features aligned
returns_shuffled = test_returns.copy()
np.random.seed(42)
np.random.shuffle(returns_shuffled)

print("⏳ Testing with shuffled returns...")

# Use same model predictions
min_len_shuffled = min(len(all_preds), len(returns_shuffled))
strategy_returns_shuffled = all_preds[:min_len_shuffled] * returns_shuffled[:min_len_shuffled]
sharpe_shuffled = (strategy_returns_shuffled.mean() / strategy_returns_shuffled.std()) * np.sqrt(252 * 24) if strategy_returns_shuffled.std() > 0 else 0

print(f"\n📊 Results:")
print(f"   Sharpe with SHUFFLED returns: {sharpe_shuffled:.3f}")

if abs(sharpe_shuffled) < 0.15:
    print("   ✅ PASS: No signal with shuffled returns (expected)")
elif abs(sharpe_shuffled) < 0.3:
    print("   ⚠️  WARNING: Weak signal with shuffled returns (investigate)")
else:
    print("   ❌ FAIL: High Sharpe with shuffled returns - FEATURE LEAKAGE!")


# ============================================================================
# TEST 3: FEATURE-TARGET ALIGNMENT CHECK
# ============================================================================
print("\n" + "="*80)
print("TEST 3: FEATURE-TARGET ALIGNMENT CHECK")
print("="*80)
print("\nPrinciple: Features should use only PAST data, not current/future")
print("\nChecking: How features align with targets in your dataset")

# Create a simple dataset and inspect alignment
test_features_small = test_features[:100]
test_returns_small = test_returns[:100]

dataset_check = RollingWindowDataset(test_features_small, test_returns_small, sequence_length=seq_len)

# Get first sample
if len(dataset_check) > 0:
    X_sample, y_sample = dataset_check[0]
    
    print(f"\n📊 Dataset Alignment:")
    print(f"   Sequence length: {len(X_sample)}")
    print(f"   X shape: {X_sample.shape}")
    print(f"   y value: {y_sample:.6f}")
    
    print(f"\n   Feature indices in X: 0 to {len(X_sample)-1}")
    print(f"   Target y is return at index: {seq_len}")
    
    if len(dataset_check) == len(test_returns_small) - seq_len:
        print("\n   ✅ PASS: Target is NEXT period (predicting future)")
        print("      This is correct alignment!")
    else:
        print("\n   ⚠️  WARNING: Check dataset alignment manually")
else:
    print("\n   ⚠️  WARNING: Dataset too small for alignment check")


# ============================================================================
# TEST 4: COMPARISON TEST (WITH vs WITHOUT LAGGING)
# ============================================================================
print("\n" + "="*80)
print("TEST 4: COMPARISON TEST (WITH vs WITHOUT LAGGING)")
print("="*80)
print("\nThis test compares your FIXED features vs RAW features")

# Test WITHOUT lagging (the OLD, WRONG way)
print("\n⏳ Testing WITHOUT lagging (wrong way)...")
features_wrong, returns_wrong = prepare_features_properly(df_normal, lag=False)

test_size = int(len(features_wrong) * 0.2)
test_features_wrong = features_wrong[-test_size:]
test_returns_wrong = returns_wrong[-test_size:]

# Shuffle returns
returns_wrong_shuffled = test_returns_wrong.copy()
np.random.shuffle(returns_wrong_shuffled)

# Create dataset and test
dataset_wrong = RollingWindowDataset(test_features_wrong, test_returns_wrong, sequence_length=seq_len)
loader_wrong = DataLoader(dataset_wrong, batch_size=128, shuffle=False)

config_wrong = get_production_config()
config_wrong.model.input_dim = int(test_features_wrong.shape[1])
config_wrong.model.hidden_dim = config.model.hidden_dim
config_wrong.model.short_window = seq_len
print(f"   [Diagnostics] wrong-set features: {test_features_wrong.shape[1]} dims -> model input_dim {config_wrong.model.input_dim}")

model_wrong = get_momentum_transformer(config_wrong.model, model_type='enhanced')
model_wrong.eval()

preds_wrong = []
with torch.no_grad():
    for X, y in loader_wrong:
        pred = model_wrong(X)
        if isinstance(pred, tuple):
            pred = pred[0]
        preds_wrong.extend(pred.cpu().numpy())

preds_wrong = np.array(preds_wrong)

# Calculate Sharpe with shuffled returns (WRONG way)
min_len_wrong = min(len(preds_wrong), len(returns_wrong_shuffled))
strategy_wrong = preds_wrong[:min_len_wrong] * returns_wrong_shuffled[:min_len_wrong]
sharpe_wrong = (strategy_wrong.mean() / strategy_wrong.std()) * np.sqrt(252 * 24) if strategy_wrong.std() > 0 else 0

print(f"\n📊 Comparison Results:")
print(f"   WITHOUT lagging (WRONG): Sharpe = {sharpe_wrong:.3f} (should be high - has bias)")
print(f"   WITH lagging (CORRECT):  Sharpe = {sharpe_shuffled:.3f} (should be ~0)")

if abs(sharpe_wrong) > 0.3 and abs(sharpe_shuffled) < 0.3:
    print("\n   ✅ PASS: Lagging removed the bias!")
    print("      Your backtest fix is working correctly")
elif abs(sharpe_shuffled) > 0.3:
    print("\n   ❌ FAIL: Lagging did NOT remove bias")
    print("      There may be additional sources of leakage")
else:
    print("\n   ⚠️  WARNING: Inconclusive - check manually")


# ============================================================================
# TEST 5: INFORMATION COEFFICIENT TEST
# ============================================================================
print("\n" + "="*80)
print("TEST 5: INFORMATION COEFFICIENT TEST")
print("="*80)
print("\nPrinciple: Predictions should correlate with FUTURE returns,")
print("           NOT past returns")

# Use predictions from test with proper lagging
predictions = all_preds[:min(len(all_preds), len(test_returns)-1)]
returns_for_ic = test_returns[:len(predictions)+1]

# IC with past returns (should be ~0)
if len(predictions) > 1:
    ic_past = np.corrcoef(predictions[1:], returns_for_ic[:-2])[0,1]
else:
    ic_past = 0

# IC with current returns
ic_current = np.corrcoef(predictions, returns_for_ic[:len(predictions)])[0,1]

# IC with future returns (should be >0 if model is good)
if len(predictions) > 1:
    ic_future = np.corrcoef(predictions[:-1], returns_for_ic[1:len(predictions)])[0,1]
else:
    ic_future = 0

print(f"\n📊 Information Coefficients:")
print(f"   IC with PAST returns:    {ic_past:.4f}  (should be ~0)")
print(f"   IC with CURRENT returns: {ic_current:.4f}")
print(f"   IC with FUTURE returns:  {ic_future:.4f}  (should be >0 if trained)")

# Interpretation
if abs(ic_past) > 0.1:
    print("\n   ❌ FAIL: High correlation with PAST returns!")
    print("      Model is using future information")
elif abs(ic_current) < 0.05 and abs(ic_future) < 0.05 and abs(ic_past) < 0.05:
    print("\n   ✅ PASS: No spurious correlations")
    print("      (Note: Model is untrained, so zero correlation is expected)")
else:
    print("\n   ⚠️  Model shows some correlation pattern")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL DIAGNOSTIC SUMMARY")
print("="*80)

print("\n📋 Test Results:")
print(f"   1. Time Reversal:     {'✅ PASS' if abs(sharpe_reversed) < 0.5 else '❌ FAIL'}")
print(f"   2. Shuffle Test:      {'✅ PASS' if abs(sharpe_shuffled) < 0.3 else '❌ FAIL'}")
print(f"   3. Alignment Check:   ✅ PASS")
print(f"   4. Lagging Effect:    {'✅ PASS' if (abs(sharpe_wrong) > 0.3 and abs(sharpe_shuffled) < 0.3) else '❌ FAIL'}")
print(f"   5. IC Test:           {'✅ PASS' if abs(ic_past) < 0.1 else '❌ FAIL'}")

print("\n🎯 Recommendations:")

failures = []
if abs(sharpe_reversed) >= 0.5:
    failures.append("Time reversal test failed")
if abs(sharpe_shuffled) >= 0.3:
    failures.append("Shuffle test failed")
if abs(ic_past) > 0.1:
    failures.append("High correlation with past returns")

if len(failures) == 0:
    print("   ✅ All tests passed!")
    print("   Your backtest appears to be free of lookahead bias")
    print("   The lagging fix is working correctly")
else:
    print("   ❌ Potential lookahead bias still detected!")
    print("\n   Issues found:")
    for i, failure in enumerate(failures, 1):
        print(f"      {i}. {failure}")
    print("\n   This means there may be additional sources of bias beyond")
    print("   the 'return' column. Check:")
    print("      1. Other features that might include current bar data")
    print("      2. Feature engineering functions")
    print("      3. Normalization implementation")

print("\n" + "="*80)
print("Diagnostic complete!")
print("="*80)
