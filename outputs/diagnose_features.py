"""
Diagnostic Script: Check Cross-Sectional Feature Variation
Run this to see if relative_vol and other features are actually varying
"""

import os
import sys

import torch
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running from outputs/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Models.Ensemble_model import RegimeFeatureExtractor

# Create some test data
print("Testing RegimeFeatureExtractor feature variation...")
print("=" * 80)

extractor = RegimeFeatureExtractor()

# Simulate different scenarios
batch_size = 10
seq_len = 252
input_dim = 53

# Create test data with varying volatility
test_data = []
for i in range(batch_size):
    # Create returns with different volatility levels
    vol_level = 0.01 + (i * 0.005)  # 0.01 to 0.055
    returns = torch.randn(seq_len) * vol_level
    
    # Add other features (simplified)
    features = torch.randn(seq_len, input_dim)
    features[:, 0] = returns  # First column is returns
    
    test_data.append(features)

x = torch.stack(test_data)  # [batch, seq_len, input_dim]

# Create market data
market_returns = torch.randn(seq_len) * 0.02  # Market with 2% volatility

market_data = {
    'market_returns': market_returns,
    'all_stock_returns': x[:, :, 0]
}

# Extract features
print("Extracting features WITH market_data...")
features_with_market = extractor(x, market_data=market_data)

print("\n" + "=" * 80)
print("FEATURE VARIATION ANALYSIS")
print("=" * 80)

feature_names = [
    'high_vol', 'med_vol', 'low_vol', 'trend_strength', 'vol_ratio',
    'mean_returns', 'abs_mean_returns', 'return_range', 'skewness', 'momentum_short',
    'rsi_current', 'vol_feature_mean', 'momentum_feature_current',
    'relative_return', 'relative_vol', 'beta', 'relative_strength', 'market_vol_level',
    'vol_persistence', 'vol_trend', 'regime_shift'
]

for i, name in enumerate(feature_names):
    values = features_with_market[:, i].numpy()
    print(f"{i:2d}. {name:25s} | Range: [{values.min():7.3f}, {values.max():7.3f}] | Std: {values.std():.3f}")

print("\n" + "=" * 80)
print("KEY FEATURES FOR REGIME DETECTION:")
print("=" * 80)

# Check the critical cross-sectional features
relative_vol_idx = 14
beta_idx = 15
market_vol_idx = 17

print(f"\nrelative_vol (index {relative_vol_idx}):")
print(f"  Range: {features_with_market[:, relative_vol_idx].min():.3f} - {features_with_market[:, relative_vol_idx].max():.3f}")
print(f"  Should range from ~0.7 (low vol stock) to ~2.5 (crisis)")
print(f"  All values: {features_with_market[:, relative_vol_idx].numpy()}")

print(f"\nbeta (index {beta_idx}):")
print(f"  Range: {features_with_market[:, beta_idx].min():.3f} - {features_with_market[:, beta_idx].max():.3f}")
print(f"  Should range from ~0.5 (independent) to ~1.8 (high correlation)")
print(f"  All values: {features_with_market[:, beta_idx].numpy()}")

print(f"\nmarket_vol_level (index {market_vol_idx}):")
print(f"  Range: {features_with_market[:, market_vol_idx].min():.3f} - {features_with_market[:, market_vol_idx].max():.3f}")
print(f"  Should vary: 0.01 (calm) to 0.09 (crisis)")

print("\n" + "=" * 80)
print("TEST WITHOUT market_data (should show defaults):")
print("=" * 80)

features_without_market = extractor(x, market_data=None)

print(f"\nrelative_vol WITHOUT market_data:")
print(f"  Values: {features_without_market[:, relative_vol_idx].numpy()}")
print(f"  Expected: All 1.0 (default)")

print(f"\nbeta WITHOUT market_data:")
print(f"  Values: {features_without_market[:, beta_idx].numpy()}")
print(f"  Expected: All 1.0 (default)")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

# Check if features are actually varying
relative_vol_std = features_with_market[:, relative_vol_idx].std()
beta_std = features_with_market[:, beta_idx].std()

if relative_vol_std < 0.1:
    print("\n⚠️  WARNING: relative_vol has LOW VARIATION (std < 0.1)")
    print("   This means stocks look identical to the market")
    print("   Model cannot distinguish regimes!")
else:
    print(f"\n✓ relative_vol varies well (std = {relative_vol_std:.3f})")

if beta_std < 0.1:
    print("\n⚠️  WARNING: beta has LOW VARIATION (std < 0.1)")
    print("   This means all stocks have similar market correlation")
else:
    print(f"\n✓ beta varies well (std = {beta_std:.3f})")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("""
If you see warnings above:
1. Check that market_returns is actually being passed during training/inference
2. Verify market_returns has correct shape and values
3. Add debug prints in RegimeFeatureExtractor to see actual market_returns values
4. Check if all your stocks have similar volatility characteristics

If features ARE varying but ensemble still collapses:
1. Increase weight_hidden_dim from 32 to 64 or 128
2. Adjust min/max weights from (0.2, 0.8) to (0.1, 0.9)
3. Add L1 regularization to encourage weight diversity
4. Try different random seeds
""")
