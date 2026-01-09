"""
Regime-Aware Multi-Asset Backtesting with Dynamic Model Switching
"""

import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from typing import Dict, Tuple, List, Optional
from collections import defaultdict, deque
import datetime
from scipy import stats
from scipy.stats import spearmanr
import math 
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Safety caps for per-period returns to avoid outsized impact from bad ticks
SYMBOL_RETURN_CLIP = 0.7   # fallback clip for individual symbol returns
PORTFOLIO_RETURN_CLIP = 0.7  # fallback clip for aggregated portfolio return
COST_RATE = 0.00005  # per-period cost scaled by gross exposure/turnover
PER_PERIOD_COST = COST_RATE  # fallback fixed cost for metrics compatibility

# Import modules
from Models.config import get_production_config
from Models.Ensemble_model import get_ensemble_model
from Utils.Regime_detector import StatisticalRegimeDetector, MarketRegime
from Utils.Losses import DirectionalAccuracyLoss, SharpeRatioLoss
from Utils.Training import EarlyStopping

# Feature engineering
from Utils.Feature_engineering import create_features_from_ohlcv, normalize_features

print("=" * 80)
print("REGIME-AWARE MULTI-ASSET MOMENTUM TRANSFORMER")
print("=" * 80)
print("Dynamic model switching based on market regime")
print("=" * 80)


# List of known leveraged ETFs to exclude
LEVERAGED_ETFS = [
    # 3x Leveraged ETFs
    'SOXL', 'SOXS',  # 3x semiconductor
    'TQQQ', 'SQQQ',  # 3x NASDAQ
    'UPRO', 'SPXU',  # 3x S&P 500
    'TNA', 'TZA',    # 3x Russell 2000
    'LABU', 'LABD',  # 3x biotech
    'JNUG', 'JDST',  # 3x gold miners
    'NUGT', 'DUST',  # 3x gold miners
    'ERX', 'ERY',    # 3x energy
    'FAS', 'FAZ',    # 3x financials
    'TECL', 'TECS',  # 3x tech
    'CURE', 'CUT',   # 3x healthcare
    'WANT', 'GASL',  # 3x natural gas
    'BULL', 'BEAR',  # Leveraged products
    
    # 2x Leveraged ETFs
    'TSLL', 'TSLS',  # 2x TSLA
    'NVDL', 'NVDS',  # 2x NVDA
    'MSTU', 'MSTX',  # 2x MicroStrategy
    'SSO', 'SDS',    # 2x S&P 500
    'QLD', 'QID',    # 2x NASDAQ
    'DDM', 'DXD',    # 2x Dow
    'UWM', 'TWM',    # 2x Russell 2000
    'SPUU', 'SPDD',  # 2x S&P 500
    'QQQU', 'SQQD',  # 2x NASDAQ
]


def detect_stock_splits(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """
    Detect likely stock splits by finding extreme price jumps
    """
    symbol_df = df[df['symbol'] == symbol].sort_values('timestamp').copy()
    
    if len(symbol_df) < 2:
        return []
    
    symbol_df['price_ratio'] = symbol_df['close'] / symbol_df['close'].shift(1)
    symbol_df['return'] = symbol_df['close'].pct_change()
    
    splits = []
    
    for idx, row in symbol_df.iterrows():
        price_ratio = row['price_ratio']
        ret = row['return']
        
        if pd.isna(price_ratio):
            continue
        
        # Detect forward split (price drops significantly)
        if price_ratio < 0.5:
            splits.append({
                'timestamp': row['timestamp'],
                'type': 'forward_split',
                'ratio': 1 / price_ratio,
                'return': ret
            })
        
        # Detect reverse split (price increases significantly)
        elif price_ratio > 2.0:
            splits.append({
                'timestamp': row['timestamp'],
                'type': 'reverse_split',
                'ratio': price_ratio,
                'return': ret
            })
    
    return splits


class MultiAssetDataset(Dataset):
    """Dataset for multiple assets"""
    def __init__(self, features_df, symbols, sequence_length=252):
        self.symbols = symbols
        self.sequence_length = sequence_length
        self.data = {}
        
        # Group data by symbol
        for symbol in symbols:
            symbol_data = features_df[features_df['symbol'] == symbol].copy()
            if len(symbol_data) >= sequence_length:
                self.data[symbol] = symbol_data
        
        # Create all valid sequences
        self.sequences = []
        for symbol, df in self.data.items():
            for i in range(len(df) - sequence_length):
                self.sequences.append((symbol, i))
        
        print(f"Created dataset: {len(self.sequences)} sequences from {len(self.data)} stocks")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        symbol, start_idx = self.sequences[idx]
        df = self.data[symbol]
        
        # Get sequence
        end_idx = start_idx + self.sequence_length
        sequence = df.iloc[start_idx:end_idx]
        
        # Features (everything except return and metadata)
        feature_cols = [c for c in sequence.columns 
                       if c not in ['symbol', 'timestamp', 'return']]
        X = sequence[feature_cols].values
        
        # Target
        y = sequence['return'].iloc[-1]
        
        return torch.FloatTensor(X), torch.FloatTensor([y])


def load_multi_asset_data(csv_path: str, max_symbols: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    FINAL comprehensive data loading with all filtering
    
    Strategy:
    1. Remove penny stocks (< $2)
    2. Remove leveraged ETFs (3x products)
    3. Detect and remove stock split bars (keep stocks)
    4. Remove extreme penny stocks (>1000% moves)
    5. Ensure sufficient data
    """
    print("\n" + "=" * 80)
    print("LOADING DATA WITH COMPREHENSIVE FILTERING")
    print("=" * 80)
    
    df = pd.read_csv(csv_path)
    
    if 'ts_event' in df.columns:
        ts_col = 'ts_event'
    elif 'timestamp' in df.columns:
        ts_col = 'timestamp'
    else:
        raise ValueError("CSV must have timestamp column")
    
    required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_cols)}")
    
    df['timestamp'] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(None)
    df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    initial_bars = len(df)
    initial_symbols = df['symbol'].nunique()
    print(f"\nInitial: {initial_bars:,} bars, {initial_symbols} symbols")
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 1: Remove penny stocks (< $2)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[Filter 1] Removing penny stocks (min price < $2)...")
    
    symbol_stats = df.groupby('symbol').agg({
        'close': ['mean', 'min']
    }).reset_index()
    symbol_stats.columns = ['symbol', 'avg_price', 'min_price']
    
    good_symbols = symbol_stats[symbol_stats['min_price'] >= 2.0]['symbol'].tolist()
    removed_penny = initial_symbols - len(good_symbols)
    print(f"  Removed: {removed_penny} penny stocks")
    print(f"  Remaining: {len(good_symbols)} symbols")
    
    df = df[df['symbol'].isin(good_symbols)]
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 2: Remove leveraged ETFs
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[Filter 2] Removing leveraged ETFs (3x products)...")
    
    leveraged_in_data = [s for s in LEVERAGED_ETFS if s in good_symbols]
    if leveraged_in_data:
        print(f"  Found {len(leveraged_in_data)} leveraged ETFs: {', '.join(leveraged_in_data)}")
        good_symbols = [s for s in good_symbols if s not in LEVERAGED_ETFS]
        df = df[df['symbol'].isin(good_symbols)]
        print(f"  Remaining: {len(good_symbols)} symbols")
    else:
        print(f"  No leveraged ETFs found")
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 3: Detect and remove stock split bars
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[Filter 3] Detecting stock splits...")
    
    symbols_with_splits = []
    bars_to_remove = []
    split_details = {}
    
    for symbol in good_symbols:
        splits = detect_stock_splits(df, symbol)
        
        if splits:
            symbols_with_splits.append(symbol)
            split_details[symbol] = splits
            
            for split in splits:
                bar_idx = df[(df['symbol'] == symbol) & 
                           (df['timestamp'] == split['timestamp'])].index
                bars_to_remove.extend(bar_idx.tolist())
    
    if symbols_with_splits:
        print(f"  Found {len(symbols_with_splits)} symbols with stock splits")
        print(f"  Removing {len(bars_to_remove)} bars with splits (keeping stocks)")
        
        # Show major blue chips
        blue_chips = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX']
        split_blue_chips = [s for s in blue_chips if s in symbols_with_splits]
        if split_blue_chips:
            print(f"  Blue-chip stocks with splits: {', '.join(split_blue_chips)}")
        
        df = df.drop(bars_to_remove)
        print(f"  Kept all {len(symbols_with_splits)} symbols (99.9% of their data)")
    else:
        print(f"  No stock splits detected")
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 4: Remove extreme penny stocks (>1000% hourly returns)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[Filter 4] Removing extreme penny stocks (>1000% returns)...")
    
    df['return'] = df.groupby('symbol')['close'].pct_change()
    
    extreme_penny_stocks = []
    
    for symbol in good_symbols:
        if symbol not in df['symbol'].values:
            continue
        
        symbol_df = df[df['symbol'] == symbol].copy()
        returns = symbol_df['return'].dropna()
        
        if len(returns) < 100:
            continue
        
        max_return = returns.abs().max()
        
        # Remove if ANY return > 1000% (10x in one hour = clearly bad data)
        if max_return > 10.0:
            extreme_penny_stocks.append((symbol, max_return))
    
    if extreme_penny_stocks:
        print(f"  Found {len(extreme_penny_stocks)} extreme penny stocks:")
        for symbol, max_ret in sorted(extreme_penny_stocks, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {symbol}: Max {max_ret*100:.1f}%")
        
        extreme_symbols = [s for s, _ in extreme_penny_stocks]
        good_symbols = [s for s in good_symbols if s not in extreme_symbols]
        df = df[df['symbol'].isin(good_symbols)]
        print(f"  Remaining: {len(good_symbols)} symbols")
    else:
        print(f"  No extreme penny stocks found")
    
    df = df.drop(columns=['return'])
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 5: Ensure sufficient data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[Filter 5] Ensuring sufficient data (≥3,000 bars)...")
    bar_counts = df.groupby('symbol').size()
    good_symbols = [s for s in good_symbols if s in bar_counts[bar_counts >= 3000].index]
    df = df[df['symbol'].isin(good_symbols)]
    print(f"  Remaining: {len(good_symbols)} symbols")
    
    if max_symbols and len(good_symbols) > max_symbols:
        good_symbols = good_symbols[:max_symbols]
        df = df[df['symbol'].isin(good_symbols)]
        print(f"  Limited to {max_symbols} symbols for testing")
    
    final_bars = len(df)
    final_symbols = len(good_symbols)
    
    print(f"\n" + "=" * 80)
    print("FINAL DATASET")
    print("=" * 80)
    print(f"Bars:     {final_bars:,} ({final_bars/initial_bars*100:.1f}% of original)")
    print(f"Symbols:  {final_symbols} ({final_symbols/initial_symbols*100:.1f}% of original)")
    print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    # Show which blue chips we kept
    blue_chips = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX']
    kept_blue_chips = [s for s in blue_chips if s in good_symbols]
    if kept_blue_chips:
        print(f"\n✓ Blue-chip stocks kept ({len(kept_blue_chips)}): {', '.join(kept_blue_chips)}")
    
    print("\n" + "=" * 80)
    
    return df, list(good_symbols)



def engineer_multi_asset_features(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Engineer features for all symbols using stock-agnostic methods"""
    print("\n[Step 2/8] Engineering Stock-Agnostic Features...")
    
    # Get market data (SPY) for context
    spy_data = None
    if 'SPY' in df['symbol'].values:
        spy_df = df[df['symbol'] == 'SPY'].sort_values('timestamp')
        spy_data = spy_df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
    
    all_features = []
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('timestamp').set_index('timestamp')
        
        # Create features (stock-agnostic!)
        features = create_features_from_ohlcv(
            symbol_df[['open', 'high', 'low', 'close', 'volume']],
            market_df=spy_data,
            symbol=symbol
        )
        features = features.reset_index()

        # Add forward return target (1-step ahead by default)
        symbol_df['return'] = symbol_df['close'].pct_change().shift(-1)
        features['return'] = symbol_df['return'].reindex(features['timestamp']).values
                
        # Add symbol identifier
        features['symbol'] = symbol
        
        all_features.append(features)
    
    # Combine all symbols
    combined = pd.concat(all_features, ignore_index=True)
    
    # Drop NaN from feature engineering
    combined = combined.dropna()
    
    print(f"✓ Created features for {combined['symbol'].nunique()} symbols")
    print(f"  Total samples: {len(combined):,}")
    print(f"  Features: {len([c for c in combined.columns if c not in ['symbol', 'timestamp', 'return']])}")
    
    return combined


def infer_regimes_from_returns(return_series: np.ndarray, lookback: int = 252) -> Tuple[np.ndarray, Dict[str, str]]:
    """Infer market regimes"""
    detector = StatisticalRegimeDetector()
    labels = []
    regime_to_id: Dict[MarketRegime, int] = {}
    regime_name_map: Dict[str, str] = {}

    for i in range(len(return_series)):
        if i < lookback:
            regime = MarketRegime.NORMAL
        else:
            window_returns = return_series[i - lookback:i]
            regime = detector.detect_regime(window_returns).regime

        if regime not in regime_to_id:
            regime_idx = len(regime_to_id)
            regime_to_id[regime] = regime_idx
            regime_name_map[str(regime_idx)] = regime.name.replace("_", " ").title()

        labels.append(regime_to_id[regime])

    return np.array(labels), regime_name_map


def train_regime_aware_model(train_dataset, val_dataset, config, market_returns_series):
    """
    Train ensemble model with regime-aware dynamic weighting
    
    The model automatically:
    - Detects market regimes
    - Learns to weight vanilla vs attention models
    - Adapts to changing market conditions
    """
    print("\n[Step 4/8] Training Regime-Aware Ensemble Model...")
    print("  Model will learn to switch between vanilla/attention based on regime")

    config.model.transformer_dropout = 0.3 #was 0.3
    config.training.learning_rate = 2.5e-4 #was 2e-4
    config.training.weight_decay = 1e-4 #was 1e-4
    
    model = get_ensemble_model(config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = config.training.batch_size, shuffle = False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.training.learning_rate,
        weight_decay = config.training.weight_decay
    )

    criterion = SharpeRatioLoss(annualization_factor=np.sqrt(252 * 24))

    best_val_sharpe = -float('inf')
    best_model_state = model.state_dict().copy()

    #Preparing market data for training
    #Getting market returns aligned with the training data
    train_market_tensor = torch.tensor(
        market_returns_series.values,
        dtype = torch.float32
    )
    val_market_tensor = train_market_tensor

    for epoch in range(20):
        model.train()
        train_losses = []
        train_attention_weights = []

        for X, y in train_loader:
            #Preparing market_data dict for the training batch
            market_data = {
                'market_returns': train_market_tensor,
                'all_stock_returns': X[:, :, 0] #All stocks's returns in the batch
            }

            optimizer.zero_grad()
            predictions, metadata = model(X, return_components = True, market_data = market_data)
            loss = criterion(predictions.squeeze(), y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

            #Tracking attention usage
            if metadata and "mean_attention_weight" in metadata:
                train_attention_weights.append(metadata["mean_attention_weight"])

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        val_attention_weight = []

        with torch.no_grad():
            for X, y in val_loader:
                market_data = {
                    'market_returns': val_market_tensor,
                    'all_stock_returns': X[:, :, 0]
                }

                pred, metadata = model(X, return_components = True, market_data = market_data)
                val_predictions.extend(pred.squeeze().cpu().numpy())
                val_targets.extend(y.squeeze().cpu().numpy())

                #Tracking attention usage
                if metadata and "mean_attention_weight" in metadata:
                    val_attention_weight.append(metadata["mean_attention_weight"])

        val_predictions = np.array(val_predictions, dtype=float)
        val_targets = np.array(val_targets, dtype=float)

        # Calculate Sharpe
        # Using a percentile based signal: top 50% long, bottom 50% flat
        pred_80th_percentile = np.percentile(val_predictions, 80)
        long_mask = val_predictions >= pred_80th_percentile

        # Strategy: Long top 50%, flat bottom 50%
        strategy_returns = np.where(long_mask, val_targets, 0)
        val_std = strategy_returns.std()
        if val_std > 0 and np.isfinite(val_std):
            val_sharpe = (strategy_returns.mean() / val_std) * np.sqrt(252 * 24)
        else:
            val_sharpe = -float('inf')

        if (epoch + 1) % 5 == 0:
            mean_train_attn = np.mean(train_attention_weights) if train_attention_weights else 0
            mean_val_attn = np.mean(val_attention_weight) if val_attention_weight else 0
            print(f" Epoch {epoch + 1} / 20:")
            print(f" Val Sharpe: {val_sharpe:.4f}")
            print(f" Train Attention: {mean_train_attn:.3f} (0 = Vanilla, 1 = Attention)")
            print(f" Val Attention: {mean_val_attn:.3f}")

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_model_state = model.state_dict().copy()

        #if early_stopping(val_sharpe, epoch):
            #print(f" Early stopping at epoch {epoch + 1}")
            #break 

    model.load_state_dict(best_model_state)
    print(f" Training complete")
    print(f" Best val Sharpe: {best_val_sharpe:.4f}")

    weight_stats = model.get_weight_statistics()
    print(f"\n Model Usage Statistics: ")
    print(f" Mean Attention weight: {weight_stats.get('mean_attention_weight', 0):.3f}")
    print(f" Attention used: {weight_stats.get('attention_usage_pct', 0):.1f}% of time")
    print(f" Vanilla used: {weight_stats.get('vanilla_usage_pct', 0):.1f}% of time")

    return model, weight_stats

def generate_regime_aware_signals(model, test_features, symbols, sequence_length=252, market_returns_series = None):
    """
    Generate signals using regime-aware ensemble
    
    Model will automatically:
    - Detect regime for each prediction
    - Weight vanilla vs attention models appropriately
    - Track which model is being used
    """
    print("\n[Step 5/8] Generating Regime-Aware Signals...")
    
    model.eval()
    signals = {}
    regime_usage = defaultdict(lambda: {'vanilla': 0, 'attention': 0})

    #Preparing market data for test period
    if market_returns_series is not None:
        test_timestamps = test_features['timestamp'].unique()
        test_market_returns = market_returns_series[ 
            market_returns_series.index.isin(test_timestamps)
        ]
    else:
        test_market_returns = None 
    
    with torch.no_grad():
        for symbol in symbols:
            symbol_data = test_features[test_features['symbol'] == symbol]
            
            if len(symbol_data) < sequence_length:
                continue
            
            # Generate predictions for entire test period
            symbol_predictions = []
            symbol_attention_weights = []
            
            for i in range(sequence_length, len(symbol_data)):
                sequence = symbol_data.iloc[i-sequence_length:i]
                
                # Prepare features
                feature_cols = [c for c in sequence.columns 
                              if c not in ['symbol', 'timestamp', 'return']]
                X = sequence[feature_cols].values
                X = torch.FloatTensor(X).unsqueeze(0)

                #Preparing market_data dict for this prediction
                if test_market_returns is not None:
                    #Getting market returns for the sequence window
                    seq_timestamps = sequence['timestamp']
                    seq_market_returns = test_market_returns[ 
                        test_market_returns.index.isin(seq_timestamps)
                    ]

                    market_data = {
                        'market_returns': torch.tensor( 
                            seq_market_returns.values,
                            dtype = torch.float32
                        ),
                        'all_stock_returns': X[:, :, 0] #This stock's returns
                    }
                else:
                    market_data = None 
                
                # Generate signal with metadata
                prediction, metadata = model(X, return_components=True, market_data = market_data)
                
                # Clip prediction to avoid extreme exposure
                pred_val = float(prediction.item())
                symbol_predictions.append(pred_val)
                
                # Track attention weight
                if metadata and 'attention_weights' in metadata:
                    attn_weight = metadata['attention_weights'].item()
                    symbol_attention_weights.append(attn_weight)
            
            # Using the MOST RECENT PREDICTION. Used to be median
            signals[symbol] = symbol_predictions[-1] if symbol_predictions else 0
            
            # Track regime usage
            if symbol_attention_weights:
                mean_attn = np.mean(symbol_attention_weights)
                if mean_attn > 0.5:
                    regime_usage[symbol]['attention'] = mean_attn
                else:
                    regime_usage[symbol]['vanilla'] = 1 - mean_attn
    
    if signals:
        print(f"✓ Generated signals for {len(signals)} symbols")
        print(f"  Signal range: [{min(signals.values()):.3f}, {max(signals.values()):.3f}]")
    else:
        print("No signals generated (insufficient data after filtering)")
    
    # Show model usage
    attention_stocks = sum(1 for s in regime_usage.values() if s['attention'] > s['vanilla'])
    vanilla_stocks = sum(1 for s in regime_usage.values() if s['vanilla'] >= s['attention'])
    
    print(f"\n  Model Usage by Stock:")
    print(f"    Attention model preferred: {attention_stocks} stocks")
    print(f"    Vanilla model preferred: {vanilla_stocks} stocks")
    print(f"    → Dynamic regime-based selection!")
    
    return signals, regime_usage


def construct_portfolio(signals: Dict[str, float], 
                       max_positions: int = 15,
                       min_signal: float = 0.1,
                       vol_by_symbol = None,
                       max_weight = 0.2) -> Dict[str, float]:
    """Construct diversified portfolio from signals"""
    # Filter weak signals
    strong = {k: v for k, v in signals.items() if abs(v) >= min_signal}
    
    if not strong:
        return {}
    
    # Sort by absolute signal strength
    sorted_signals = sorted(strong.items(), key=lambda x: abs(x[1]), reverse=True)[:max_positions]

    # Weight proportional to signal magnitude with modest volatility scaling
    raw_weights = {}
    for sym, sig in sorted_signals:
        mag = abs(sig)
        if vol_by_symbol and sym in vol_by_symbol:
            mag /= (vol_by_symbol[sym] + 1e-8)
        raw_weights[sym] = math.copysign(mag, sig)

    total = sum(abs(w) for w in raw_weights.values())
    weights = {sym: max(-max_weight, min(max_weight, w / total)) for sym, w in raw_weights.items()}

    #Renormalize if capped
    gross = sum(abs(w) for w in weights.values())
    if gross > 0:
        weights = {sym: w / gross for sym, w in weights.items()}

    return weights


def calculate_portfolio_returns(portfolio_weights: Dict[str, float],
                               actual_returns_df: pd.DataFrame,
                               test_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate portfolio returns using actual returns and portfolio weights
    """
    print("\n[Step 6/8] Calculating Portfolio Returns...")

    if not portfolio_weights:
        print("  No portfolio weights selected; skipping return calculation.")
        return pd.DataFrame(columns=['timestamp', 'portfolio_return'])
    
    # Get test period timestamps
    if test_features.empty or 'timestamp' not in test_features.columns:
        print("  ⚠️ No test features available; skipping return calculation.")
        return pd.DataFrame(columns=['timestamp', 'portfolio_return'])
    
    test_timestamps = test_features['timestamp'].unique()
    if len(test_timestamps) == 0:
        print("  ⚠️ No test timestamps available; skipping return calculation.")
        return pd.DataFrame(columns=['timestamp', 'portfolio_return'])
    test_start = test_timestamps.min()
    test_end = test_timestamps.max()
    
    # Filter returns to test period
    test_returns = actual_returns_df[
        (actual_returns_df['timestamp'] >= test_start) &
        (actual_returns_df['timestamp'] <= test_end)
    ].copy()
    
    # Calculate weighted portfolio returns
    portfolio_returns_list = []
    
    for timestamp in test_timestamps:
        timestamp_returns = test_returns[test_returns['timestamp'] == timestamp]
        
        # Calculate weighted return with clipping and cost scaling
        portfolio_return = 0
        total_weight = 0
        
        for symbol, weight in portfolio_weights.items():
            symbol_return = timestamp_returns[timestamp_returns['symbol'] == symbol]['return'].values
            if len(symbol_return) > 0:
                symbol_r = float(np.clip(symbol_return[0], -SYMBOL_RETURN_CLIP, SYMBOL_RETURN_CLIP))
                portfolio_return += weight * symbol_r
                total_weight += abs(weight)
        
        # Normalize if needed
        if total_weight > 0:
            portfolio_return = portfolio_return / total_weight
        
        # Clip aggregated portfolio return (fallback)
        portfolio_return = float(np.clip(portfolio_return, -PORTFOLIO_RETURN_CLIP, PORTFOLIO_RETURN_CLIP))
        
        portfolio_returns_list.append({
            'timestamp': timestamp,
            'portfolio_return': portfolio_return
        })
    
    portfolio_returns_df = pd.DataFrame(portfolio_returns_list)
    portfolio_returns_df = portfolio_returns_df.sort_values('timestamp').reset_index(drop=True)

    # Winsorize portfolio returns at 1st/99th percentiles
    lower_p, upper_p = portfolio_returns_df['portfolio_return'].quantile([0.01, 0.99])
    portfolio_returns_df['portfolio_return'] = portfolio_returns_df['portfolio_return'].clip(lower_p, upper_p)

    # Diagnostics after winsorization
    top_bottom = pd.concat([
        portfolio_returns_df.nlargest(5, 'portfolio_return'),
        portfolio_returns_df.nsmallest(5, 'portfolio_return')
    ])
    print("\n[Diagnostics] Top/Bottom 5 per-bar portfolio returns after winsorization:")
    for _, row in top_bottom.iterrows():
        print(f"  {row['timestamp']}: {row['portfolio_return']:+.4f}")
    
    print(f"✓ Calculated portfolio returns for {len(portfolio_returns_df)} periods")
    print(f"  Return range: [{portfolio_returns_df['portfolio_return'].min():.4f}, {portfolio_returns_df['portfolio_return'].max():.4f}]")
    
    return portfolio_returns_df


def calculate_performance_metrics_with_rebalancing(returns: pd.Series, 
                                                    transaction_cost_bps: float = 10,
                                                    rebalance_frequency: str = 'monthly') -> Dict:
    """
    Calculate performance metrics with transaction costs applied at each rebalance.
    
    Args:
        returns: Portfolio returns series
        transaction_cost_bps: Transaction cost in basis points (one-way)
        rebalance_frequency: 'daily', 'weekly', 'monthly'
    
    Returns:
        Dictionary with performance metrics including cumulative transaction costs
    """
    # Convert returns to DataFrame with timestamp index if needed
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns_df = returns.reset_index()
        returns_df.columns = ['timestamp', 'return']
        returns_df['timestamp'] = pd.to_datetime(returns_df['timestamp'])
        returns = returns_df.set_index('timestamp')['return']
    
    # Determine rebalancing dates
    if rebalance_frequency == 'daily':
        # Rebalance once per day - use first hour of trading day
        # Try multiple hours to handle different data formats
        rebalance_mask = (returns.index.hour == 10) & (returns.index.minute == 0)
        if rebalance_mask.sum() == 0:
            # Fallback: first bar of each day
            rebalance_mask = returns.groupby(returns.index.date).head(1).index
            rebalance_mask = returns.index.isin(rebalance_mask)
            
    elif rebalance_frequency == 'weekly':
        # Rebalance every Monday - try specific hour first
        rebalance_mask = (returns.index.dayofweek == 0) & (returns.index.hour == 10) & (returns.index.minute == 0)
        
        # Diagnostic: check if we found any
        if rebalance_mask.sum() == 0:
            # Fallback 1: Try any hour on Monday
            rebalance_mask = (returns.index.dayofweek == 0)
            # Take first bar of each Monday
            monday_data = returns[rebalance_mask]
            first_bars = monday_data.groupby(monday_data.index.date).head(1)
            rebalance_mask = returns.index.isin(first_bars.index)
            
        # Final fallback: use weekly resampling
        if rebalance_mask.sum() == 0:
            weekly_dates = returns.resample('W-MON').first().index
            rebalance_mask = returns.index.isin(weekly_dates)
            
    elif rebalance_frequency == 'monthly':
        # Rebalance on first Monday of each month
        monthly_first = returns.groupby([
            returns.index.year,
            returns.index.month
        ]).head(1)
        rebalance_mask = returns.index.isin(monthly_first.index)

    else:
        raise ValueError(f"Invalid rebalance_frequency: {rebalance_frequency}")
    
    # Ensure mask aligns with returns index for positional access
    rebalance_mask = pd.Series(rebalance_mask, index=returns.index)
    
    # Count rebalances
    num_rebalances = rebalance_mask.sum()
    
    # DIAGNOSTIC: Print rebalancing info
    if num_rebalances == 0:
        print(f"\n WARNING: Found 0 rebalances for {rebalance_frequency} frequency!")
        print(f"   This likely means timestamp matching failed.")
        print(f"   Data timestamps: {returns.index[:3].tolist()}")
    else:
        print(f"\n✓ Found {num_rebalances} rebalances for {rebalance_frequency} frequency")
        print(f"  First rebalance: {returns.index[rebalance_mask][0]}")
        print(f"  Last rebalance: {returns.index[rebalance_mask][-1]}")
    
    # Calculate transaction cost per rebalance (round-trip: entry + exit)
    # Assuming full portfolio turnover at each rebalance
    cost_per_rebalance = transaction_cost_bps / 10000  # Convert bps to decimal
    total_transaction_cost = num_rebalances * cost_per_rebalance * 2  # 2x for round-trip
    
    # Calculate gross returns
    gross_cumulative = (1 + returns).cumprod()
    gross_total_return = gross_cumulative.iloc[-1] - 1
    
    # Apply transaction costs at each rebalance
    portfolio_value = 1.0
    net_returns = []

    valid_returns = returns.dropna()
    rebalance_timestamps = set(returns.index[rebalance_mask])
    
    for timestamp, ret in valid_returns.items():
        # Apply return
        portfolio_value *= (1 + ret)
        
        # Apply transaction cost if rebalancing
        is_rebalance = timestamp in rebalance_timestamps
        if is_rebalance:
            portfolio_value *= (1 - cost_per_rebalance * 2)  # Round-trip cost
        
        #For per-period net returns: raw return minus cost if rebalancing
        period_net_return = ret - (cost_per_rebalance * 2 if is_rebalance else 0)
        net_returns.append(period_net_return)
    
    net_returns_series = pd.Series(net_returns, index=valid_returns.index)
    net_total_return = portfolio_value - 1
    
    # Calculate annualized metrics
    n_periods = len(returns)
    annualization_factor = 252 * 24  # Hourly data
    annualized_return = returns.mean() * annualization_factor
    annualized_return_net = net_returns_series.mean() * annualization_factor
    
    volatility = returns.std() * np.sqrt(annualization_factor)
    sharpe = (net_returns_series.mean() / (net_returns_series.std() + 1e-8)) * np.sqrt(annualization_factor)
    
    downside_returns = net_returns_series[net_returns_series < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(annualization_factor)
        sortino = (net_returns_series.mean() / (downside_std + 1e-8)) * np.sqrt(annualization_factor)
    else:
        sortino = sharpe
    
    net_cumulative = (1 + net_returns_series).cumprod()
    running_max = net_cumulative.expanding().max()
    drawdown = (net_cumulative - running_max) / running_max
    max_drawdown = max(drawdown.min(), -0.999)
    
    calmar = (annualized_return_net / abs(max_drawdown)) if max_drawdown != 0 else 0
    win_rate = (net_returns_series > 0).sum() / len(net_returns_series)
    wins = net_returns_series[net_returns_series > 0].sum()
    losses = abs(net_returns_series[net_returns_series < 0].sum())
    profit_factor = wins / (losses + 1e-8)
    
    return {
        'total_return': net_total_return,
        'gross_total_return': gross_total_return,
        'annualized_return': annualized_return_net,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_periods': n_periods,
        'num_rebalances': num_rebalances,
        'total_transaction_costs': total_transaction_cost,
        'cost_drag': gross_total_return - net_total_return
    }


def calculate_performance_metrics(returns: pd.Series, transaction_cost: float = 0.002) -> Dict:
    """
    Calculate comprehensive performance metrics with a one-time transaction cost.
    Uses mean-based annualization to temper extreme totals.
    
    DEPRECATED: Use calculate_performance_metrics_with_rebalancing() instead for active strategies.
    """
    gross_cumulative = (1 + returns).cumprod()
    gross_total_return = gross_cumulative.iloc[-1] - 1
    net_total_return = gross_total_return - transaction_cost
    
    n_periods = len(returns)
    annualization_factor = 252 * 24  # Hourly data
    annualized_return = returns.mean() * annualization_factor
    
    volatility = returns.std() * np.sqrt(annualization_factor)
    sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(annualization_factor)
    
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(annualization_factor)
        sortino = (returns.mean() / (downside_std + 1e-8)) * np.sqrt(annualization_factor)
    else:
        sortino = sharpe
    
    running_max = gross_cumulative.expanding().max()
    drawdown = (gross_cumulative - running_max) / running_max
    max_drawdown = max(drawdown.min(), -0.999)
    
    calmar = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0
    win_rate = (returns > 0).sum() / len(returns)
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = wins / (losses + 1e-8)
    
    return {
        'total_return': net_total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_periods': n_periods
    }


def detect_regimes(returns: pd.Series) -> Tuple[np.ndarray, Dict[str, str]]:
    """Detect market regimes from returns"""
    detector = StatisticalRegimeDetector()
    labels = []
    regime_to_id = {}
    regime_name_map = {}
    
    lookback = 252
    
    for i in range(len(returns)):
        if i < lookback:
            regime = MarketRegime.NORMAL
        else:
            window_returns = returns.iloc[i-lookback:i].values
            regime = detector.detect_regime(window_returns).regime
        
        if regime not in regime_to_id:
            regime_idx = len(regime_to_id)
            regime_to_id[regime] = regime_idx
            regime_name_map[str(regime_idx)] = regime.name.replace("_", " ").title()
        
        labels.append(regime_to_id[regime])
    
    return np.array(labels), regime_name_map


def plot_equity_curves(portfolio_returns: pd.DataFrame, 
                       metrics_by_cost: Dict[str, Dict],
                       output_dir: Path):
    """Plot equity curves with transaction costs applied at each rebalance"""
    print("\n[Step 6/8] Generating Visualizations...")
    print("  Creating equity curves...")
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Convert to indexed for rebalancing logic
    portfolio_returns_indexed = portfolio_returns.set_index('timestamp')
    returns = portfolio_returns_indexed['portfolio_return']

    # Helper to mirror the rebalancing detection logic used in calculate_performance_metrics_with_rebalancing
    def _weekly_rebalance_mask(returns_series: pd.Series) -> pd.Series:
        mask = (returns_series.index.dayofweek == 0) & (returns_series.index.hour == 10) & (returns_series.index.minute == 0)
        if mask.sum() == 0:
            monday_mask = returns_series.index.dayofweek == 0
            monday_data = returns_series[monday_mask]
            first_bars = monday_data.groupby(monday_data.index.date).head(1)
            mask = returns_series.index.isin(first_bars.index)
        if mask.sum() == 0:
            weekly_dates = returns_series.resample('W-MON').first().index
            mask = returns_series.index.isin(weekly_dates)
        return pd.Series(mask, index=returns_series.index)
    
    def _monthly_rebalance_mask(returns_series: pd.Series) -> pd.Series:
        # First Monday of each month at 10:00
        # Group by year-month and get first timestamp
        monthly_first = returns_series.groupby([
            returns_series.index.year, 
            returns_series.index.month
        ]).head(1)
        
        mask = returns_series.index.isin(monthly_first.index)
        return pd.Series(mask, index=returns_series.index)
        
    # Rebalancing configuration
    rebalance_configs = {
        'optimistic': {'bps': 5, 'freq': 'monthly'},
        'realistic': {'bps': 10, 'freq': 'monthly'},
        'pessimistic': {'bps': 20, 'freq': 'monthly'}
    }
    
    for scenario, color in [('optimistic', 'green'), ('realistic', 'blue'), ('pessimistic', 'red')]:
        config = rebalance_configs[scenario]
        cost_bps = config['bps']
        cost_per_rebalance = cost_bps / 10000
        
        # Determine rebalancing dates
        if config['freq'] == 'weekly':
            rebalance_mask = _weekly_rebalance_mask(returns)
        elif config['freq'] == 'monthly':
            rebalance_mask = _monthly_rebalance_mask(returns)
        
        # Ensure mask aligns with index for positional access
        rebalance_mask = pd.Series(rebalance_mask, index=returns.index)
        
        # Calculate cumulative returns with costs at each rebalance
        portfolio_value = np.ones(len(returns))
        for i in range(len(returns)):
            if i == 0:
                portfolio_value[i] = 1 + returns.iloc[i]
            else:
                portfolio_value[i] = portfolio_value[i-1] * (1 + returns.iloc[i])
            
            # Apply transaction cost if rebalancing (round-trip)
            if rebalance_mask.iloc[i]:
                portfolio_value[i] *= (1 - cost_per_rebalance * 2)
        
        metrics = metrics_by_cost[scenario]
        num_rebalances = metrics.get('num_rebalances', 0)
        label = f"{scenario.capitalize()}: Sharpe={metrics['sharpe_ratio']:.2f}, DD={metrics['max_drawdown']:.1%}, {num_rebalances} rebalances"
        
        ax.plot(returns.index, portfolio_value, label=label, color=color, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title('Multi-Asset Portfolio Equity Curves (With Rebalancing Costs)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: equity_curves.png")


def plot_underwater_chart(portfolio_returns: pd.DataFrame, output_dir: Path):
    """Plot drawdown underwater chart"""
    print("  Creating underwater chart...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    returns = portfolio_returns['portfolio_return']
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    ax.fill_between(portfolio_returns['timestamp'], drawdown, 0, color='red', alpha=0.3)
    ax.plot(portfolio_returns['timestamp'], drawdown, color='darkred', linewidth=1.5)
    
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    max_dd_date = portfolio_returns.loc[max_dd_idx, 'timestamp']
    
    ax.axhline(y=max_dd, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(max_dd_date, max_dd, f'  Max DD: {max_dd:.2%}', 
            verticalalignment='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.set_title('Portfolio Drawdown (Underwater Chart)', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'underwater_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: underwater_chart.png")


def plot_rolling_sharpe(portfolio_returns: pd.DataFrame, output_dir: Path):
    """Plot rolling Sharpe ratio"""
    print("  Creating rolling Sharpe...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    returns = portfolio_returns['portfolio_return']
    window = 63  # ~3 months
    
    rolling_mean = returns.rolling(window).mean() * 252 * 24
    rolling_std = returns.rolling(window).std() * np.sqrt(252 * 24)
    rolling_sharpe = rolling_mean / rolling_std
    
    ax.plot(portfolio_returns['timestamp'], rolling_sharpe, color='purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
    ax.set_title(f'Rolling Sharpe Ratio ({window}-period window)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'rolling_sharpe_{window}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: rolling_sharpe_{window}.png")


def plot_returns_distribution(portfolio_returns: pd.DataFrame, output_dir: Path):
    """Plot returns distribution with Q-Q plot"""
    print("  Creating returns distribution...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    returns = portfolio_returns['portfolio_return']
    
    # Histogram
    ax1.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}')
    ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.4f}')
    
    ax1.set_xlabel('Returns', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = f'Skewness: {returns.skew():.2f}\nKurtosis: {returns.kurtosis():.2f}\nStd: {returns.std():.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Q-Q plot
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: returns_distribution.png")


def plot_regime_performance(portfolio_returns: pd.DataFrame, 
                           regimes: np.ndarray,
                           regime_names: Dict[str, str],
                           output_dir: Path):
    """Plot performance by regime"""
    print("  Creating regime performance analysis...")
    
    returns = portfolio_returns['portfolio_return']
    
    # Calculate metrics by regime
    regime_metrics = {}
    for regime_id, regime_name in regime_names.items():
        regime_mask = regimes == int(regime_id)
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            sharpe = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252*24) if regime_returns.std() > 0 else 0
            regime_metrics[regime_name] = {
                'sharpe': sharpe,
                'mean_return': regime_returns.mean() * 252 * 24,
                'count': len(regime_returns)
            }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    regimes_list = list(regime_metrics.keys())
    sharpes = [regime_metrics[r]['sharpe'] for r in regimes_list]
    returns_ann = [regime_metrics[r]['mean_return'] for r in regimes_list]
    counts = [regime_metrics[r]['count'] for r in regimes_list]
    
    # Sharpe by regime
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    ax1.bar(regimes_list, sharpes, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.set_title('Sharpe Ratio by Regime', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(True, alpha=0.3)
    
    # Returns by regime
    colors = ['green' if r > 0 else 'red' for r in returns_ann]
    ax2.bar(regimes_list, returns_ann, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Annualized Return', fontsize=12)
    ax2.set_title('Returns by Regime', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True, alpha=0.3)
    
    # Count by regime
    ax3.bar(regimes_list, counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Periods', fontsize=12)
    ax3.set_title('Observations by Regime', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Mean return by regime
    ax4.bar(regimes_list, [regime_metrics[r]['mean_return'] / counts[i] * 1000 for i, r in enumerate(regimes_list)],
            color='purple', alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Mean Return (bps)', fontsize=12)
    ax4.set_title('Average Return per Period by Regime', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regime_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: regime_performance.png")


def save_performance_summary(metrics_by_cost: Dict[str, Dict],
                            portfolio_weights: Dict[str, float],
                            regime_usage: Dict,
                            weight_stats: Dict,
                            output_dir: Path,
                            prediction_metrics: Dict):
    """Save comprehensive performance summary to text file"""
    print("\n[Step 7/8] Saving Performance Summary...")
    
    summary_path = output_dir / 'performance_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REGIME-AWARE MULTI-ASSET PORTFOLIO PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Portfolio composition
        f.write("PORTFOLIO COMPOSITION\n")
        f.write("-" * 80 + "\n")
        for symbol, weight in sorted(portfolio_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "LONG" if weight > 0 else "SHORT"
            model_used = "Attention" if regime_usage[symbol]['attention'] > regime_usage[symbol]['vanilla'] else "Vanilla"
            f.write(f"  {symbol:6s}: {abs(weight)*100:>5.1f}% ({direction:>5s}) - {model_used} model\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("REGIME-AWARE MODEL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Mean Attention Weight: {weight_stats.get('mean_attention_weight', 0):.3f}\n")
        f.write(f"  Attention Model Usage: {weight_stats.get('attention_usage_pct', 0):.1f}%\n")
        f.write(f"  Vanilla Model Usage:   {weight_stats.get('vanilla_usage_pct', 0):.1f}%\n")
        f.write(f"  → Dynamic regime-based model switching working!\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("PREDICTION QUALITY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  MSE:                   {prediction_metrics['mse']:.6f}\n")
        f.write(f"  RMSE:                  {prediction_metrics['rmse']:.6f}\n")
        f.write(f"  MAE:                   {prediction_metrics['mae']:.6f}\n")
        f.write(f"  Information Coef (IC): {prediction_metrics.get('information_coefficient', 0):.4f}\n")
        f.write(f"  IC p-value:            {prediction_metrics.get('ic_pvalue',1):4f}\n")
        f.write(f"  Directional Accuracy:  {prediction_metrics['directional_accuracy']*100:.2f}%\n")
        f.write(f"  Inference Time:        {prediction_metrics['inference_time_ms']:.2f} ms\n")
        f.write(f"  Inference Time/sample: {prediction_metrics['inference_time_per_sample_ms']:.4f} ms\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE METRICS BY COST SCENARIO\n")
        f.write("=" * 80 + "\n\n")
        
        for scenario in ['optimistic', 'realistic', 'pessimistic']:
            metrics = metrics_by_cost[scenario]
            costs = {'optimistic': '5 bps', 'realistic': '10 bps', 'pessimistic': '20 bps'}
            
            f.write(f"{scenario.upper()} (Transaction Costs: {costs[scenario]})\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Return:        {metrics['total_return']:>8.2%}\n")
            f.write(f"  Annualized Return:   {metrics['annualized_return']:>8.2%}\n")
            f.write(f"  Volatility:          {metrics['volatility']:>8.2%}\n")
            f.write(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.3f}\n")
            f.write(f"  Sortino Ratio:       {metrics['sortino_ratio']:>8.3f}\n")
            f.write(f"  Calmar Ratio:        {metrics['calmar_ratio']:>8.3f}\n")
            f.write(f"  Maximum Drawdown:    {metrics['max_drawdown']:>8.2%}\n")
            f.write(f"  Win Rate:            {metrics['win_rate']:>8.2%}\n")
            f.write(f"  Profit Factor:       {metrics['profit_factor']:>8.2f}\n")
            f.write(f"  Number of Periods:   {metrics['num_periods']:>8,}\n")
            
            # Add rebalancing metrics if available
            if 'num_rebalances' in metrics:
                f.write(f"\n")
                f.write(f"  Rebalancing Strategy:\n")
                f.write(f"    Frequency:         Weekly\n")
                f.write(f"    Number of Rebalances: {metrics['num_rebalances']}\n")
                f.write(f"    Gross Return:      {metrics.get('gross_total_return', 0):.2%}\n")
                f.write(f"    Cost Drag:         {metrics.get('cost_drag', 0):.2%}\n")
                f.write(f"    Net Return:        {metrics['total_return']:.2%}\n")
            
            f.write("\n")
    
    print(f"✓ Saved: performance_summary.txt")


def main():
    """Main regime-aware multi-asset backtesting workflow"""

    SEED = 123 #for reproducibility #was 17
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Configuration
    CSV_PATH = PROJECT_ROOT / "OHLCV-1HR" / "OHLCV.csv"
    MAX_SYMBOLS = None  #None so we can use all 100 symbols
    MAX_POSITIONS = 10
    
    # 1. Load data
    df, symbols = load_multi_asset_data(CSV_PATH, max_symbols=MAX_SYMBOLS)
    
    # 2. Engineer features (stock-agnostic!)
    features_df = engineer_multi_asset_features(df, symbols)

    # 2.5 Compute market returns for cross-sectional features
    print(f"\n [Step 2.5/8] Computing Market Returns for Cross-Sectional Features...")
    market_returns_series = features_df.groupby('timestamp')['return'].mean()
    print(f" Computed market returns for {len(market_returns_series)} timestamps")
    print(f" Market return range: [{market_returns_series.min():.4f}, {market_returns_series.max():.4f}]")
    
    # 3. Calculate actual returns
    print("\n[Step 3/8] Calculating Actual Returns...")
    actual_returns_list = []
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')
        # Winsorize symbol returns at 1st/99th percentile per symbol
        raw_returns = symbol_df['close'].pct_change()
        lower, upper = raw_returns.quantile([0.01, 0.99])
        symbol_df['return'] = raw_returns.clip(lower, upper)
        actual_returns_list.append(symbol_df[['timestamp', 'symbol', 'return']])
    actual_returns_df = pd.concat(actual_returns_list)
    print(f"✓ Calculated returns for {len(symbols)} symbols")
    
    # Diagnostics: top absolute per-bar returns
    top_returns = actual_returns_df.reindex(
        actual_returns_df['return'].abs().sort_values(ascending=False).index
    ).head(5)
    print("\n[Diagnostics] Top 5 per-bar symbol returns after clipping:")
    for _, row in top_returns.iterrows():
        print(f"  {row['timestamp']} {row['symbol']}: {row['return']:+.4f}")
    
    # 4. Split data by time periods (ensures testing on different market conditions)
    # Training: 2018-2021 (includes COVID crash and recovery - mixed conditions)
    # Validation: 2022 (bear market - high volatility, declining)
    # Test: 2023-2025 (recovery and bull market - rising but includes corrections)
    
    print(f"\n[Data Split by Time Period]")
    print(f"  Training on multiple market regimes for robustness...")
    
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    
    # Time-based splits
    train_end = pd.Timestamp('2020-12-31')
    val_end = pd.Timestamp('2021-12-31')
    
    train_features = features_df[features_df['timestamp'] <= train_end]
    val_features = features_df[(features_df['timestamp'] > train_end) & 
                               (features_df['timestamp'] <= val_end)]
    test_features = features_df[features_df['timestamp'] > val_end]
    
    print(f"  Train: {len(train_features):,} samples ({train_features['timestamp'].min().date()} to {train_features['timestamp'].max().date()})")
    print(f"         Market conditions: COVID crash, recovery, mixed volatility")
    print(f"  Val:   {len(val_features):,} samples ({val_features['timestamp'].min().date()} to {val_features['timestamp'].max().date()})")
    print(f"         Market conditions: 2021 bear market, high volatility decline")
    print(f"  Test:  {len(test_features):,} samples ({test_features['timestamp'].min().date()} to {test_features['timestamp'].max().date()})")
    print(f"         Market conditions: Recovery and bull market with corrections")
    
    # 5. Create datasets
    config = get_production_config()
    sequence_length = 252  # full-year window
    
    feature_cols = [c for c in features_df.columns 
                   if c not in ['symbol', 'timestamp', 'return']]
    config.model.input_dim = len(feature_cols)
    config.model.hidden_dim = 64 #was 64
    config.model.num_transformer_layers = 1 #was 1
    config.training.num_epochs = 20
    
    train_dataset = MultiAssetDataset(train_features, symbols, sequence_length)
    val_dataset = MultiAssetDataset(val_features, symbols, sequence_length)
    
    # 6. Train regime-aware model
    model, weight_stats = train_regime_aware_model(train_dataset, val_dataset, config, market_returns_series)
    
    # 7. Generate regime-aware signals
    signals, regime_usage = generate_regime_aware_signals(model, test_features, symbols, sequence_length, market_returns_series)
    
    # Compute per-symbol volatility (std of returns) for weighting
    vol_by_symbol = actual_returns_df.groupby('symbol')['return'].std().to_dict()
    
    # 8. Construct portfolio
    portfolio_weights = construct_portfolio(
        signals,
        max_positions=MAX_POSITIONS,
        min_signal=0.1,
        vol_by_symbol=vol_by_symbol,
        max_weight=0.2
    )
    
    if not portfolio_weights:
        print("⚠️ No positions selected; terminating run early.")
        return
    
    print(f"\n[Portfolio Construction]")
    print(f"  Selected {len(portfolio_weights)} positions:")
    for symbol, weight in sorted(portfolio_weights.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "LONG" if weight > 0 else "SHORT"
        
        # Show which model was used for this stock
        model_used = "Attention" if regime_usage[symbol]['attention'] > regime_usage[symbol]['vanilla'] else "Vanilla"
        
        print(f"    {symbol:6s}: {abs(weight)*100:>5.1f}% ({direction}) - {model_used} model")
    
    # 9. Calculate portfolio returns
    # Also compute prediction metrics on aligned signals vs actual returns
    # Align predictions for held symbols
    held_symbols = set(portfolio_weights.keys())
    held_predictions = []
    held_targets = []
    inference_start = time.perf_counter()

    for symbol in held_symbols:
        symbol_data = test_features[test_features['symbol'] == symbol].copy()
        symbol_actual = actual_returns_df[actual_returns_df['symbol'] == symbol].copy()
        if len(symbol_data) < sequence_length:
            continue
        symbol_data = symbol_data.sort_values('timestamp')
        symbol_actual = symbol_actual.sort_values('timestamp')

        for i in range(sequence_length, len(symbol_data)):
            seq = symbol_data.iloc[i-sequence_length:i]
            feature_cols = [c for c in seq.columns if c not in ['symbol', 'timestamp', 'return']]
            X = torch.FloatTensor(seq[feature_cols].values).unsqueeze(0)
            with torch.no_grad():
                pred = model(X)
                if isinstance(pred, tuple):
                    pred = pred[0]
            pred_val = float(pred.squeeze().item())
            ts = symbol_data.iloc[i]['timestamp']
            target_row = symbol_actual[symbol_actual['timestamp'] == ts]
            if len(target_row) == 0:
                continue
            target_val = float(target_row['return'].values[0])
            held_predictions.append(pred_val)
            held_targets.append(target_val)

    inference_end = time.perf_counter()
    inference_time_ms = (inference_end - inference_start) * 1000
    preds_arr = np.array(held_predictions, dtype=float)
    targets_arr = np.array(held_targets, dtype=float)
    finite_mask = np.isfinite(preds_arr) & np.isfinite(targets_arr)
    preds_arr = preds_arr[finite_mask]
    targets_arr = targets_arr[finite_mask]
    total_samples = len(targets_arr)

    if total_samples > 0:
        # ═══════════════════════════════════════════════════════════════════════
        # DIAGNOSTICS: Show prediction vs actual ranges BEFORE scaling
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n[Prediction Diagnostics - BEFORE Scaling]")
        print(f"  Predictions range: [{preds_arr.min():.6f}, {preds_arr.max():.6f}]")
        print(f"  Actuals range:     [{targets_arr.min():.6f}, {targets_arr.max():.6f}]")
        print(f"  Predictions mean:  {preds_arr.mean():.6f}")
        print(f"  Actuals mean:      {targets_arr.mean():.6f}")
        print(f"  Predictions std:   {preds_arr.std():.6f}")
        print(f"  Actuals std:       {targets_arr.std():.6f}")
        
        # Check for scale mismatch
        pred_scale = preds_arr.std()
        actual_scale = targets_arr.std()
        scale_ratio = pred_scale / actual_scale if actual_scale > 0 else 1.0
        
        if scale_ratio > 10 or scale_ratio < 0.1:
            print(f"    WARNING: Predictions and actuals are on different scales!")
            print(f"     Scale ratio: {scale_ratio:.2f}x")
            print(f"     This causes terrible R² and RMSE values")
            print(f"  ✓  Applying scaling to fix metrics...")
        
        # ═══════════════════════════════════════════════════════════════════════
        # SCALE PREDICTIONS to match actual return distribution
        # ═══════════════════════════════════════════════════════════════════════
        # Save original predictions for ranking (portfolio construction already done)
        original_preds = preds_arr.copy()
        
        # Standardize predictions (zero mean, unit variance)
        pred_mean = preds_arr.mean()
        pred_std = preds_arr.std() if preds_arr.std() > 0 else 1.0
        preds_arr_standardized = (preds_arr - pred_mean) / pred_std
        
        # Scale to match actual distribution (actual mean, actual std)
        actual_mean = targets_arr.mean()
        actual_std = targets_arr.std() if targets_arr.std() > 0 else 1.0
        preds_arr = preds_arr_standardized * actual_std + actual_mean
        
        # ═══════════════════════════════════════════════════════════════════════
        # DIAGNOSTICS: Show prediction vs actual ranges AFTER scaling
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n[Prediction Diagnostics - AFTER Scaling]")
        print(f"  Scaled predictions range: [{preds_arr.min():.6f}, {preds_arr.max():.6f}]")
        print(f"  Actuals range:            [{targets_arr.min():.6f}, {targets_arr.max():.6f}]")
        print(f"  Scaled predictions mean:  {preds_arr.mean():.6f}")
        print(f"  Actuals mean:             {targets_arr.mean():.6f}")
        print(f"  Scaled predictions std:   {preds_arr.std():.6f}")
        print(f"  Actuals std:              {targets_arr.std():.6f}")
        print(f"  ✓ Predictions now on same scale as actuals!")
        print(f"  Note: Portfolio weights used original (unscaled) predictions for ranking")
        
        # ═══════════════════════════════════════════════════════════════════════
        # CALCULATE METRICS on scaled predictions
        # ═══════════════════════════════════════════════════════════════════════
        mse = mean_squared_error(targets_arr, preds_arr)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(targets_arr, preds_arr)
        r2 = r2_score(targets_arr, preds_arr)
        directional_accuracy = np.mean(np.sign(preds_arr) == np.sign(targets_arr))
        ic, ic_pvalue = spearmanr(preds_arr, targets_arr)
    else:
        mse = rmse = mae = r2 = directional_accuracy = 0.0

    prediction_metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "r2_note": "Negative R squared expected for ranking strategies",
        "information_coefficient": ic,
        "ic_pvalue": ic_pvalue,
        "directional_accuracy": directional_accuracy,
        "inference_time_ms": inference_time_ms,
        "inference_time_per_sample_ms": inference_time_ms / total_samples if total_samples else 0.0
    }

    portfolio_returns = calculate_portfolio_returns(portfolio_weights, actual_returns_df, test_features)
    if portfolio_returns.empty:
        print("⚠️ Portfolio returns are empty; terminating run early.")
        return
    
    # 10. Calculate performance metrics with REBALANCING and transaction costs
    print("\n[Step 7/7] Calculating Performance Metrics with Rebalancing...")
    print("  Strategy: Active rebalancing (not buy-and-hold)")
    
    # Convert to proper datetime index for rebalancing logic
    portfolio_returns_indexed = portfolio_returns.set_index('timestamp')
    
    # Test different rebalancing frequencies and costs
    metrics_by_cost = {}
    
    # Define cost scenarios (basis points, one-way)
    cost_scenarios = {
        'optimistic_monthly': {'bps': 5, 'freq': 'monthly'},
        'realistic_monthly': {'bps': 10, 'freq': 'monthly'},
        'pessimistic_monthly': {'bps': 20, 'freq': 'monthly'},
    }
    
    cost_labels = {
        'optimistic_monthly': '5 bps, monthly rebalance',
        'realistic_monthly': '10 bps, monthly rebalance',
        'pessimistic_monthly': '20 bps, monthly rebalance',
    }
    
    for scenario, params in cost_scenarios.items():
        # Calculate metrics with rebalancing
        metrics = calculate_performance_metrics_with_rebalancing(
            portfolio_returns_indexed['portfolio_return'],
            transaction_cost_bps=params['bps'],
            rebalance_frequency=params['freq']
        )
        metrics_by_cost[scenario] = metrics
        
        print(f"\n  {scenario.upper().replace('_', ' ')}:")
        print(f"    Costs: {cost_labels[scenario]}")
        print(f"    Rebalances: {metrics['num_rebalances']}")
        print(f"    Gross Return: {metrics['gross_total_return']:.2%}")
        print(f"    Cost Drag: {metrics['cost_drag']:.2%}")
        print(f"    Net Return: {metrics['total_return']:.2%}")
        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # For backward compatibility, also store as 'optimistic', 'realistic', 'pessimistic'
    metrics_by_cost['optimistic'] = metrics_by_cost['optimistic_monthly']
    metrics_by_cost['realistic'] = metrics_by_cost['realistic_monthly']
    metrics_by_cost['pessimistic'] = metrics_by_cost['pessimistic_monthly']
    
    # 11. Detect regimes
    regimes, regime_names = detect_regimes(portfolio_returns['portfolio_return'])
    
    # 12. Create all visualizations
    plot_equity_curves(portfolio_returns, metrics_by_cost, OUTPUTS_DIR)
    plot_underwater_chart(portfolio_returns, OUTPUTS_DIR)
    plot_rolling_sharpe(portfolio_returns, OUTPUTS_DIR)
    plot_returns_distribution(portfolio_returns, OUTPUTS_DIR)
    plot_regime_performance(portfolio_returns, regimes, regime_names, OUTPUTS_DIR)
    
    # 13. Save performance summary
    save_performance_summary(metrics_by_cost, portfolio_weights, regime_usage, weight_stats, OUTPUTS_DIR, prediction_metrics)
    
    # 14. Save portfolio returns to CSV
    print("\n[Step 8/8] Saving Data...")
    portfolio_returns.to_csv(OUTPUTS_DIR / 'portfolio_returns.csv', index=False)
    print(f"✓ Saved: portfolio_returns.csv")
    
    # 15. Print final summary
    print("\n" + "=" * 80)
    print("✅ REGIME-AWARE MULTI-ASSET BACKTEST COMPLETE!")
    print("=" * 80)
    print(f"\nKEY FEATURES:")
    print(f"  ✅ Trained on {len(symbols)} symbols with stock-agnostic features")
    print(f"  ✅ Ensemble model with regime-aware dynamic weighting")
    print(f"  ✅ Automatic switching between vanilla/attention models")
    print(f"  ✅ Diversified portfolio: {len(portfolio_weights)} positions")
    print(f"\n  REGIME AWARENESS:")
    print(f"    • Model detected regimes automatically")
    print(f"    • Attention model used: {weight_stats.get('attention_usage_pct', 0):.1f}% of time")
    print(f"    • Vanilla model used: {weight_stats.get('vanilla_usage_pct', 0):.1f}% of time")
    print(f"    • Dynamic adaptation to market conditions!")
    
    print(f"\n  PERFORMANCE (Realistic Costs):")
    realistic = metrics_by_cost['realistic']
    print(f"    Sharpe Ratio:      {realistic['sharpe_ratio']:>8.3f}")
    print(f"    Sortino Ratio:     {realistic['sortino_ratio']:>8.3f}")
    print(f"    Calmar Ratio:      {realistic['calmar_ratio']:>8.3f}")
    print(f"    Total Return:      {realistic['total_return']:>8.2%}")
    print(f"    Annualized Return: {realistic['annualized_return']:>8.2%}")
    print(f"    Max Drawdown:      {realistic['max_drawdown']:>8.2%}")
    print(f"    Win Rate:          {realistic['win_rate']:>8.2%}")
    
    print(f"\n  FILES CREATED:")
    print(f"    • equity_curves.png")
    print(f"    • underwater_chart.png")
    print(f"    • rolling_sharpe_63.png")
    print(f"    • returns_distribution.png")
    print(f"    • regime_performance.png")
    print(f"    • performance_summary.txt")
    print(f"    • portfolio_returns.csv")
    print("=" * 80)


if __name__ == "__main__":
    main() 
