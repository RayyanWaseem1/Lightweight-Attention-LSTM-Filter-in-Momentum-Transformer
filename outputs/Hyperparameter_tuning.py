"""
Hyperparameter Tuning for Momentum Transformer using Ray Tune with REAL OHLCV DATA

Optimizes model architecture and training parameters using your actual market data.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Tuple, List
import warnings
import json
from datetime import datetime

# Ray Tune imports
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.bayesopt import BayesOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️  Ray Tune not installed. Install with: pip install ray[tune] hyperopt bayesian-optimization")

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Ensure Ray workers inherit project root on PYTHONPATH
pythonpath_parts = os.environ.get("PYTHONPATH", "").split(os.pathsep) if os.environ.get("PYTHONPATH") else []
if str(PROJECT_ROOT) not in pythonpath_parts:
    pythonpath_parts.insert(0, str(PROJECT_ROOT))
    os.environ["PYTHONPATH"] = os.pathsep.join([p for p in pythonpath_parts if p])

# Import your models and utilities
from Models.config import Config
from Models.Ensemble_model import get_ensemble_model
from Utils.Losses import SharpeRatioLoss
from Utils.Feature_engineering import create_features_from_ohlcv

# Leveraged ETFs to exclude
LEVERAGED_ETFS = [
    'SOXL', 'SOXS', 'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'TNA', 'TZA',
    'LABU', 'LABD', 'JNUG', 'JDST', 'NUGT', 'DUST', 'ERX', 'ERY',
    'FAS', 'FAZ', 'TECL', 'TECS', 'CURE', 'CUT', 'WANT', 'GASL',
    'BULL', 'BEAR', 'TSLL', 'TSLS', 'NVDL', 'NVDS', 'MSTU', 'MSTX',
    'SSO', 'SDS', 'QLD', 'QID', 'DDM', 'DXD', 'UWM', 'TWM', 'SPUU', 'SPDD', 'QQQU', 'SQQD'
]


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


def load_real_data(
    csv_path: str = None,
    max_symbols: int = 52,  #Using all of the symbols from backtest for same distribution
    train_split: float = 0.7,
    val_split: float = 0.15
) -> Tuple[Dataset, Dataset, Dataset, int]:
    """
    Load your actual OHLCV data for hyperparameter tuning
    
    Returns:
        train_dataset, val_dataset, test_dataset, num_features
    """
    if csv_path is None:
        csv_path = PROJECT_ROOT / "OHLCV-1HR" / "OHLCV.csv"
    
    print(f"📊 Loading real OHLCV data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Handle timestamp column
    if 'ts_event' in df.columns:
        ts_col = 'ts_event'
    elif 'timestamp' in df.columns:
        ts_col = 'timestamp'
    else:
        raise ValueError("CSV must have timestamp column")
    
    df['timestamp'] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(None)
    df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Convert to numeric and drop NaNs
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Filter 1: Remove penny stocks (< $2)
    symbol_stats = df.groupby('symbol').agg({'close': ['mean', 'min']}).reset_index()
    symbol_stats.columns = ['symbol', 'avg_price', 'min_price']
    good_symbols = symbol_stats[symbol_stats['min_price'] >= 2.0]['symbol'].tolist()
    df = df[df['symbol'].isin(good_symbols)]
    
    # Filter 2: Remove leveraged ETFs
    leveraged_in_data = [s for s in LEVERAGED_ETFS if s in good_symbols]
    if leveraged_in_data:
        good_symbols = [s for s in good_symbols if s not in LEVERAGED_ETFS]
        df = df[df['symbol'].isin(good_symbols)]
    
    # Filter 3: Ensure sufficient data (≥3,000 bars)
    bar_counts = df.groupby('symbol').size()
    good_symbols = bar_counts[bar_counts >= 3000].index.tolist()
    df = df[df['symbol'].isin(good_symbols)]
    
    # Limit to max_symbols for faster tuning
    if max_symbols and len(good_symbols) > max_symbols:
        # Select diverse set of symbols
        good_symbols = sorted(good_symbols)[:max_symbols]
        df = df[df['symbol'].isin(good_symbols)]
    
    print(f"  Selected {len(good_symbols)} symbols for tuning")
    print(f"  Total bars: {len(df):,}")
    
    # Engineer features for all symbols
    print("  Creating features...")
    
    # Get SPY for market context if available
    spy_data = None
    if 'SPY' in df['symbol'].values:
        spy_df = df[df['symbol'] == 'SPY'].sort_values('timestamp')
        spy_data = spy_df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
    
    all_features = []
    for symbol in good_symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('timestamp').set_index('timestamp')
        
        # Create features (stock-agnostic)
        features = create_features_from_ohlcv(
            symbol_df[['open', 'high', 'low', 'close', 'volume']],
            market_df=spy_data,
            symbol=symbol
        )
        features = features.reset_index()
        
        # Add forward return target (1-step ahead)
        symbol_df['return'] = symbol_df['close'].pct_change().shift(-1)
        features['return'] = symbol_df['return'].reindex(features['timestamp']).values
        features['symbol'] = symbol
        
        all_features.append(features)
    
    # Combine all symbols
    combined = pd.concat(all_features, ignore_index=True)
    combined = combined.dropna()
    
    # Time-based splits
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    train_end = pd.Timestamp('2020-12-31')
    val_start = pd.Timestamp('2021-01-01')
    val_end = pd.Timestamp('2022-12-31')
    
    train_features = combined[combined['timestamp'] <= train_end]
    val_features = combined[(combined['timestamp'] >= val_start) & 
                           (combined['timestamp'] <= val_end)]
    test_features = combined[combined['timestamp'] > val_end]
    
    # Get number of features
    feature_cols = [c for c in combined.columns if c not in ['symbol', 'timestamp', 'return']]
    num_features = len(feature_cols)
    
    # Create datasets
    sequence_length = 252
    train_dataset = MultiAssetDataset(train_features, good_symbols, sequence_length)
    val_dataset = MultiAssetDataset(val_features, good_symbols, sequence_length)
    test_dataset = MultiAssetDataset(test_features, good_symbols, sequence_length)
    
    print(f"✓ Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"  Features: {num_features}")
    print(f"  Date ranges:")
    print(f"    Train: {train_features['timestamp'].min().date()} to {train_features['timestamp'].max().date()}")
    print(f"    Val:   {val_features['timestamp'].min().date()} to {val_features['timestamp'].max().date()}")
    print(f"    Test:  {test_features['timestamp'].min().date()} to {test_features['timestamp'].max().date()}")
    
    return train_dataset, val_dataset, test_dataset, num_features


def train_model_trial(config_dict: Dict) -> None:
    """
    Training function for a single Ray Tune trial
    """
    
    # Create config from hyperparameters
    config = Config()
    
    # Model hyperparameters
    config.model.hidden_dim = config_dict['hidden_dim']
    config.model.num_transformer_layers = config_dict['num_transformer_layers']
    config.model.num_attention_heads = config_dict['num_attention_heads']
    config.model.transformer_dropout = config_dict['dropout']
    config.model.lstm_num_layers = config_dict['lstm_num_layers']
    config.model.lstm_dropout = config_dict['lstm_dropout']
    
    # Training hyperparameters
    config.training.learning_rate = config_dict['learning_rate']
    config.training.weight_decay = config_dict['weight_decay']
    config.training.batch_size = config_dict['batch_size']
    
    # Ensemble hyperparameters
    config.ensemble.weight_hidden_dim = config_dict['weight_hidden_dim']
    config.ensemble.weight_dropout = config_dict['weight_dropout']
    
    # Load real data (cached globally to avoid reloading each trial)
    global CACHED_DATA
    if CACHED_DATA is None:
        CACHED_DATA = load_real_data(max_symbols=52)  #Ensures tuned hyperparameters work well on the full backtest
    
    train_dataset, val_dataset, _, num_features = CACHED_DATA
    config.model.input_dim = num_features
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_ensemble_model(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Loss function
    criterion = SharpeRatioLoss(annualization_factor=np.sqrt(252 * 24))
    
    # Training loop
    num_epochs = config_dict.get('num_epochs', 15)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(X, return_components=True)
            loss = criterion(predictions.squeeze(), y.squeeze())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred, _ = model(X, return_components=True)
                val_predictions.extend(pred.squeeze().cpu().numpy())
                val_targets.extend(y.squeeze().cpu().numpy())
        
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        # Calculate validation Sharpe (long-only top 20%)
        pred_80th = np.percentile(val_predictions, 80)
        long_mask = val_predictions >= pred_80th
        strategy_returns = np.where(long_mask, val_targets, 0)
        
        val_std = strategy_returns.std()
        if val_std > 0 and np.isfinite(val_std):
            val_sharpe = (strategy_returns.mean() / val_std) * np.sqrt(252 * 24)
        else:
            val_sharpe = -float('inf')
        
        # Report metrics to Ray Tune
        tune.report({
            'val_sharpe': float(val_sharpe),
            'val_loss': np.mean(train_losses),
            'train_loss': np.mean(train_losses),
            'epoch': epoch
        })


def get_search_space(search_type: str = 'default') -> Dict:
    """Define hyperparameter search space"""
    
    if search_type == 'quick':
        # Fast search for testing (5-10 trials)
        return {
            'hidden_dim': tune.choice([32, 64]),
            'num_transformer_layers': tune.choice([1, 2]),
            'num_attention_heads': tune.choice([2, 4]),
            'dropout': tune.uniform(0.1, 0.3),
            'lstm_num_layers': 2,
            'lstm_dropout': 0.2,
            'learning_rate': tune.loguniform(1e-4, 1e-3),
            'weight_decay': tune.loguniform(1e-6, 1e-4),
            'batch_size': tune.choice([64, 128]),
            'weight_hidden_dim': 32,
            'weight_dropout': 0.2,
            'num_epochs': 10
        }
    
    elif search_type == 'comprehensive':
        # Thorough search (50-100 trials)
        return {
            'hidden_dim': tune.choice([32, 64, 128]),
            'num_transformer_layers': tune.choice([1, 2, 3]),
            'num_attention_heads': tune.choice([2, 4, 8]),
            'dropout': tune.uniform(0.1, 0.4),
            'lstm_num_layers': tune.choice([1, 2]),
            'lstm_dropout': tune.uniform(0.1, 0.3),
            'learning_rate': tune.loguniform(1e-4, 5e-3),
            'weight_decay': tune.loguniform(1e-6, 1e-3),
            'batch_size': tune.choice([64, 128, 256]),
            'weight_hidden_dim': tune.choice([16, 32, 64]),
            'weight_dropout': tune.uniform(0.1, 0.3),
            'num_epochs': 15
        }
    
    elif search_type == 'production':
        # Fine-tuning around known good values (20-30 trials)
        return {
            'hidden_dim': tune.choice([56, 64, 72]),
            'num_transformer_layers': tune.choice([1, 2]),
            'num_attention_heads': 4,
            'dropout': tune.uniform(0.15, 0.35),
            'lstm_num_layers': 2,
            'lstm_dropout': tune.uniform(0.15, 0.3),
            'learning_rate': tune.loguniform(2e-4, 8e-4),
            'weight_decay': tune.loguniform(5e-5, 2e-4),
            'batch_size': 128,
            'weight_hidden_dim': tune.choice([24, 32, 40]),
            'weight_dropout': tune.uniform(0.15, 0.3),
            'num_epochs': 15
        }
    
    else:  # 'default'
        # Balanced search (20-30 trials)
        return {
            'hidden_dim': tune.choice([32, 64, 128]),
            'num_transformer_layers': tune.choice([1, 2, 3]),
            'num_attention_heads': tune.choice([2, 4]),
            'dropout': tune.uniform(0.1, 0.4),
            'lstm_num_layers': tune.choice([1, 2]),
            'lstm_dropout': tune.uniform(0.1, 0.3),
            'learning_rate': tune.loguniform(1e-4, 5e-3),
            'weight_decay': tune.loguniform(1e-6, 1e-3),
            'batch_size': tune.choice([64, 128]),
            'weight_hidden_dim': tune.choice([24, 32, 48]),
            'weight_dropout': tune.uniform(0.1, 0.3),
            'num_epochs': 15
        }


def run_hyperparameter_search(
    search_type: str = 'default',
    num_samples: int = 20,
    max_concurrent_trials: int = 4,
    search_algorithm: str = 'hyperopt',
    use_gpu: bool = False,
    output_dir: str = './ray_results'
) -> pd.DataFrame:
    """Run hyperparameter search using Ray Tune"""
    
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune not installed. Install with: pip install ray[tune] hyperopt bayesian-optimization")
    
    print("=" * 80)
    print("HYPERPARAMETER TUNING WITH RAY TUNE")
    print("=" * 80)
    print(f"Search Type: {search_type}")
    print(f"Number of Trials: {num_samples}")
    print(f"Search Algorithm: {search_algorithm}")
    print(f"Max Concurrent: {max_concurrent_trials}")
    print(f"Using GPU: {use_gpu}")
    print("=" * 80)
    
    # Initialize Ray
    if not ray.is_initialized():
        runtime_env = {
            "working_dir": str(PROJECT_ROOT),
            "env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", str(PROJECT_ROOT))},
            "excludes": ["OHLCV-1HR/OHLCV.csv"]  # Don't upload large data file
        }
        
        ray.init(
            num_cpus=os.cpu_count(),
            num_gpus=1 if use_gpu and torch.cuda.is_available() else 0,
            ignore_reinit_error=True,
            runtime_env=runtime_env
        )
    
    # Get search space
    search_space = get_search_space(search_type)
    
    # Configure search algorithm
    if search_algorithm == 'hyperopt':
        search_alg = HyperOptSearch(metric='val_sharpe', mode='max')
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent_trials)
    elif search_algorithm == 'bayesopt':
        search_alg = BayesOptSearch(metric='val_sharpe', mode='max')
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent_trials)
    else:
        search_alg = None
    
    # Configure scheduler
    scheduler = ASHAScheduler(
        metric='val_sharpe',
        mode='max',
        max_t=search_space.get('num_epochs', 15),
        grace_period=5,
        reduction_factor=2
    )
    
    # Configure reporter
    reporter = CLIReporter(
        metric_columns=['val_sharpe', 'train_loss', 'val_loss', 'epoch'],
        max_progress_rows=10,
        max_report_frequency=1800
    )
    
    # Configure resources
    resources_per_trial = {
        'cpu': 2,
        'gpu': 0.5 if use_gpu and torch.cuda.is_available() else 0
    }
    
    # Resolve storage path to absolute URI (Ray 2.9+ expects a URI)
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    storage_uri = output_dir_path.as_uri()
    
    print("\n🚀 Starting hyperparameter search with REAL OHLCV data...\n")
    
    # Run tuning
    analysis = tune.run(
        train_model_trial,
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        resources_per_trial=resources_per_trial,
        storage_path=storage_uri,
        name=f"momentum_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        stop={'training_iteration': search_space.get('num_epochs', 15)},
        verbose=0,
        raise_on_failed_trial=False
    )
    
    print("\n✓ Hyperparameter search completed!")
    
    # Get results
    results_df = analysis.results_df
    completed_df = results_df[results_df['val_sharpe'].notna()] if 'val_sharpe' in results_df.columns else pd.DataFrame()
    
    if len(completed_df) == 0:
        print("\n❌ No completed trials with metric 'val_sharpe'. Check trial logs for errors.")
        results_df.to_csv(output_dir_path / 'all_results.csv', index=False)
        ray.shutdown()
        return results_df
    
    best_trial = analysis.get_best_trial(metric='val_sharpe', mode='max')
    if best_trial is None:
        print("\n❌ Ray Tune could not determine a best trial. Check the metric name or trial outputs.")
        results_df.to_csv(output_dir_path / 'all_results.csv', index=False)
        ray.shutdown()
        return results_df
    best_config = best_trial.config
    best_sharpe = best_trial.last_result.get('val_sharpe', float('nan'))
    
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Best Validation Sharpe: {best_sharpe:.4f}\n")
    print("Hyperparameters:")
    for key, value in best_config.items():
        if key != 'num_epochs':
            print(f"  {key:25s}: {value}")
    print("=" * 80)
    
    # Save results
    output_path = output_dir_path
    
    with open(output_path / 'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    results_df.to_csv(output_path / 'all_results.csv', index=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  - best_config.json")
    print(f"  - all_results.csv")
    
    ray.shutdown()
    
    return results_df


def load_best_config(config_path: str = './ray_results/best_config.json') -> Config:
    """Load best configuration from tuning results"""
    with open(config_path, 'r') as f:
        best_params = json.load(f)
    
    config = Config()
    
    config.model.hidden_dim = best_params['hidden_dim']
    config.model.num_transformer_layers = best_params['num_transformer_layers']
    config.model.num_attention_heads = best_params['num_attention_heads']
    config.model.transformer_dropout = best_params['dropout']
    config.model.lstm_num_layers = best_params['lstm_num_layers']
    config.model.lstm_dropout = best_params['lstm_dropout']
    
    config.training.learning_rate = best_params['learning_rate']
    config.training.weight_decay = best_params['weight_decay']
    config.training.batch_size = best_params['batch_size']
    
    config.ensemble.weight_hidden_dim = best_params['weight_hidden_dim']
    config.ensemble.weight_dropout = best_params['weight_dropout']
    
    print("✓ Loaded best configuration from tuning")
    return config


# Global cache for data (to avoid reloading for each trial)
CACHED_DATA = None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with REAL OHLCV data')
    parser.add_argument('--search_type', type=str, default='default',
                       choices=['quick', 'default', 'comprehensive', 'production'],
                       help='Type of search space')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of trials')
    parser.add_argument('--max_concurrent', type=int, default=4,
                       help='Maximum concurrent trials')
    parser.add_argument('--search_alg', type=str, default='hyperopt',
                       choices=['random', 'hyperopt', 'bayesopt'],
                       help='Search algorithm')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU if available')
    parser.add_argument('--output_dir', type=str, default='./ray_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run hyperparameter search
    results = run_hyperparameter_search(
        search_type=args.search_type,
        num_samples=args.num_samples,
        max_concurrent_trials=args.max_concurrent,
        search_algorithm=args.search_alg,
        use_gpu=args.use_gpu,
        output_dir=args.output_dir
    )
    
    print("\n✓ Hyperparameter tuning complete!")
    print(f"\nTo use best config:")
    print(f"  from Utils.Hyperparameter_tuning import load_best_config")
    print(f"  config = load_best_config('{args.output_dir}/best_config.json')")
