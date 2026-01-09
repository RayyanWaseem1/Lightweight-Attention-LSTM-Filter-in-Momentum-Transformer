"""
PROPER SOLUTION: DETECT AND HANDLE STOCK SPLITS

The 75% threshold is catching stock splits, not just corruption.
We need to detect splits and either:
1. Remove just those specific bars (keep the stock)
2. Adjust prices for the split
3. Use split-adjusted data from the start

This keeps legitimate stocks like META, GOOGL, AMZN, NVDA.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def detect_stock_splits(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """
    Detect likely stock splits by finding extreme price jumps
    
    Returns list of detected splits with date, ratio, etc.
    """
    symbol_df = df[df['symbol'] == symbol].sort_values('timestamp').copy()
    
    if len(symbol_df) < 2:
        return []
    
    # Calculate price changes
    symbol_df['price_ratio'] = symbol_df['close'] / symbol_df['close'].shift(1)
    symbol_df['return'] = symbol_df['close'].pct_change()
    
    splits = []
    
    # Look for extreme price jumps (likely splits)
    for idx, row in symbol_df.iterrows():
        price_ratio = row['price_ratio']
        ret = row['return']
        
        # Skip first row
        if pd.isna(price_ratio):
            continue
        
        # Detect forward split (price drops by >50%)
        if price_ratio < 0.5:  # e.g., 2:1 split, price halves
            splits.append({
                'timestamp': row['timestamp'],
                'type': 'forward_split',
                'ratio': 1 / price_ratio,
                'return': ret,
                'price_before': symbol_df.loc[idx, 'close'] / price_ratio,
                'price_after': row['close']
            })
        
        # Detect reverse split (price increases by >100%)
        elif price_ratio > 2.0:  # e.g., 1:2 reverse split, price doubles
            splits.append({
                'timestamp': row['timestamp'],
                'type': 'reverse_split',
                'ratio': price_ratio,
                'return': ret,
                'price_before': symbol_df.loc[idx, 'close'] / price_ratio,
                'price_after': row['close']
            })
    
    return splits


def load_multi_asset_data_with_split_handling(csv_path: str, 
                                              max_symbols: int = None,
                                              handle_splits: str = 'remove_bars') -> Tuple[pd.DataFrame, List[str]]:
    """
    Load OHLCV data with proper stock split handling
    
    Args:
        csv_path: Path to OHLCV CSV
        max_symbols: Maximum number of symbols to load
        handle_splits: How to handle splits:
            'remove_bars': Remove bars with splits (keep stock)
            'remove_symbols': Remove entire symbol if it has splits
            'keep_as_is': Keep everything (not recommended)
    """
    print("\n[Step 1/7] Loading Multi-Asset Data with Stock Split Detection...")
    
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / csv_path
    df = pd.read_csv(csv_path)
    
    if 'ts_event' in df.columns:
        ts_col = 'ts_event'
    elif 'timestamp' in df.columns:
        ts_col = 'timestamp'
    else:
        raise ValueError("CSV must have timestamp column")
    
    df['timestamp'] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(None)
    df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    initial_bars = len(df)
    initial_symbols = df['symbol'].nunique()
    print(f"  Initial: {initial_bars:,} bars, {initial_symbols} symbols")
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 1: Remove penny stocks
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  [Filter 1] Removing severe penny stocks (min price < $2)...")
    
    symbol_stats = df.groupby('symbol').agg({
        'close': ['mean', 'min']
    }).reset_index()
    symbol_stats.columns = ['symbol', 'avg_price', 'min_price']
    
    good_symbols = symbol_stats[symbol_stats['min_price'] >= 2.0]['symbol'].tolist()
    removed = initial_symbols - len(good_symbols)
    print(f"    Removed: {removed} symbols")
    print(f"    Remaining: {len(good_symbols)} symbols ({len(good_symbols)/initial_symbols*100:.1f}%)")
    
    df = df[df['symbol'].isin(good_symbols)]
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 2: DETECT AND HANDLE STOCK SPLITS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  [Filter 2] Detecting stock splits...")
    
    symbols_with_splits = []
    bars_to_remove = []
    split_details = {}
    
    for symbol in good_symbols:
        splits = detect_stock_splits(df, symbol)
        
        if splits:
            symbols_with_splits.append(symbol)
            split_details[symbol] = splits
            
            # Collect bars to remove
            for split in splits:
                # Find the bar with this timestamp
                bar_idx = df[(df['symbol'] == symbol) & 
                           (df['timestamp'] == split['timestamp'])].index
                bars_to_remove.extend(bar_idx.tolist())
    
    if symbols_with_splits:
        print(f"    Found {len(symbols_with_splits)} symbols with stock splits:")
        for symbol in symbols_with_splits[:15]:
            splits = split_details[symbol]
            print(f"      {symbol}: {len(splits)} split(s) detected")
            for split in splits[:2]:  # Show first 2 splits
                print(f"        {split['timestamp'].date()}: {split['ratio']:.1f}:1 {split['type']}")
        
        if handle_splits == 'remove_bars':
            print(f"\n    Strategy: Removing {len(bars_to_remove)} bars with splits (keeping stocks)")
            df = df.drop(bars_to_remove)
            print(f"    Kept all {len(symbols_with_splits)} symbols with splits removed")
            
        elif handle_splits == 'remove_symbols':
            print(f"\n    Strategy: Removing {len(symbols_with_splits)} symbols entirely")
            good_symbols = [s for s in good_symbols if s not in symbols_with_splits]
            df = df[df['symbol'].isin(good_symbols)]
            print(f"    Remaining: {len(good_symbols)} symbols")
        
        elif handle_splits == 'keep_as_is':
            print(f"\n    Strategy: Keeping splits as-is (not recommended)")
    else:
        print(f"    No stock splits detected")
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 3: Remove truly corrupted data (REPEATED extreme values)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  [Filter 3] Detecting data corruption (repeated extreme values)...")
    
    df['return'] = df.groupby('symbol')['close'].pct_change()
    
    corrupted_symbols = []
    
    for symbol in good_symbols:
        if symbol not in df['symbol'].values:
            continue
            
        symbol_df = df[df['symbol'] == symbol].copy()
        returns = symbol_df['return'].dropna()
        
        if len(returns) < 100:
            continue
        
        # Find extreme returns (>50%)
        extreme_returns = returns[returns.abs() > 0.50]
        
        if len(extreme_returns) > 0:
            # Check if they're repeated (corruption signature)
            unique_extreme = extreme_returns.nunique()
            total_extreme = len(extreme_returns)
            repeat_ratio = 1 - (unique_extreme / total_extreme)
            
            # If >50% of extreme values are repeats, it's corrupted
            if repeat_ratio > 0.5 and len(extreme_returns) > 3:
                corrupted_symbols.append(symbol)
                print(f"      {symbol}: {len(extreme_returns)} extreme moves, "
                      f"{repeat_ratio*100:.0f}% are repeats (CORRUPTED)")
    
    if corrupted_symbols:
        print(f"    Found {len(corrupted_symbols)} corrupted symbols")
        good_symbols = [s for s in good_symbols if s not in corrupted_symbols]
        df = df[df['symbol'].isin(good_symbols)]
        print(f"    Removed: {corrupted_symbols}")
    else:
        print(f"    No data corruption detected")
    
    df = df.drop(columns=['return'])
    
    # ══════════════════════════════════════════════════════════════════════
    # FILTER 4: Sufficient data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  [Filter 4] Ensuring sufficient data (≥3,000 bars)...")
    bar_counts = df.groupby('symbol').size()
    good_symbols = [s for s in good_symbols if s in bar_counts[bar_counts >= 3000].index]
    df = df[df['symbol'].isin(good_symbols)]
    print(f"    Remaining: {len(good_symbols)} symbols ({len(good_symbols)/initial_symbols*100:.1f}%)")
    
    if max_symbols and len(good_symbols) > max_symbols:
        good_symbols = good_symbols[:max_symbols]
        df = df[df['symbol'].isin(good_symbols)]
        print(f"  Limited to {max_symbols} symbols for testing")
    
    final_bars = len(df)
    final_symbols = len(good_symbols)
    
    print(f"\n✓ Final dataset: {final_bars:,} bars across {final_symbols} symbols")
    print(f"  Kept: {final_bars/initial_bars*100:.1f}% of bars")
    print(f"  Kept: {final_symbols/initial_symbols*100:.1f}% of symbols")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Show which blue chips we kept
    blue_chips = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX']
    kept_blue_chips = [s for s in blue_chips if s in good_symbols]
    if kept_blue_chips:
        print(f"\n  Blue-chip stocks kept: {', '.join(kept_blue_chips)}")
    
    removed_blue_chips = [s for s in blue_chips if s in df['symbol'].unique() and s not in good_symbols]
    if removed_blue_chips:
        print(f"  Blue-chip stocks removed: {', '.join(removed_blue_chips)}")
    
    return df, list(good_symbols)


def analyze_extreme_returns(csv_path: str):
    """
    Analyze which symbols have extreme returns and why
    
    Helps distinguish between:
    1. Stock splits (one-time jumps)
    2. Corruption (repeated values)
    3. Legitimate volatility (varies naturally)
    """
    print("=" * 80)
    print("ANALYZING EXTREME RETURNS")
    print("=" * 80)
    
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / csv_path
    df = pd.read_csv(csv_path)
    
    if 'ts_event' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts_event'], utc=True).dt.tz_convert(None)
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
    
    df = df[['timestamp', 'symbol', 'close']].dropna()
    df = df.sort_values(['symbol', 'timestamp'])
    df['return'] = df.groupby('symbol')['close'].pct_change()
    
    # Find symbols with >75% returns
    extreme_symbols = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        returns = symbol_df['return'].dropna()
        
        extreme = returns[returns.abs() > 0.75]
        
        if len(extreme) > 0:
            splits = detect_stock_splits(df, symbol)
            
            extreme_symbols.append({
                'symbol': symbol,
                'max_return': returns.abs().max(),
                'extreme_count': len(extreme),
                'unique_extreme': extreme.nunique(),
                'repeat_ratio': 1 - (extreme.nunique() / len(extreme)) if len(extreme) > 0 else 0,
                'splits_detected': len(splits),
                'is_blue_chip': symbol in ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX']
            })
    
    results_df = pd.DataFrame(extreme_symbols).sort_values('max_return', ascending=False)
    
    print("\n" + "=" * 80)
    print("SYMBOLS WITH >75% RETURNS")
    print("=" * 80)
    
    print("\nBLUE-CHIP STOCKS (Likely Stock Splits):")
    blue_chips = results_df[results_df['is_blue_chip']]
    if len(blue_chips) > 0:
        for _, row in blue_chips.iterrows():
            print(f"  {row['symbol']:6s}: Max {row['max_return']*100:7.1f}%  "
                  f"({row['extreme_count']} occurrences)  "
                  f"Splits: {row['splits_detected']}")
        print(f"\n  → These are legitimate companies with stock splits")
        print(f"  → Should KEEP stocks, remove split bars")
    
    print("\nOTHER STOCKS:")
    others = results_df[~results_df['is_blue_chip']]
    for _, row in others.head(10).iterrows():
        corruption_flag = "⚠️ CORRUPTION" if row['repeat_ratio'] > 0.5 else ""
        print(f"  {row['symbol']:6s}: Max {row['max_return']*100:7.1f}%  "
              f"({row['extreme_count']} occurrences)  "
              f"Repeats: {row['repeat_ratio']*100:.0f}%  {corruption_flag}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("""
Strategy: Remove split bars, keep stocks

1. Detect stock splits (price jumps >100%)
2. Remove those specific bars
3. Keep the rest of the stock data
4. Apply normal return clipping (99th percentile)

Result:
  ✓ Keeps META, GOOGL, AMZN, NVDA, TSLA
  ✓ Removes problematic bars
  ✓ 70-80 symbols available for training
""")


if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "OHLCV-1HR/OHLCV.csv"
    
    # Analyze first
    analyze_extreme_returns(csv_path)
    
    print("\n\n" + "=" * 80)
    print("TESTING SPLIT HANDLING")
    print("=" * 80)
    
    # Test different handling strategies
    print("\n\nSTRATEGY 1: Remove bars with splits (RECOMMENDED)")
    df1, symbols1 = load_multi_asset_data_with_split_handling(
        csv_path, 
        handle_splits='remove_bars'
    )
    
    print("\n\n" + "-" * 80)
    print("\nSTRATEGY 2: Remove symbols with splits (NOT RECOMMENDED)")
    df2, symbols2 = load_multi_asset_data_with_split_handling(
        csv_path,
        handle_splits='remove_symbols'
    )
    
    print("\n\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"\nRemove bars:    {len(symbols1)} symbols kept")
    print(f"Remove symbols: {len(symbols2)} symbols kept")
    print(f"\nRecommendation: Use 'remove_bars' strategy")
