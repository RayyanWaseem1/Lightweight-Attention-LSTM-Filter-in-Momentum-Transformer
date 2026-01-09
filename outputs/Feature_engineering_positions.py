"""
Enhanced Stock-Agnostic Feature Engineering with Position Awareness

CRITICAL ADDITIONS:
✅ Position-aware features (current positions, entry prices, P&L)
✅ All original stock-agnostic features preserved
✅ Dynamic feature generation based on position state
✅ No lookahead bias maintained

NEW POSITION FEATURES:
1. current_position: -1 to +1 (short/neutral/long)
2. position_age_normalized: 0 to 1 (normalized by 1 week)
3. unrealized_pnl: % profit/loss from entry
4. entry_signal_strength: Signal when position opened
5. signal_change: entry_signal - current_signal
6. position_size_normalized: 0 to 1 (% of portfolio)
7. is_winning: 1 if P&L > 0, else 0
8. is_losing: 1 if P&L < 0, else 0
9. days_since_signal_peak: How long since signal was strongest
10. volatility_since_entry: Vol since position opened
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Import original feature engineering
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import original classes
from Utils.Feature_engineering import (
    StockAgnosticConfig,
    StockAgnosticFeatureEngineer,
    create_features_from_ohlcv,
    normalize_features
)


@dataclass
class Position:
    """Represents a current trading position"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    entry_signal: float
    portfolio_weight: float  # % of portfolio
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L as percentage"""
        if self.direction == 'LONG':
            return (self.current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - self.current_price) / self.entry_price
    
    @property
    def age_hours(self) -> float:
        """Position age in hours"""
        return (datetime.now() - self.entry_time).total_seconds() / 3600
    
    @property
    def age_days(self) -> float:
        """Position age in days"""
        return self.age_hours / 24
    
    @property
    def is_winning(self) -> bool:
        """Is position profitable?"""
        return self.unrealized_pnl_pct > 0
    
    @property
    def is_losing(self) -> bool:
        """Is position losing?"""
        return self.unrealized_pnl_pct < 0


class PositionAwareFeatureEngineer(StockAgnosticFeatureEngineer):
    """
    Enhanced feature engineer that includes position-aware features
    
    Inherits all stock-agnostic features, adds position context
    """
    
    def __init__(self, config: Optional[StockAgnosticConfig] = None):
        super().__init__(config)
        self.position_feature_names = [
            'current_position',
            'position_age_normalized',
            'unrealized_pnl',
            'entry_signal_strength',
            'signal_change',
            'position_size_normalized',
            'is_winning',
            'is_losing',
            'days_since_signal_peak',
            'volatility_since_entry'
        ]
    
    def transform_with_positions(self,
                                 df: pd.DataFrame,
                                 symbol: str,
                                 position: Optional[Position] = None,
                                 signal_history: Optional[pd.Series] = None,
                                 current_signal: Optional[float] = None) -> pd.DataFrame:
        """
        Transform OHLCV data with position-aware features
        
        Args:
            df: DataFrame with OHLCV columns
            symbol: Stock symbol
            position: Current position (if any)
            signal_history: Historical signals for this stock
            current_signal: Current model signal
        
        Returns:
            DataFrame with stock-agnostic + position-aware features
        """
        # Get base stock-agnostic features
        features = self.transform(df, symbol)
        
        # Add position-aware features
        position_features = self._create_position_features(
            df, position, signal_history, current_signal
        )
        
        # Combine
        for col, values in position_features.items():
            features[col] = values
        
        return features
    
    def _create_position_features(self,
                                  df: pd.DataFrame,
                                  position: Optional[Position],
                                  signal_history: Optional[pd.Series],
                                  current_signal: Optional[float]) -> Dict[str, np.ndarray]:
        """Create position-aware features"""
        
        n_samples = len(df)
        features = {}
        
        if position is None:
            # No position - all zeros
            features['current_position'] = np.zeros(n_samples)
            features['position_age_normalized'] = np.zeros(n_samples)
            features['unrealized_pnl'] = np.zeros(n_samples)
            features['entry_signal_strength'] = np.zeros(n_samples)
            features['signal_change'] = np.zeros(n_samples)
            features['position_size_normalized'] = np.zeros(n_samples)
            features['is_winning'] = np.zeros(n_samples)
            features['is_losing'] = np.zeros(n_samples)
            features['days_since_signal_peak'] = np.zeros(n_samples)
            features['volatility_since_entry'] = np.zeros(n_samples)
        
        else:
            # Has position - calculate features
            
            # 1. Current position direction and size
            if position.direction == 'LONG':
                position_value = position.portfolio_weight
            else:  # SHORT
                position_value = -position.portfolio_weight
            features['current_position'] = np.full(n_samples, position_value)
            
            # 2. Position age (normalized by 1 week = 168 hours)
            age_normalized = min(position.age_hours / 168.0, 1.0)
            features['position_age_normalized'] = np.full(n_samples, age_normalized)
            
            # 3. Unrealized P&L
            features['unrealized_pnl'] = np.full(n_samples, position.unrealized_pnl_pct)
            
            # 4. Entry signal strength
            features['entry_signal_strength'] = np.full(n_samples, abs(position.entry_signal))
            
            # 5. Signal change (how much signal weakened/strengthened)
            if current_signal is not None:
                signal_change = position.entry_signal - current_signal
            else:
                signal_change = 0
            features['signal_change'] = np.full(n_samples, signal_change)
            
            # 6. Position size (normalized)
            features['position_size_normalized'] = np.full(n_samples, position.portfolio_weight)
            
            # 7. Is winning/losing
            features['is_winning'] = np.full(n_samples, 1.0 if position.is_winning else 0.0)
            features['is_losing'] = np.full(n_samples, 1.0 if position.is_losing else 0.0)
            
            # 8. Days since signal was at peak
            if signal_history is not None and len(signal_history) > 0:
                # Find when signal was strongest
                peak_idx = signal_history.abs().idxmax()
                peak_time = signal_history.index[peak_idx]
                days_since_peak = (datetime.now() - peak_time).total_seconds() / 86400
                days_since_peak = min(days_since_peak, 30)  # Cap at 30 days
            else:
                days_since_peak = 0
            features['days_since_signal_peak'] = np.full(n_samples, days_since_peak / 30.0)
            
            # 9. Volatility since entry
            if len(df) > 0:
                # Calculate returns since entry
                entry_idx = df.index.get_indexer([position.entry_time], method='nearest')[0]
                if entry_idx < len(df):
                    returns_since_entry = df['close'].iloc[entry_idx:].pct_change()
                    vol_since_entry = returns_since_entry.std()
                else:
                    vol_since_entry = 0
            else:
                vol_since_entry = 0
            features['volatility_since_entry'] = np.full(n_samples, vol_since_entry)
        
        return features


def create_features_with_positions(df: pd.DataFrame,
                                   symbol: str,
                                   position: Optional[Position] = None,
                                   signal_history: Optional[pd.Series] = None,
                                   current_signal: Optional[float] = None,
                                   config: Optional[StockAgnosticConfig] = None) -> pd.DataFrame:
    """
    Convenience function to create features with position awareness
    
    Args:
        df: OHLCV DataFrame
        symbol: Stock symbol
        position: Current position (if any)
        signal_history: Historical signals
        current_signal: Current model signal
        config: Feature configuration
    
    Returns:
        DataFrame with all features
    """
    engineer = PositionAwareFeatureEngineer(config)
    return engineer.transform_with_positions(df, symbol, position, signal_history, current_signal)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("POSITION-AWARE FEATURE ENGINEERING DEMO")
    print("=" * 80)
    
    # Create sample data
    dates = pd.date_range('2024-11-01', '2024-11-29', freq='1H')
    np.random.seed(42)
    
    price = 180
    prices = [price]
    for _ in range(len(dates) - 1):
        price *= (1 + np.random.normal(0, 0.01))
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    df = df.set_index('timestamp')
    
    print("\n[SCENARIO 1: No Position]")
    print("-" * 80)
    
    features_no_position = create_features_with_positions(
        df, 
        symbol='AAPL',
        position=None
    )
    
    print(f"Features created: {len(features_no_position.columns)}")
    print(f"Position features:")
    position_cols = [c for c in features_no_position.columns if 'position' in c or 'unrealized' in c or 'entry' in c]
    for col in position_cols:
        print(f"  {col}: {features_no_position[col].iloc[-1]:.4f}")
    
    print("\n[SCENARIO 2: Winning Long Position]")
    print("-" * 80)
    
    position_long = Position(
        symbol='AAPL',
        direction='LONG',
        quantity=100,
        entry_price=175.0,
        current_price=182.5,
        entry_time=datetime.now() - timedelta(hours=48),
        entry_signal=0.73,
        portfolio_weight=0.15
    )
    
    print(f"Position: {position_long.direction}")
    print(f"Entry: ${position_long.entry_price:.2f}")
    print(f"Current: ${position_long.current_price:.2f}")
    print(f"P&L: {position_long.unrealized_pnl_pct:+.2%}")
    print(f"Age: {position_long.age_hours:.1f} hours ({position_long.age_days:.1f} days)")
    
    features_long = create_features_with_positions(
        df,
        symbol='AAPL',
        position=position_long,
        current_signal=0.68  # Signal slightly weakened from 0.73
    )
    
    print(f"\nPosition features:")
    for col in position_cols:
        if col in features_long.columns:
            print(f"  {col}: {features_long[col].iloc[-1]:.4f}")
    
    print("\n[SCENARIO 3: Losing Short Position]")
    print("-" * 80)
    
    position_short = Position(
        symbol='TSLA',
        direction='SHORT',
        quantity=50,
        entry_price=250.0,
        current_price=258.0,
        entry_time=datetime.now() - timedelta(hours=72),
        entry_signal=-0.58,
        portfolio_weight=0.12
    )
    
    print(f"Position: {position_short.direction}")
    print(f"Entry: ${position_short.entry_price:.2f}")
    print(f"Current: ${position_short.current_price:.2f}")
    print(f"P&L: {position_short.unrealized_pnl_pct:+.2%}")
    print(f"Age: {position_short.age_hours:.1f} hours ({position_short.age_days:.1f} days)")
    
    features_short = create_features_with_positions(
        df,
        symbol='TSLA',
        position=position_short,
        current_signal=-0.45  # Signal weakened from -0.58 to -0.45
    )
    
    print(f"\nPosition features:")
    for col in position_cols:
        if col in features_short.columns:
            print(f"  {col}: {features_short[col].iloc[-1]:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ POSITION-AWARE FEATURES WORKING!")
    print("=" * 80)
    
    print(f"\nTotal features per scenario: {len(features_long.columns)}")
    print(f"Stock-agnostic features: {len(features_long.columns) - 10}")
    print(f"Position-aware features: 10")
    
    print("\nKey Benefits:")
    print("  ✅ Model knows if it's already in a position")
    print("  ✅ Model knows if position is winning/losing")
    print("  ✅ Model knows how long position has been held")
    print("  ✅ Model knows how much signal has changed")
    print("  ✅ Model can make better exit decisions!")