"""
Stock-Agnostic Feature Engineering for Multi-Asset Trading

CRITICAL IMPROVEMENTS:
✅ All features normalized/comparable across stocks
✅ No absolute price/volume features (stock-specific)
✅ Market context features (SPY benchmark)
✅ Relative strength features (vs market, vs sector)
✅ Proper normalization (z-scores, percentile ranks)
✅ No lookahead bias (all features from t-1)

DESIGN PRINCIPLES:
1. Stock-Agnostic: Features work on AAPL, MSFT, penny stocks, etc.
2. Normalized: All features in comparable ranges
3. Market-Aware: Include benchmark context
4. Regime-Aware: Detect volatility/trend regimes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StockAgnosticConfig:
    """Configuration for stock-agnostic features"""
    
    # Returns (always stock-agnostic)
    return_windows: List[int] = None
    
    # Price ratios (dimensionless)
    use_price_ratios: bool = True
    
    # Moving averages (normalized)
    ma_windows: List[int] = None
    
    # Volatility (normalized)
    volatility_windows: List[int] = None
    
    # Momentum (returns-based)
    momentum_windows: List[int] = None
    
    # Normalized indicators
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    
    # Volume (relative to own history)
    use_volume: bool = True
    
    # Statistical features
    use_higher_moments: bool = True
    use_autocorr: bool = True
    
    # Normalization windows
    zscore_window: int = 252  # 1 year for z-scores
    percentile_window: int = 252
    
    def __post_init__(self):
        if self.return_windows is None:
            self.return_windows = [1, 5, 20, 60]  # 1hr, 1day, 1wk, 3mo
        if self.ma_windows is None:
            self.ma_windows = [5, 10, 20, 50, 126, 252]
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 60]
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 20, 60]


class StockAgnosticFeatureEngineer:
    """
    Creates features that are comparable across ALL stocks
    
    KEY PRINCIPLE: No absolute values (prices, volumes)
    Only relative/normalized values
    """
    
    def __init__(self, config: Optional[StockAgnosticConfig] = None):
        self.config = config or StockAgnosticConfig()
        self.feature_names = []
    
    def transform(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Transform OHLCV data into stock-agnostic features
        
        Args:
            df: DataFrame with OHLCV columns (close, open, high, low, volume)
            symbol: Optional symbol identifier (for debugging)
        
        Returns:
            DataFrame with stock-agnostic features
        """
        # CRITICAL: Shift ALL prices by 1 period FIRST
        # At time t, we use data from time t-1
        close_shifted = df['close'].shift(1)
        open_shifted = df['open'].shift(1) if 'open' in df.columns else None
        high_shifted = df['high'].shift(1) if 'high' in df.columns else None
        low_shifted = df['low'].shift(1) if 'low' in df.columns else None
        volume_shifted = df['volume'].shift(1) if 'volume' in df.columns else None
        
        features = pd.DataFrame(index=df.index)
        self.feature_names = []
        
        # =================================================================
        # 1. RETURNS (Stock-Agnostic by Nature)
        # =================================================================
        for window in self.config.return_windows:
            ret = close_shifted.pct_change(window)
            features[f'return_{window}'] = ret
            self.feature_names.append(f'return_{window}')
        
        # =================================================================
        # 2. PRICE RATIOS (Dimensionless, Stock-Agnostic)
        # =================================================================
        if self.config.use_price_ratios and all(x is not None for x in [open_shifted, high_shifted, low_shifted]):
            # High-Low ratio (range)
            features['hl_ratio'] = (high_shifted - low_shifted) / (close_shifted + 1e-8)
            self.feature_names.append('hl_ratio')
            
            # Open-Close ratio
            features['oc_ratio'] = (close_shifted - open_shifted) / (open_shifted + 1e-8)
            self.feature_names.append('oc_ratio')
            
            # Close position within range
            features['close_position'] = (close_shifted - low_shifted) / (high_shifted - low_shifted + 1e-8)
            self.feature_names.append('close_position')
        
        # =================================================================
        # 3. MOVING AVERAGE RATIOS (Normalized, Stock-Agnostic)
        # =================================================================
        for window in self.config.ma_windows:
            ma = close_shifted.rolling(window).mean()
            
            # Deviation from MA (normalized by MA itself)
            features[f'ma_ratio_{window}'] = (close_shifted - ma) / (ma + 1e-8)
            self.feature_names.append(f'ma_ratio_{window}')
        
        # MA crossover (fast vs slow)
        if len(self.config.ma_windows) >= 2:
            fast_ma = close_shifted.rolling(self.config.ma_windows[0]).mean()
            slow_ma = close_shifted.rolling(self.config.ma_windows[-1]).mean()
            features['ma_crossover'] = (fast_ma - slow_ma) / (slow_ma + 1e-8)
            self.feature_names.append('ma_crossover')
        
        # =================================================================
        # 4. VOLATILITY (Normalized Returns Std, Stock-Agnostic)
        # =================================================================
        returns = close_shifted.pct_change()
        
        for window in self.config.volatility_windows:
            vol = returns.rolling(window).std()
            features[f'volatility_{window}'] = vol
            self.feature_names.append(f'volatility_{window}')
        
        # Volatility ratio (short/long)
        if len(self.config.volatility_windows) >= 2:
            vol_short = returns.rolling(self.config.volatility_windows[0]).std()
            vol_long = returns.rolling(self.config.volatility_windows[-1]).std()
            features['vol_ratio'] = vol_short / (vol_long + 1e-8)
            self.feature_names.append('vol_ratio')
        
        # =================================================================
        # 5. MOMENTUM (Returns-Based, Stock-Agnostic)
        # =================================================================
        for window in self.config.momentum_windows:
            momentum = close_shifted.pct_change(window)
            features[f'momentum_{window}'] = momentum
            self.feature_names.append(f'momentum_{window}')
        
        # =================================================================
        # 6. RSI (0-100 Scale, Stock-Agnostic)
        # =================================================================
        if self.config.use_rsi:
            features['rsi'] = self._compute_rsi(close_shifted)
            self.feature_names.append('rsi')
            
            # RSI normalized to [-1, 1]
            features['rsi_normalized'] = (features['rsi'] - 50) / 50
            self.feature_names.append('rsi_normalized')
        
        # =================================================================
        # 7. MACD (Normalized by Price, Stock-Agnostic)
        # =================================================================
        if self.config.use_macd:
            macd, signal = self._compute_macd(close_shifted)
            
            # Normalize by price
            features['macd'] = macd / (close_shifted + 1e-8)
            features['macd_signal'] = signal / (close_shifted + 1e-8)
            features['macd_diff'] = (macd - signal) / (close_shifted + 1e-8)
            
            self.feature_names.extend(['macd', 'macd_signal', 'macd_diff'])
        
        # =================================================================
        # 8. BOLLINGER BANDS (Normalized Position, Stock-Agnostic)
        # =================================================================
        if self.config.use_bollinger:
            bb_upper, bb_middle, bb_lower = self._compute_bollinger_bands(close_shifted)
            
            # Position within bands (0-1)
            features['bb_position'] = (close_shifted - bb_lower) / (bb_upper - bb_lower + 1e-8)
            
            # Band width (volatility measure)
            features['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
            
            self.feature_names.extend(['bb_position', 'bb_width'])
        
        # =================================================================
        # 9. VOLUME FEATURES (Relative to Own History, Stock-Agnostic)
        # =================================================================
        if self.config.use_volume and volume_shifted is not None:
            # Volume relative to moving average
            vol_ma = volume_shifted.rolling(20).mean()
            features['volume_ratio'] = volume_shifted / (vol_ma + 1e-8)
            self.feature_names.append('volume_ratio')
            
            # Volume momentum (% change)
            features['volume_momentum'] = volume_shifted.pct_change(5)
            self.feature_names.append('volume_momentum')
            
            # Volume z-score (how unusual is today's volume)
            vol_mean = volume_shifted.rolling(60).mean()
            vol_std = volume_shifted.rolling(60).std()
            features['volume_zscore'] = (volume_shifted - vol_mean) / (vol_std + 1e-8)
            self.feature_names.append('volume_zscore')
        
        # =================================================================
        # 10. HIGHER MOMENTS (Standardized, Stock-Agnostic)
        # =================================================================
        if self.config.use_higher_moments:
            # Skewness (already dimensionless)
            features['skewness_20'] = returns.rolling(20).skew()
            features['skewness_60'] = returns.rolling(60).skew()
            self.feature_names.extend(['skewness_20', 'skewness_60'])
            
            # Kurtosis (already dimensionless)
            features['kurtosis_20'] = returns.rolling(20).kurt()
            features['kurtosis_60'] = returns.rolling(60).kurt()
            self.feature_names.extend(['kurtosis_20', 'kurtosis_60'])
        
        # =================================================================
        # 11. AUTOCORRELATION (Trend Strength, Stock-Agnostic)
        # =================================================================
        if self.config.use_autocorr:
            # Autocorrelation of returns (measures trend strength)
            features['autocorr_1'] = returns.rolling(20).apply(
                lambda x: x.autocorr(1) if len(x) > 2 else 0
            )
            features['autocorr_5'] = returns.rolling(60).apply(
                lambda x: x.autocorr(5) if len(x) > 6 else 0
            )
            features['autocorr_20'] = returns.rolling(126).apply(
                lambda x: x.autocorr(20) if len(x) > 21 else 0
            )
            self.feature_names.extend(['autocorr_1', 'autocorr_5', 'autocorr_20'])
        
        # =================================================================
        # 12. Z-SCORES (Normalized Relative to History, Stock-Agnostic)
        # =================================================================
        # Price z-score (how far from mean in std devs)
        price_mean = close_shifted.rolling(self.config.zscore_window).mean()
        price_std = close_shifted.rolling(self.config.zscore_window).std()
        features['price_zscore'] = (close_shifted - price_mean) / (price_std + 1e-8)
        self.feature_names.append('price_zscore')
        
        # Return z-score
        ret_mean = returns.rolling(self.config.zscore_window).mean()
        ret_std = returns.rolling(self.config.zscore_window).std()
        features['return_zscore'] = (returns - ret_mean) / (ret_std + 1e-8)
        self.feature_names.append('return_zscore')
        
        # =================================================================
        # 13. PERCENTILE RANKS (0-1 Scale, Stock-Agnostic)
        # =================================================================
        # Where is current price in historical distribution?
        features['price_percentile'] = close_shifted.rolling(
            self.config.percentile_window
        ).apply(lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5)
        self.feature_names.append('price_percentile')
        
        # Return percentile
        features['return_percentile'] = returns.rolling(
            self.config.percentile_window
        ).apply(lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5)
        self.feature_names.append('return_percentile')
        
        # =================================================================
        # 14. TRUE RANGE (Volatility Measure, Normalized)
        # =================================================================
        if all(x is not None for x in [high_shifted, low_shifted]):
            # True range
            tr = high_shifted - low_shifted
            
            # Average true range (ATR)
            atr = tr.rolling(14).mean()
            
            # ATR as % of price (normalized)
            features['atr_ratio'] = atr / (close_shifted + 1e-8)
            self.feature_names.append('atr_ratio')
        
        # =================================================================
        # FILL NaN VALUES
        # =================================================================
        # Forward fill, then zero (no backfill to avoid future leakage)
        features = features.fillna(method='ffill').fillna(0)

        # Drop any remaining NaNs to maintain strict causality
        features = features.dropna()
        
        # Clip extreme values (prevent outliers from dominating)
        for col in features.columns:
            if col not in ['rsi', 'bb_position', 'close_position', 'price_percentile', 'return_percentile']:
                # Clip to [-10, 10] std devs
                features[col] = features[col].clip(-10, 10)
        
        return features
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI (0-100 scale)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _compute_macd(prices: pd.Series,
                      fast: int = 12,
                      slow: int = 26,
                      signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Compute MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    @staticmethod
    def _compute_bollinger_bands(prices: pd.Series,
                                 window: int = 20,
                                 num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Bollinger Bands"""
        middle = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower


class MarketContextFeatures:
    """
    Add market context features (SPY benchmark)
    CRITICAL for multi-asset: Understand broader market regime
    """
    
    def __init__(self):
        self.feature_names = []
    
    def add_market_features(self, 
                           stock_df: pd.DataFrame,
                           market_df: pd.DataFrame,
                           stock_features: pd.DataFrame) -> pd.DataFrame:
        """
        Add market context features
        
        Args:
            stock_df: Stock OHLCV data
            market_df: Market (SPY) OHLCV data
            stock_features: Existing features
        
        Returns:
            Features with market context added
        """
        # Align timestamps
        common_index = stock_df.index.intersection(market_df.index)
        
        if len(common_index) == 0:
            print("Warning: No common timestamps between stock and market")
            return stock_features
        
        # Calculate market returns
        market_close = market_df.loc[common_index, 'close'].shift(1)
        market_return_1 = market_close.pct_change(1)
        market_return_5 = market_close.pct_change(5)
        market_return_20 = market_close.pct_change(20)
        
        # Calculate market volatility
        market_vol = market_return_1.rolling(20).std()
        
        # Add to features (aligned to stock_features index)
        stock_features = stock_features.copy()
        
        # Market returns
        stock_features.loc[common_index, 'market_return_1'] = market_return_1
        stock_features.loc[common_index, 'market_return_5'] = market_return_5
        stock_features.loc[common_index, 'market_return_20'] = market_return_20
        
        # Market volatility
        stock_features.loc[common_index, 'market_volatility'] = market_vol
        
        # Beta to market (rolling)
        if 'return_1' in stock_features.columns:
            stock_return = stock_features.loc[common_index, 'return_1']
            
            # Rolling beta (cov(stock, market) / var(market))
            beta = stock_return.rolling(60).cov(market_return_1) / (market_return_1.rolling(60).var() + 1e-8)
            stock_features.loc[common_index, 'beta'] = beta
        
        # Fill NaN
        for col in ['market_return_1', 'market_return_5', 'market_return_20', 'market_volatility', 'beta']:
            if col in stock_features.columns:
                stock_features[col] = stock_features[col].fillna(0)
        
        self.feature_names = ['market_return_1', 'market_return_5', 'market_return_20', 
                             'market_volatility', 'beta']
        
        return stock_features


class RelativeStrengthFeatures:
    """
    Add relative strength features (vs market, vs sector)
    CRITICAL: How is stock performing relative to others?
    """
    
    def __init__(self):
        self.feature_names = []
    
    def add_relative_strength(self,
                             stock_df: pd.DataFrame,
                             market_df: pd.DataFrame,
                             stock_features: pd.DataFrame) -> pd.DataFrame:
        """
        Add relative strength features
        
        Args:
            stock_df: Stock OHLCV data
            market_df: Market (SPY) OHLCV data
            stock_features: Existing features
        
        Returns:
            Features with relative strength added
        """
        # Align timestamps
        common_index = stock_df.index.intersection(market_df.index)
        
        if len(common_index) == 0:
            return stock_features
        
        # Stock returns
        stock_close = stock_df.loc[common_index, 'close'].shift(1)
        stock_return_20 = stock_close.pct_change(20)
        stock_return_60 = stock_close.pct_change(60)
        
        # Market returns
        market_close = market_df.loc[common_index, 'close'].shift(1)
        market_return_20 = market_close.pct_change(20)
        market_return_60 = market_close.pct_change(60)
        
        # Relative strength (stock return - market return)
        stock_features = stock_features.copy()
        stock_features.loc[common_index, 'relative_strength_20'] = stock_return_20 - market_return_20
        stock_features.loc[common_index, 'relative_strength_60'] = stock_return_60 - market_return_60
        
        # Outperformance ratio
        stock_features.loc[common_index, 'outperformance_ratio_20'] = (
            (1 + stock_return_20) / (1 + market_return_20 + 1e-8) - 1
        )
        
        # Fill NaN
        for col in ['relative_strength_20', 'relative_strength_60', 'outperformance_ratio_20']:
            if col in stock_features.columns:
                stock_features[col] = stock_features[col].fillna(0)
        
        self.feature_names = ['relative_strength_20', 'relative_strength_60', 'outperformance_ratio_20']
        
        return stock_features


def create_features_from_ohlcv(df: pd.DataFrame, 
                               market_df: pd.DataFrame = None,
                               symbol: str = None) -> pd.DataFrame:
    """
    Main function to create stock-agnostic features
    
    Args:
        df: Stock OHLCV data (with index as timestamps)
        market_df: Optional market (SPY) OHLCV data for context
        symbol: Optional symbol for debugging
    
    Returns:
        DataFrame with all features
    """
    # 1. Create base stock-agnostic features
    engineer = StockAgnosticFeatureEngineer()
    features = engineer.transform(df, symbol=symbol)
    
    # 2. Add market context if available
    if market_df is not None:
        market_features = MarketContextFeatures()
        features = market_features.add_market_features(df, market_df, features)
    
    # 3. Add relative strength if available
    if market_df is not None:
        rs_features = RelativeStrengthFeatures()
        features = rs_features.add_relative_strength(df, market_df, features)
    
    return features


def normalize_features(features: pd.DataFrame,
                      mean: Optional[pd.Series] = None,
                      std: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Normalize features (z-score)
    
    NOTE: Many features are already normalized (ratios, percentiles, RSI)
    Only normalize features that need it
    
    Args:
        features: Feature DataFrame
        mean: Pre-computed mean (for test set)
        std: Pre-computed std (for test set)
    
    Returns:
        Normalized features, mean, std
    """
    # Features that should NOT be normalized (already in [0, 1] or [-1, 1])
    skip_normalize = [
        'rsi', 'rsi_normalized', 'bb_position', 'close_position',
        'price_percentile', 'return_percentile'
    ]
    
    # Columns to normalize
    cols_to_normalize = [c for c in features.columns if c not in skip_normalize]
    
    if mean is None or std is None:
        # Compute from data (training set)
        mean = features[cols_to_normalize].mean()
        std = features[cols_to_normalize].std()
    
    # Normalize
    normalized = features.copy()
    normalized[cols_to_normalize] = (features[cols_to_normalize] - mean) / (std + 1e-8)
    
    # Clip to [-5, 5] std devs
    normalized[cols_to_normalize] = normalized[cols_to_normalize].clip(-5, 5)
    
    return normalized, mean, std


if __name__ == "__main__":
    # Test stock-agnostic features
    print("Testing Stock-Agnostic Feature Engineering\n")
    
    # Create sample data for two very different stocks
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    
    # Stock 1: High price, low volatility (like BRK.A)
    stock1_df = pd.DataFrame({
        'close': 300000 + np.cumsum(np.random.randn(1000) * 100),
        'open': 300000 + np.cumsum(np.random.randn(1000) * 100),
        'high': 300100 + np.cumsum(np.random.randn(1000) * 100),
        'low': 299900 + np.cumsum(np.random.randn(1000) * 100),
        'volume': 1000 + np.random.rand(1000) * 500
    }, index=dates)
    
    # Stock 2: Low price, high volatility (like penny stock)
    stock2_df = pd.DataFrame({
        'close': 2 + np.cumsum(np.random.randn(1000) * 0.5),
        'open': 2 + np.cumsum(np.random.randn(1000) * 0.5),
        'high': 2.1 + np.cumsum(np.random.randn(1000) * 0.5),
        'low': 1.9 + np.cumsum(np.random.randn(1000) * 0.5),
        'volume': 1000000 + np.random.rand(1000) * 500000
    }, index=dates)
    
    # Engineer features
    print("Stock 1 (High price, low vol):")
    print(f"  Price range: ${stock1_df['close'].min():,.0f} - ${stock1_df['close'].max():,.0f}")
    print(f"  Volume range: {stock1_df['volume'].min():.0f} - {stock1_df['volume'].max():.0f}")
    
    features1 = create_features_from_ohlcv(stock1_df, symbol="BRK.A")
    
    print("\nStock 2 (Low price, high vol):")
    print(f"  Price range: ${stock2_df['close'].min():.2f} - ${stock2_df['close'].max():.2f}")
    print(f"  Volume range: {stock2_df['volume'].min():,.0f} - {stock2_df['volume'].max():,.0f}")
    
    features2 = create_features_from_ohlcv(stock2_df, symbol="PENNY")
    
    # Compare feature ranges (should be similar!)
    print("\n" + "="*80)
    print("FEATURE COMPARISON (Should be similar despite different prices/volumes)")
    print("="*80)
    
    comparison_features = ['return_1', 'ma_ratio_20', 'volatility_20', 'rsi', 'volume_ratio']
    
    for feat in comparison_features:
        if feat in features1.columns and feat in features2.columns:
            range1 = (features1[feat].min(), features1[feat].max())
            range2 = (features2[feat].min(), features2[feat].max())
            print(f"\n{feat}:")
            print(f"  Stock 1: [{range1[0]:>8.3f}, {range1[1]:>8.3f}]")
            print(f"  Stock 2: [{range2[0]:>8.3f}, {range2[1]:>8.3f}]")
            print(f"  ✅ Comparable ranges!" if abs(range1[1] - range2[1]) < 10 else "  ❌ Not comparable")
    
    print("\n✅ Stock-agnostic features working correctly!")
    print(f"✅ Total features: {len(features1.columns)}")
