"""Stock-Agnostic Feature Engineering for Multi-Asset Trading

DESIGN PRINCIPLES:
1. Stock-Agnostic: Features work on AAPL, MSFT, penny stocks, etc.
2. Normalized: All features in comparable ranges
3. Market-Aware: Include benchmark context
4. Regime-Aware: Detect volatility/trend regimes
"""

from __future__ import annotations

from dataclasses import dataclass, field 
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np 
import pandas as pd

from Models.config import BARS_PER_DAY, bars

EPS = 1e-8

# Columns that are already bounded and must not be z-scored
BOUNDED_FEATURES = {"bb_position", "close_position", "price_percentile",
                    "return_percentile"}

@dataclass
class FeatureConfig:
    """ Feature windows, all declared in **trading days** and converted to bars.
    
    Keeping the declaration in days is what makes the horizons auditable"""

    return_days: List[float] = field(
        default_factory = lambda: [1 / BARS_PER_DAY, 1, 5, 21, 63, 126]
    )

    ma_days: List[float] = field(default_factory = lambda: [1, 3, 5, 10, 21, 63, 126])
    volatility_days: List[float] = field(default_factory = lambda: [1, 3, 5, 21])
    momentum_days: List[float] = field(default_factory = lambda: [5, 21, 63, 126])

    rsi_days: float = 2
    macd_fast_days: float = 2
    macd_slow_days: float = 4
    macd_signal_days: float = 1.5
    bollinger_days: float = 3
    atr_days: float = 2

    volume_ma_days: float = 3
    volume_zscore_days: float = 9

    zscore_days: float = 63
    percentile_days: float = 63

    # Ex-ante volatility used to risk-adjust momentum
    vol_scaling_days: float = 9

    use_volume: bool = True
    use_higher_moments: bool = True
    use_autocorr: bool = True 

    def window(self, days: float) -> int:
        return bars(days)

    @property
    def max_lookback(self) -> int: 
        """ Longest window any feature depends on, in bars."""
        candidates = (
            [self.window(d) for d in self.return_days]
            + [self.window(d) for d in self.ma_days]
            + [self.window(d) for d in self.volatility_days]
            + [self.window(d) for d in self.momentum_days]
            + [self.window(self.zscore_days), self.window(self.percentile_days),
                self.window(self.volume_zscore_days), self.window(self.vol_scaling_days)]
        )
        return int(max(candidates))

class StockAgnosticFeatureEngineer:
    """Builds the per-symbol, dimensionless feature frame"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []

    # -- indicators --
    @staticmethod
    def _wilder_rsi(prices: pd.Series, window: int) -> pd.Series:
        """ RSI using Wilder's smoothing (the standard), not a simple average"""
        delta = prices.diff()
        gain = delta.clip(lower = 0.0)
        loss = (-delta).clip(lower = 0.0)

        avg_gain = gain.ewm(alpha = 1.0 / window, adjust = False, min_periods = window).mean()
        avg_loss = loss.ewm(alpha = 1.0 / window, adjust = False, min_periods = window).mean() 

        rs = avg_gain / (avg_loss + EPS)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod 
    def _macd(
        prices: pd.Series, fast: int, slow: int, signal: int
    ) -> Tuple[pd.Series, pd.Series]:
        """ MACD with ``adjust = False ``, matching standard charting packages"""
        ema_fast = prices.ewm(span = fast, adjust = False, min_periods = fast).mean()
        ema_slow = prices.ewm(span = slow, adjust = False, min_periods = slow).mean()
        macd = ema_fast - ema_slow 
        signal_line = macd.ewm(span = signal, adjust = False, min_periods = signal).mean()
        return macd, signal_line 

    @staticmethod 
    def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
        """ Fraction of the window at or below the current value, in [0,1].
        
        `` rolling().rank(pct = True)`` ranks the *last* element of each window,
        which is what "where is the current value in its recent distribution" means -- 
        and it is vectorized"""
        return series.rolling(window, min_periods = window).rank(pct = True)

    # -- main -- 
    def transform(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        cfg = self.config 

        # Shift ALL raw inputs first: at time t we may only use data from t-1 
        close = df["close"].shift(1)
        open_ = df["open"].shift(1) if "open" in df else None 
        high = df["high"].shift(1) if "high" in df else None 
        low = df["low"].shift(1) if "low" in df else None 
        volume = df["volume"].shift(1) if "volume" in df else None 

        features = pd.DataFrame(index = df.index)
        names: List[str] = []
        returns = close.pct_change() 

        def add(name: str, series: pd.Series):
            features[name] = series
            names.append(name)

        # 1. Returns over multiple horizons 
        for days in cfg.return_days:
            w = cfg.window(days)
            add(f"return_{w}", close.pct_change(w))

        # 2. Intrabar price ratios
        if open_ is not None and high is not None and low is not None:
            add("hl_ratio", (high - low) / (close + EPS))
            add("oc_ratio", (close - open_) / (open_ + EPS))
            add("close_position", (close - low) / (high - low + EPS))

        # 3. Moving average ratios 
        ma_windows = [cfg.window(d) for d in cfg.ma_days]
        for w in ma_windows:
            ma = close.rolling(w, min_periods = w).mean()
            add(f"ma_ratio_{w}", (close - ma) / (ma + EPS))

        if len(ma_windows) >= 3:
            # Fast vs a MID window. Crossing the longest window duplicates 
            # ma_ratio <longest> almost exactly (measured rho = 0.996), because 
            # both are essentially "price relative to its slow average"
            fast = close.rolling(ma_windows[0], min_periods = ma_windows[0]).mean()
            mid = ma_windows[len(ma_windows) // 2]
            slow = close.rolling(mid, min_periods = mid).mean()
            add("ma_crossover", (fast - slow) / (slow + EPS))

        # 4. Volatility
        vol_windows = [cfg.window(d) for d in cfg.volatility_days]
        for w in vol_windows:
            add(f"volatility_{w}", returns.rolling(w, min_periods = w).std())

        if len(vol_windows) >= 2:
            vs = returns.rolling(vol_windows[0], min_periods = vol_windows[0]).std()
            vl = returns.rolling(vol_windows[-1], min_periods = vol_windows[-1]).std()
            add("vol_ratio", vs / (vl + EPS))

        # 5. Risk-adjust momentum -- r_{t-w, t} / (sigma * sqrt(w)),
        # This is the Deep Momentum Networks normalization and is what makes 
        # momentum_ * genuinely distinct from return_*
        vol_w = cfg.window(cfg.vol_scaling_days)
        ex_ante_vol = returns.ewm(span = vol_w, adjust = False, min_periods = vol_w).std()
        for days in cfg.momentum_days:
            w = cfg.window(days)
            cum = close.pct_change(w)
            add(f"momentum_{w}", cum / (ex_ante_vol * np.sqrt(w) + EPS))

        # 6. RSI (Wilder). Kept on its native 0-100 scale here; the normalizer
        # puts it on the same footing as everything else 
        rsi_w = cfg.window(cfg.rsi_days)
        add("rsi", self._wilder_rsi(close, rsi_w))

        #7. MACD, normalized by price
        macd, signal = self._macd(
            close,
            cfg.window(cfg.macd_fast_days),
            cfg.window(cfg.macd_slow_days),
            cfg.window(cfg.macd_signal_days),
        )
        add("macd", macd / (close + EPS))
        add("macd_signal", signal / (close + EPS))
        add("macd_diff", (macd - signal) / (close + EPS))

        # 8. Bollinger position and width 
        bb_w = cfg.window(cfg.bollinger_days)
        bb_mid = close.rolling(bb_w, min_periods = bb_w).mean() 
        bb_std = close.rolling(bb_w, min_periods = bb_w).std() 
        bb_upper, bb_lower = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
        add("bb_position", (close - bb_lower) / (bb_upper - bb_lower + EPS))
        add("bb_width", (bb_upper - bb_lower) / (bb_mid + EPS))

        # 9: Volume, relative to the symbol's own history 
        if cfg.use_volume and volume is not None:
            vma_w = cfg.window(cfg.volume_ma_days)
            vz_w = cfg.window(cfg.volume_zscore_days)
            add("volume_ratio", 
                volume / (volume.rolling(vma_w, min_periods = vma_w).mean() + EPS))
            add("volume_momentum", volume.pct_change(vma_w))
            vmean = volume.rolling(vz_w, min_periods = vz_w).mean()
            vstd = volume.rolling(vz_w, min_periods = vz_w).std()
            add("volume_zscore", (volume - vmean) / (vstd + EPS))

        # 10. Higher moments
        if cfg.use_higher_moments:
            for days in (3, 9):
                w = cfg.window(days)
                add(f"skewness_{w}", returns.rolling(w, min_periods = w).skew())
                add(f"kurtosis_{w}", returns.rolling(w, min_periods = w).kurt())

        # 11. Autocorreltaion (vectorized via rolling corr with a lagged copy)
        if cfg.use_autocorr:
            for lag, days in ((1, 3), (5, 9), (21, 21)):
                w = cfg.window(days)
                add(f"autocorr_{lag}",
                    returns.rolling(w, min_periods = w).corr(returns.shift(lag)))
                
        # 12. Z-scores
        z_w = cfg.window(cfg.zscore_days)
        p_mean = close.rolling(z_w, min_periods = z_w).mean()
        p_std = close.rolling(z_w, min_periods = z_w).std()
        add("price_zscore", (close - p_mean) / (p_std + EPS))

        r_mean = returns.rolling(z_w, min_periods = z_w).mean()
        r_std = returns.rolling(z_w, min_periods = z_w).std()
        add("return_zscore", (returns - r_mean) / (r_std + EPS))

        # 13. Percentile ranks -- fraction of the window <= the current value
        pct_w = cfg.window(cfg.percentile_days)
        add("price_percentile", self._rolling_percentile(close, pct_w))
        add("return_percentile", self._rolling_percentile(returns, pct_w))

        # 14. ATR as a fraction of price
        if high is not None and low is not None:
            atr_w = cfg.window(cfg.atr_days)
            tr = high - low
            add("atr_ratio", tr.rolling(atr_w, min_periods = atr_w).mean() / (close + EPS))

        self.feature_names = names
        return features

class FeatureNormalizer:
    """ Z-score normalizer whose statistics are fitted on the training split only
    
    Fit once on the training data, persist alongside the model, apply unchanged to validation and test. 
    Bounded features (percentiles, Bollinger position) are passed through"""

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.mean_: Optional[pd.Series] = None 
        self.std_: Optional[pd.Series] = None 
        self.columns_: Optional[List[str]] = None 

    def fit(self, features: pd.DataFrame) -> "FeatureNormalizer":
        cols = [c for c in features.columns if c not in BOUNDED_FEATURES]
        self.columns_ = cols
        self.mean_ = features[cols].mean()
        self.std_ = features[cols].std().replace(0.0, 1.0)
        return self 

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None or self.columns_ is None:
            raise RuntimeError("FeatureNormalizer must be fitted before transform")

        out = features.copy()
        cols = [c for c in self.columns_ if c in out.columns]
        out[cols] = (out[cols] - self.mean_[cols]) / (self.std_[cols] + EPS)
        # Clip after standardizing, so the bound really is in std devs
        out[cols] = out[cols].clip(-self.clip, self.clip)
        return out 

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)

    def to_dict(self) -> Dict:
        # Explicit None checks: `self.mean_ or ...` puts a Series in boolean
        # context, which raises "truth value of a Series is ambiguous".
        return {
            "clip": self.clip,
            "columns": list(self.columns_) if self.columns_ is not None else [],
            "mean": self.mean_.to_dict() if self.mean_ is not None else {},
            "std": self.std_.to_dict() if self.std_ is not None else {},
        }

    @classmethod 
    def from_dict(cls, payload: Dict) -> "FeatureNormalizer":
        obj = cls(clip = payload.get("clip", 5.0))
        obj.columns_ = list(payload["columns"])
        obj.mean_ = pd.Series(payload["mean"])
        obj.std_ = pd.Series(payload["std"])
        return obj 

### Market Context / Relative Strength ###

MARKET_FEATURE_NAMES = [
    "market_return_1",
    "market_return_35",
    "market_return_147",
    "market_volatility",
    "beta",
]

RELATIVE_STRENGTH_FEATURE_NAMES = [
    "relative_strength_35",
    "relative_strength_147",
]

def add_market_features(
    features:pd.DataFrame,
    market_df: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:

    """ Add benchmark context, reindexed explicitly onto the symbol's index
    
    Non-overlapping timestamps become NaN and are dropped with the burn-in rows,
    rather than being filled with zeros that read as a real observation of 
    "market return exactly 0, beta exactly 0
    """

    cfg = cfg or FeatureConfig()
    market_close = market_df["close"].shift(1).reindex(features.index)

    m1 = market_close.pct_change(1)
    w_short, w_long = cfg.window(5), cfg.window(21)
    vol_w = cfg.window(3)

    out = features.copy()
    out["market_return_1"] = m1
    out["market_return_35"] = market_close.pct_change(w_short)
    out["market_return_147"] = market_close.pct_change(w_long)
    out["market_volatility"] = m1.rolling(vol_w, min_periods = vol_w).std() 

    # Beta = cov(stock, market) / var(market). Not correlation 
    if "return_1" in out.columns:
        beta_w = cfg.window(9)
        cov = out["return_1"].rolling(beta_w, min_periods = beta_w).cov(m1)
        var = m1.rolling(beta_w, min_periods = beta_w).var()
        out["beta"] = cov / (var + EPS)

    return out 

def add_relative_strength(
    features: pd.DataFrame,
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """ Add stock-versus-benchmark relative performance"""
    cfg = cfg or FeatureConfig()
    w_short, w_long = cfg.window(5), cfg.window(21)

    stock_close = stock_df["close"].shift(1).reindex(features.index)
    market_close = market_df["close"].shift(1).reindex(features.index)

    s_short, s_long = stock_close.pct_change(w_short), stock_close.pct_change(w_long)
    m_short, m_long = market_close.pct_change(w_short), market_close.pct_change(w_long)

    out = features.copy()
    out["relative_strength_35"] = s_short - m_short
    out["relative_strength_147"] = s_long - m_long 
    # `outperformance_ratio -- (1 + s)/ (1 + m) - 1 -- was dropped: for bar-scale
    # # returns it is a first-order Taylor expansion of (s-m), and measured 
    # rho against relative_strength_35 on real data was 0.9997
    return out 

def create_features_from_ohlcv(
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None, 
    config: Optional[FeatureConfig] = None, 
    drop_burn_in: bool = True,
) -> pd.DataFrame:
    """ Build the full feature frame for one symbol
    
    ``df`` and ``market_df`` must be indexed by timestamp. When ``market_df``
    is supplied the market and relative-strength blocks are added and asserted;
    passing ``None`` is only appropriate for unit tests on synthetic series
    """
    config = config or FeatureConfig() 

    engineer = StockAgnosticFeatureEngineer(config)
    features = engineer.transform(df, symbol = symbol)

    if market_df is not None:
        features = add_market_features(features, market_df, config)
        features = add_relative_strength(features, df, market_df, config)
        missing = [c for c in MARKET_FEATURE_NAMES + RELATIVE_STRENGTH_FEATURE_NAMES
                   if c not in features.columns]
        if missing:
            raise AssertionError(
                f"Market feature columns missing for {symbol!r}: {missing}"
            )
    if drop_burn_in:
        # Drop the leading rows whose long-window features are undefined, rather
        # than fabricating values for them. Anything still NaN afterwards is a 
        # genuine gap and is removed too
        features = features.iloc[config.max_lookback:]
        features = features.dropna()

    return features 

### Feature-quality checks ###

def check_collinearity(
        features: pd.DataFrame, threshold: float = 0.99
) -> List[Tuple[str, str, float]]:
    """ Return feature pairs with |rho| above ``threshold``,
    
    Guards against silently reintroducing duplicates like the previously
    verified ``return_5 == momentum_5``"""
    numeric = features.select_dtypes(include = [np.number])
    if numeric.shape[1] < 2:
        return []

    corr = numeric.corr().abs()
    corr_values = corr.to_numpy(dtype = float)
    offenders = []
    cols = [str(c) for c in corr.columns]
    for i, a in enumerate(cols):
        for j, b in enumerate(cols[i + 1:], start = i + 1):
            rho = float(corr_values[i, j])
            if np.isfinite(rho) and rho > threshold:
                offenders.append((a, b, rho))
    return sorted(offenders, key = lambda t: -t[2])

def check_feature_scales(features: pd.DataFrame, max_ratio: float = 100.0) -> Dict:
    """ Report the spread of per-feature standard deviations
    
    A ratio in the thousands means some input dominates an unnormalized network
    purely by scale -- which is what raw RSI (0-100) did next to ``return_1``
    (std 0.0056)
    """
    numeric = features.select_dtypes(include = [np.number])
    stds = numeric.std().replace(0.0, np.nan).dropna()
    if stds.empty:
        return {"ok": True, "ratio": 0.0, "min_std": 0.0, "max_std": 0.0}

    ratio = float(stds.max() / stds.min())
    return {
        "ok": ratio <= max_ratio,
        "ratio": ratio,
        "min_std": float(stds.min()),
        "max_std": float(stds.max()),
        "largest": stds.idxmax(),
        "smallest": stds.idxmin(),
    }

def assert_contiguous(
        timestamps: Sequence[object], calendar: pd.DatetimeIndex, name: str = "series"
) -> None:
    """ Asert a timestamp series has no gaps relative to the trading calendar"""
    ts = pd.DatetimeIndex(list(timestamps))
    positions = calendar.get_indexer(ts)
    if (positions < 0).any():
        raise AssertionError(f"{name}: timestamps not present on the trading calendar")
    if len(positions) > 1 and not np.all(np.diff(positions) == 1):
        raise AssertionError(f"{name}: timestamps are not contiguous on the calendar")
