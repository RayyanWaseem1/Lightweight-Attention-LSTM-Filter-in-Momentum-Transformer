""" Data loading, split adjustment, trading calendar and causal cleaning.

**Splits are back-adjusted** not deleted. 
**An explicit regular-hours trading calendar** replaces the union of all 
symbols' timestamps. The union grid contained sparse extended-hours bars at which only
one or two holdings printed
**Bad ticks are filtered causally** (``|r| > k * training_vol``) using a 
backward-looking, shifted volatility estimate) at the *data* layer. Realized
PnL is never winsorized.
**The benchmark is loaded outside the tradeable universe.** Previously SPY could 
be dropped by any of the five universe filters, which silently removed 
eight feature columns and changed ``input_dim`` from 53 to 45 with no error
"""

from __future__ import annotations

from pathlib import Path 
from typing import Dict, List, Optional, Sequence, Tuple 

import numpy as np 
import pandas as pd

from Models.config import BARS_PER_DAY, DataConfig

# Leveraged / inverse ETFs: their returns are a mechanical multiple of an 
# underlying index and they are not a sensible momentum universe

LEVERAGED_ETFS = {
    "SOXL", "SOXS", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA",
    "LABU", "LABD", "JNUG", "JDST", "NUGT", "DUST", "ERX", "ERY",
    "FAS", "FAZ", "TECL", "TECS", "CURE", "CUT", "WANT", "GASL",
    "BULL", "BEAR", "TSLL", "TSLS", "NVDL", "NVDS", "MSTU", "MSTX",
    "SSO", "SDS", "QLD", "QID", "DDM", "DXD", "UWM", "TWM",
    "SPUU", "SPDD", "QQQU", "SQQD",
}

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

### Raw Load ###

def load_raw_ohlcv(csv_path, symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """ Load the OHLCV CSV into a tidy long frame.
    
    Returns columns ``timestamp, symbol, open, high, low, close, volume`` sorted
    by (symbol, timestamp), tz-naive UTC
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "ts_event" in df.columns:
        ts_col = "ts_event"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        raise ValueError("CSV must have a 'ts_event' or 'timestamp' column")

    required = ["symbol"] + OHLCV_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    if symbols is not None:
        df = df[df["symbol"].isin(set(symbols))]

    df["timestamp"] = pd.to_datetime(df[ts_col], utc = True, format = "mixed").dt.tz_convert(None)
    df = df[["timestamp", "symbol"] + OHLCV_COLUMNS].copy()
    df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].apply(pd.to_numeric, errors = "coerce")

    df = df.dropna(subset = OHLCV_COLUMNS)
    df = df.drop_duplicates(subset = ["symbol", "timestamp"], keep = "last")
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop = True)

### Split Adjustment ###

def detect_splits(
    close: pd.Series, low_ratio: float = 0.5, high_ratio: float = 2.0
) -> pd.DataFrame:
    """ Detect probable split bars from extreme single-bar price ratios.
    
    An hourly bar that moves more than 2x in either direction is far more
    likely to be a corporate action or a bad print than a real return
    """

    ratio = close / close.shift(1)
    # Inclusive: a clean 2:1 split gives exactly 0.5, which a strict '<' misses
    is_split = (ratio <= low_ratio) | (ratio >= high_ratio)
    idx = np.flatnonzero(is_split.to_numpy())

    if idx.size == 0:
        return pd.DataFrame(columns = ["position", "timestamp", "ratio", "factor", "type"])

    records = []
    for pos in idx:
        row_pos = int(pos)
        r = float(ratio.iloc[row_pos])
        records.append(
            {
                "position": row_pos,
                "timestamp": close.index[row_pos],
                "ratio": r,
                # Factor by which pre-split prices must be divided
                "factor": _snap_split_factor(1.0 / r),
                "raw_factor": 1.0 / r,
                "type": "forward_split" if r < 1 else "reverse_split",
            }
        )
    return pd.DataFrame(records)

def _snap_split_factor(factor: float, tolerance: float = 0.10) -> float:
    """ Round a measured price ratio to the nearest plausible split ratio.
    
    The measured ratio is contaminated by the bar's own return: a true 2:1
    split on a bar that also moved 0.3% reads as 1.994 not 2.000. Corporate 
    actions are simple rationals, so snapping ot the nearest fraction with 
    a small denominator recovers the intended factor and keeps the adjusted 
    series exact. Ratios that are not near any simple rational are left alone
    """

    from fractions import Fraction 

    if not np.isfinite(factor) or factor <= 0:
        return 1.0 

    candidate = Fraction(factor).limit_denominator(10)
    snapped = float(candidate)
    if snapped > 0 and abs(snapped - factor) / factor <= tolerance:
        return snapped
    return factor 

def back_adjust_splits(
    symbol_df: pd.DataFrame,
    low_ratio: float = 0.5,
    high_ratio: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Back adjusted OHLC prices (and volume) for detected splits.
    
    ``adjusted[t] = raw[t] / prod(factor_i for every split i occurring after t)``
    
    This keeps the return series continous across the split instead of punching a 
    hole in it, so window-spanning features stay valid
    """

    df = symbol_df.sort_values("timestamp").reset_index(drop = True).copy()
    close = df.set_index("timestamp")["close"]
    splits = detect_splits(close, low_ratio, high_ratio)

    if splits.empty:
        return df, splits 

    factors = np.ones(len(df), dtype = float)
    for _, row in splits.iterrows():
        factors[int(row["position"])] = float(row["factor"])

    # cum_factor[t] = product of factors strictly after t
    reversed_factors = factors[::-1]
    shifted = np.concatenate([[1.0], reversed_factors[:-1]])
    cum_factor = np.cumprod(shifted)[::-1]

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].to_numpy(dtype = float) / cum_factor
    # Share count moves inversely to price 
    df["volume"] = df["volume"].to_numpy(dtype = float) * cum_factor

    return df, splits

### Trading Calendar ###

def infer_regular_hours(df: pd.DataFrame, coverage: float = 0.5) -> List[int]:
    """ Infer the regular-trading-hours UTC hours from bar density.
    
    Extended-hours bars are sparse: for this feed the dense hours are
    13:00 - 19:00 UTC with ~1,800 bars per symbol, while 11:00 has ~50 and 22:00
    has ~15. Any hour carrying at least ``coverage`` of the busiest hour's bar
    count is treated as regular hours
    """

    counts = df.groupby(df["timestamp"].dt.hour).size()
    if counts.empty:
        return []
    threshold = int(np.ceil(float(counts.max()) * coverage))
    regular = counts[counts >= threshold]
    hours = sorted(int(h) for h in regular.index.to_numpy(dtype = int))
    return hours 

def build_trading_calendar(
        df: pd.DataFrame, regular_hours: Optional[Sequence[int]] = None
) -> pd.DatetimeIndex:
    """ Build the epxlicit bar grid the strategy trades on
    
    Only regular-hours timestamps are kept. Every symbol is later reindexed
    onto this grid, so ``n_periods`` means the same thing for every series
    """

    if regular_hours is None:
        regular_hours = infer_regular_hours(df)

    mask = df["timestamp"].dt.hour.isin(list(regular_hours))
    calendar = pd.DatetimeIndex(sorted(df.loc[mask, "timestamp"].unique()))
    return calendar 

def reindex_to_calendar(
    symbol_df: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """ Restrict a symbol to the calendar grid over its own trading lifetime.
    
    Bars outsdie the calendar are dropped. The grid is *not* extended beyond
    the symbol's first/last observation, so a late-listing name does not gain
    fabricated history
    """

    df = symbol_df.sort_values("timestamp").set_index("timestamp")
    if df.empty:
        return symbol_df.iloc[0:0]

    window = calendar[(calendar >= df.index.min()) & (calendar <= df.index.max())]
    out = df.reindex(window)
    out.index.name = "timestamp"
    return out.reset_index()

### Causal cleaning ###

def causal_bad_tick_mask(
    close: pd.Series, k: float = 12.0, window: int = 147
) -> pd.Series:
    """ Flag bars whose returns exceeds ``k`` times a *trailing* volatility.
    
    The volatility estimate is shifted by one bar so it uses only information
    available strictly before the bar being judged. This is the documented, 
    causal replacement for winsorizing realized returns with full-sample
    quantiles.
    """

    returns = close.pct_change()
    trailing_vol = returns.rolling(window, min_periods = window // 4).std().shift(1)
    threshold = k * trailing_vol
    flagged = returns.abs() > threshold
    return flagged.fillna(False)

### Universe construction ###

def filter_universe(
    df: pd.DataFrame,
    cfg: DataConfig,
    max_symbols: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """ Apply the tradable-unvierse filters.
    
    The benchmark symbol is deliberately **not** protected here -- it is loaded
    separately by :func: `load_market_series` so that a filter dropping it from 
    the tradable universe cannot silently remove the market feature columns
    """

    def log(msg):
        if verbose:
            print(msg)

    symbols = sorted(df["symbol"].unique())
    log(f" Initial: {len(df):,} bars, {len(symbols)} symbols")

    # 1. Penny stocks
    min_price = df.groupby("symbol")["close"].min()
    symbols = [s for s in symbols if min_price.get(s, 0) >= cfg.min_price]
    log(f" After price >= ${cfg.min_price}: {len(symbols)} symbols")

    # 2. Leveraged / inverse ETFs
    dropped_lev = [s for s in symbols if s in LEVERAGED_ETFS]
    symbols = [s for s in symbols if s not in LEVERAGED_ETFS]
    if dropped_lev:
        log(f" Removed {len(dropped_lev)} leveraged ETFs: {', '.join(dropped_lev)}")

    df = df[df["symbol"].isin(symbols)]

    # 3. Symbols with implausible returns even after split adjustment
    returns = df.groupby("symbol")["close"].pct_change()
    max_abs = returns.abs().groupby(df["symbol"]).max()
    extreme = [s for s in symbols if max_abs.get(s, 0) > cfg.max_abs_return]
    symbols = [s for s in symbols if s not in extreme]
    if extreme:
        log(f" Removed {len(extreme)} symbols with |return| > {cfg.max_abs_return:.0%}")

    # 4. Sufficient history
    counts = df[df["symbol"].isin(symbols)].groupby("symbol").size()
    symbols = [s for s in symbols if counts.get(s, 0) >= cfg.min_bars]
    log(f" After >= {cfg.min_bars:,} bars: {len(symbols)} symbols")

    if max_symbols is not None and len(symbols) > max_symbols:
        symbols = symbols[:max_symbols]
        log(f" Limited to {max_symbols} symbols")

    df = df[df["symbol"].isin(symbols)].reset_index(drop = True)
    return df, symbols 

def load_market_series(
    raw_df: pd.DataFrame, calendar: pd.DatetimeIndex, market_symbol: str = "SPY"
) -> pd.DataFrame:
    """ Load the benchmark OHLCV, independent of the tradable-universe filters.
    
    Raises if the benchmark is absent rather than silently dropping the eight
    market/relative-strength feature columns and changin ``input_dim``
    """

    market = raw_df[raw_df["symbol"] == market_symbol]
    if market.empty:
        raise ValueError(
            f"Benchmark symbol {market_symbol!r} not present in the data. "
            "Market-context and relative-strength features cannot be built; "
            "refusing to continue with a silently different feature set."
        )

    market, _ = back_adjust_splits(market)
    market = reindex_to_calendar(market, calendar)
    market = market.set_index("timestamp")[OHLCV_COLUMNS]
    return market.ffill() 

def prepare_market_data(
    csv_path,
    cfg: DataConfig,
    max_symbols: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """ End to end data preparation.
    
    Order matters: splits are adjusted **before** the unvierse filters run, so 
    a real split is not mistaken for an implausible return.
    
    Returns a dict with ``prices`` (long frmae on the calendar), ``symbols``,
    ``market`` (benchmark OHLCV indexed by timestamp), ``calendar`` and 
    ``diagnostics``
    """

    def log(msg):
        if verbose:
            print(msg)

    log("\n[Data] Loading OHLCV ...")
    raw = load_raw_ohlcv(csv_path)

    if start is not None:
        raw = raw[raw["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        raw = raw[raw["timestamp"] <= pd.Timestamp(end)]
    if cfg.data_end is not None:
        raw = raw[raw["timestamp"] <= pd.Timestamp(cfg.data_end)]

    log(f" Loaded {len(raw):,} bars, {raw['symbol'].nunique()} symbols")

    # Trading calendar first: everything is expressed on this grid
    regular_hours = infer_regular_hours(raw)
    calendar = build_trading_calendar(raw, regular_hours)
    log(f" Regular hours (UTC): {regular_hours} -> {len(calendar):,} calendar bars")

    n_days = len(set(calendar.date))
    log(f" Calendar spans {n_days:,} sessions " f"({len(calendar) / max(n_days, 1):.2f} bars/session)")

    # Split-adjust every symbol before any return-based filtering
    log("\n [Data] Back-adjusted splits...")
    adjusted, split_report = [], {}
    for symbol, group in raw.groupby("symbol", sort = False):
        adj, splits = back_adjust_splits(group)
        if not splits.empty:
            split_report[symbol] = len(splits)
        adjusted.append(adj)
    raw_adj = pd.concat(adjusted, ignore_index = True)
    log(f" Adjusted {len(split_report)} symbols with detected splits " f"({sum(split_report.values())} split events)")

    # Benchmark, outside the universe filters
    market = load_market_series(raw_adj, calendar, cfg.market_symbol)
    log(f" Benchmark {cfg.market_symbol}: {len(market):,} bars on calendar")

    # Tradable universe
    log("\n [Data] filtering tradable universe...")
    filtered, symbols = filter_universe(raw_adj, cfg, max_symbols, verbose)

    # Reindex onto the calendar and apply the causal bad-tick filter
    log("\n [Data] Reindexing to calendar and removing bad ticks ...")
    cleaned, n_bad = [], 0
    for symbol, group in filtered.groupby("symbol", sort = False):
        g = reindex_to_calendar(group, calendar)
        g["symbol"] = symbol
        g[OHLCV_COLUMNS] = g[OHLCV_COLUMNS].ffill()
        g = g.dropna(subset = ["close"])
        if g.empty:
            continue 

        bad = causal_bad_tick_mask(
            g.set_index("timestamp")["close"], cfg.bad_tick_k, cfg.bad_tick_vol_window
        )
        n_bad += int(bad.sum())
        g = g.loc[~bad.to_numpy()]
        cleaned.append(g)

    prices = pd.concat(cleaned, ignore_index = True)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop = True)
    log(f" Removed {n_bad:,} bad-tick bars " f"(|r| > {cfg.bad_tick_k} x trailing vol)")
    log(f" Final: {len(prices):,} bars, {prices['symbol'].nunique()} symbols")

    return {
        "prices": prices,
        "symbols": sorted(prices["symbol"].unique()),
        "market": market,
        "calendar": calendar,
        "diagnostics": {
            "regular_hours": regular_hours,
            "n_calendar_bars": int(len(calendar)),
            "n_sessions": int(n_days),
            "bars_per_session": float(len(calendar) / max(n_days, 1)),
            "symbols_with_splits": split_report,
            "n_bad_ticks_removed": int(n_bad),
        },
    }

def tradeable_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """The return earned by acting on the information in feature row ``t``
    
    ``tradeable_return[t] = close[t] / close[t-1] - 1``
    """

    df = prices.sort_values(["symbol", "timestamp"])

    out = df[["timestamp", "symbol"]].copy()
    out["tradeable_return"] = df.groupby("symbol")["close"].pct_change()
    return out.dropna(subset=["tradeable_return"]).reset_index(drop=True)
