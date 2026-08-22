"""Feature engineering and data-preparation tests (Tier 3 / Tier 0.4-0.5).

Includes the stock-agnostic comparability check that used to live in
``Utils/Feature_engineering.__main__`` -- the audit called that a good test to
have written, so it is kept and moved somewhere that actually runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Models.config import BARS_PER_DAY
from Utils.Feature_engineering import (
    FeatureConfig,
    FeatureNormalizer,
    check_collinearity,
    check_feature_scales,
    create_features_from_ohlcv,
)
from Utils.Market_data import (
    back_adjust_splits,
    build_trading_calendar,
    causal_bad_tick_mask,
    detect_splits,
    infer_regular_hours,
)


def make_series(n=2500, price=100.0, vol=0.004, seed=0, stochastic_vol=True):
    """Synthetic OHLCV with time-varying volatility.

    The stochastic volatility matters: under *constant* volatility a
    risk-adjusted momentum feature is just the raw return divided by a
    constant, so it correlates ~0.997 with ``return_*`` and the
    non-duplication test fails for reasons that have nothing to do with the
    feature definition. Real return series are heteroskedastic, and that is
    what makes the risk adjustment carry information.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=n, freq="h")

    if stochastic_vol:
        # Slow-moving log-AR(1) volatility process spanning roughly 0.4x-2.5x.
        log_vol = np.zeros(n)
        for i in range(1, n):
            log_vol[i] = 0.995 * log_vol[i - 1] + rng.normal(0, 0.06)
        sigma = vol * np.exp(log_vol - log_vol.mean())
    else:
        sigma = np.full(n, vol)

    close = price * np.exp(np.cumsum(rng.normal(0, 1, n) * sigma))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.0004, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.0012, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.0012, n))),
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# 3.1 -- no duplicate features
# ---------------------------------------------------------------------------
def test_momentum_is_not_a_duplicate_of_return():
    """``return_5 == momentum_5`` etc. were verified identical before."""
    features = create_features_from_ohlcv(make_series(), config=FeatureConfig())
    cfg = FeatureConfig()

    for days in (5, 21, 63, 126):
        w = cfg.window(days)
        ret_col, mom_col = f"return_{w}", f"momentum_{w}"
        if ret_col in features and mom_col in features:
            assert not np.allclose(features[ret_col], features[mom_col]), (
                f"{ret_col} and {mom_col} are identical again"
            )
            rho = features[ret_col].corr(features[mom_col])
            assert abs(rho) < 0.99, f"{ret_col} ~ {mom_col} rho={rho:.4f}"


def test_no_feature_pair_is_perfectly_collinear():
    features = create_features_from_ohlcv(make_series(), config=FeatureConfig())
    offenders = check_collinearity(features, threshold=0.99)
    assert not offenders, f"near-duplicate features: {offenders[:5]}"


def test_rsi_normalized_is_gone():
    """It was an exact affine map of ``rsi``."""
    features = create_features_from_ohlcv(make_series(), config=FeatureConfig())
    assert "rsi" in features.columns
    assert "rsi_normalized" not in features.columns


# ---------------------------------------------------------------------------
# 3.2 -- normalisation, fitted on the training split only
# ---------------------------------------------------------------------------
def test_normalizer_fits_on_train_and_applies_unchanged_to_test():
    features = create_features_from_ohlcv(make_series(n=3000), config=FeatureConfig())
    split = len(features) // 2
    train, test = features.iloc[:split], features.iloc[split:]

    normalizer = FeatureNormalizer().fit(train)
    train_n = normalizer.transform(train)
    test_n = normalizer.transform(test)

    # Training statistics are standard by construction -- up to the +/-5 sigma
    # clip, which trims one tail more than the other and so moves the mean a
    # little off zero. That is the clip doing its job, not a fitting error.
    assert train_n["return_1"].mean() == pytest.approx(0.0, abs=5e-3)
    assert train_n["return_1"].std() == pytest.approx(1.0, abs=5e-2)

    # Test statistics are NOT forced to zero/one -- that would be leakage.
    assert test_n["return_1"].std() != pytest.approx(1.0, abs=1e-6)

    # Round-tripping the persisted statistics reproduces the transform exactly.
    restored = FeatureNormalizer.from_dict(normalizer.to_dict())
    pd.testing.assert_frame_equal(restored.transform(test), test_n)


def test_normalisation_brings_rsi_onto_the_same_scale():
    """Raw RSI ranges 0-100 next to returns with std ~0.005 -- a 10,000x
    disparity into an LSTM with no input normalisation."""
    features = create_features_from_ohlcv(make_series(), config=FeatureConfig())

    raw = check_feature_scales(features)
    assert raw["ratio"] > 100, "expected a large raw scale disparity"

    normalised = FeatureNormalizer().fit_transform(features)
    assert check_feature_scales(normalised)["ratio"] < 50


def test_clipping_happens_after_standardising():
    """The old comment said '[-10, 10] std devs' over code clipping raw values."""
    features = create_features_from_ohlcv(make_series(), config=FeatureConfig())
    normalised = FeatureNormalizer(clip=5.0).fit_transform(features)

    unbounded = [c for c in normalised.columns
                 if c not in {"bb_position", "close_position",
                              "price_percentile", "return_percentile"}]
    assert normalised[unbounded].abs().max().max() <= 5.0 + 1e-9


# ---------------------------------------------------------------------------
# 3.3 -- burn-in
# ---------------------------------------------------------------------------
def test_burn_in_rows_are_dropped_not_filled():
    cfg = FeatureConfig()
    raw = make_series(n=2500)
    features = create_features_from_ohlcv(raw, config=cfg)

    assert len(features) < len(raw)
    assert len(features) <= len(raw) - cfg.max_lookback
    assert not features.isna().any().any()


# ---------------------------------------------------------------------------
# 3.4 -- percentile direction
# ---------------------------------------------------------------------------
def test_price_percentile_is_high_at_window_highs():
    """The old formula counted values ABOVE the current one, so a reading near
    1.0 meant the price was near its window MINIMUM."""
    cfg = FeatureConfig()
    n = 1600
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    # Strictly increasing price: every bar is its window's maximum.
    close = np.linspace(100, 300, n)
    rising = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": np.full(n, 1e6)},
        index=index,
    )

    features = create_features_from_ohlcv(rising, config=cfg)
    assert features["price_percentile"].tail(100).mean() > 0.95

    falling = rising.copy()
    falling[["open", "high", "low", "close"]] = falling[
        ["open", "high", "low", "close"]
    ].to_numpy()[::-1]
    features_down = create_features_from_ohlcv(falling, config=cfg)
    assert features_down["price_percentile"].tail(100).mean() < 0.05


# ---------------------------------------------------------------------------
# 3.5 -- horizons expressed in bars
# ---------------------------------------------------------------------------
def test_long_horizon_features_exist():
    """A momentum model needs a horizon longer than ~36 trading days."""
    cfg = FeatureConfig()
    features = create_features_from_ohlcv(make_series(n=3000), config=cfg)

    longest = max(
        int(c.rsplit("_", 1)[1])
        for c in features.columns
        if c.startswith(("return_", "ma_ratio_", "momentum_")) and c.rsplit("_", 1)[1].isdigit()
    )
    assert longest >= 126 * BARS_PER_DAY * 0.9, (
        f"longest feature horizon is {longest} bars "
        f"({longest / BARS_PER_DAY:.0f} trading days)"
    )


def test_windows_are_multiples_of_bars_per_day():
    cfg = FeatureConfig()
    assert cfg.window(1) == BARS_PER_DAY
    assert cfg.window(21) == 21 * BARS_PER_DAY
    assert cfg.window(126) == 126 * BARS_PER_DAY


# ---------------------------------------------------------------------------
# The stock-agnostic property (kept from the old __main__ demo)
# ---------------------------------------------------------------------------
def test_features_are_comparable_across_wildly_different_prices():
    """A $300k share and a $2 share must produce comparable feature ranges."""
    cfg = FeatureConfig()
    # Same seed => identical return path, only the price LEVEL differs. That is
    # exactly the property "stock-agnostic" claims.
    berkshire = create_features_from_ohlcv(
        make_series(n=2500, price=300_000, vol=0.002, seed=1), config=cfg
    )
    penny = create_features_from_ohlcv(
        make_series(n=2500, price=2.0, vol=0.002, seed=1), config=cfg
    )

    for column in ("return_1", "ma_ratio_147", "volatility_21", "rsi", "volume_ratio"):
        if column not in berkshire or column not in penny:
            continue
        a, b = berkshire[column].std(), penny[column].std()
        assert a == pytest.approx(b, rel=0.5), (
            f"{column} not stock-agnostic: std {a:.5f} vs {b:.5f}"
        )


# ---------------------------------------------------------------------------
# 0.4 -- split adjustment
# ---------------------------------------------------------------------------
def test_splits_are_back_adjusted_not_deleted():
    n = 500
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    close = np.full(n, 100.0)
    close[250:] = 50.0  # a clean 2:1 forward split

    frame = pd.DataFrame(
        {"timestamp": index, "symbol": "AAA", "open": close, "high": close,
         "low": close, "close": close, "volume": np.full(n, 1e6)}
    )

    splits = detect_splits(frame.set_index("timestamp")["close"])
    assert len(splits) == 1
    assert splits.iloc[0]["type"] == "forward_split"
    assert splits.iloc[0]["factor"] == pytest.approx(2.0)

    adjusted, _ = back_adjust_splits(frame)

    # Every bar is retained -- the old code deleted the split bar.
    assert len(adjusted) == n
    # And the return series is continuous across the split.
    returns = adjusted["close"].pct_change()
    assert returns.abs().max() < 1e-9
    # Pre-split prices are halved; post-split are untouched.
    assert adjusted["close"].iloc[0] == pytest.approx(50.0)
    assert adjusted["close"].iloc[-1] == pytest.approx(50.0)
    # Volume moves inversely.
    assert adjusted["volume"].iloc[0] == pytest.approx(2e6)


def test_split_adjustment_protects_long_window_features():
    """One unadjusted split corrupts every window spanning it."""
    n = 2000
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    split_at = 1200
    unadjusted = close.copy()
    unadjusted[split_at:] /= 2.0

    frame = pd.DataFrame(
        {"timestamp": index, "symbol": "AAA", "open": unadjusted,
         "high": unadjusted, "low": unadjusted, "close": unadjusted,
         "volume": np.full(n, 1e6)}
    )
    adjusted, _ = back_adjust_splits(frame)

    # After adjustment the series matches the true (pre-split-scaled) path.
    expected = close / 2.0
    assert np.allclose(adjusted["close"].to_numpy(), expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# 0.5 -- trading calendar
# ---------------------------------------------------------------------------
def test_regular_hours_are_inferred_from_bar_density():
    rows = []
    for day in pd.date_range("2022-01-03", periods=30, freq="B"):
        for hour in range(13, 20):          # dense regular hours
            rows.append(day + pd.Timedelta(hours=hour))
        if day.day % 10 == 0:                # sparse extended hours
            rows.append(day + pd.Timedelta(hours=23))

    frame = pd.DataFrame({"timestamp": rows})
    hours = infer_regular_hours(frame)

    assert hours == list(range(13, 20))
    assert 23 not in hours

    calendar = build_trading_calendar(frame)
    assert len(calendar) == 30 * 7
    assert set(pd.DatetimeIndex(calendar).hour) == set(range(13, 20))


# ---------------------------------------------------------------------------
# 0.3 -- causal bad-tick filter
# ---------------------------------------------------------------------------
def test_bad_tick_filter_uses_only_trailing_information():
    rng = np.random.default_rng(3)
    n = 1200
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))))
    close.iloc[900] *= 3.0  # a bad print

    flagged = causal_bad_tick_mask(close, k=12.0, window=147)
    assert flagged.iloc[900]

    # Changing a LATER bar must not change whether an EARLIER bar is flagged --
    # the property full-sample quantile winsorisation violates.
    tampered = close.copy()
    tampered.iloc[1100] *= 5.0
    reflagged = causal_bad_tick_mask(tampered, k=12.0, window=147)
    pd.testing.assert_series_equal(flagged.iloc[:1000], reflagged.iloc[:1000])
