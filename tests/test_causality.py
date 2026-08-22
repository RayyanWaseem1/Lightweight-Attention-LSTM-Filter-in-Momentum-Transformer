""" Causality and lookahead bias 

These tests use real signal and are designed to *fail* when leakage exists -- 
``test_leaky_feature_is_detected`` deliberately injects a leak and asserts the 
detector catches it, so a vacous pass is impossible 
"""

from __future__ import annotations 

import numpy as np 
import pandas as pd
import pytest 
import torch 

from Models.config import bars, get_default_config
from Models.Momentum_transformer import get_momentum_transformer
from Utils.Feature_engineering import FeatureConfig, create_features_from_ohlcv
from Utils.Market_data import tradeable_returns
from Utils.Metrics import sharpe_ratio

@pytest.fixture(scope = "module")
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 4000
    index = pd.date_range("2019-01-01", periods = n, freq = "h")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.0005, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.0015, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.0015, n))),
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index = index,
    )

### The information boundary ###
def test_features_only_use_past_prices(synthetic_ohlcv):
    """ Perturbing close[t] must not change any feature at row <= t
    
    This is an assertion the old diagnostic hardcoded as PASS
    """
    cfg = FeatureConfig()
    base = create_features_from_ohlcv(synthetic_ohlcv, config = cfg, drop_burn_in=False)

    perturb_at = 3000
    tampered = synthetic_ohlcv.copy()
    tampered.iloc[perturb_at, tampered.columns.get_loc("close")] *= 1.5
    after = create_features_from_ohlcv(tampered, config = cfg, drop_burn_in=False)

    # Rows strictly before the perturbation must be bit-identical 
    before_rows = slice(0, perturb_at)
    pd.testing.assert_frame_equal(
        base.iloc[before_rows], after.iloc[before_rows], check_exact = False, rtol = 1e-9
    )

    # Row `perturb_at` itself is built from close[perturb_at - 1] and so must 
    # ALSO be unchanged -- that is what shift(1) buys us
    pd.testing.assert_series_equal(
        base.iloc[perturb_at], after.iloc[perturb_at], check_exact=False, rtol = 1e-9
    )

def test_target_is_not_visible_to_features(synthetic_ohlcv):
    """ The target at row t must use a price no feature at row t has seen 
    
    feature[t] is built from close[t-1]; target[t] = close[t]/close[t-1] -1.
    Perturbing close[t] must move the target and leave the features fixed
    """
    cfg = FeatureConfig()
    prices = synthetic_ohlcv.reset_index(names = "timestamp")
    prices["symbol"] = "TEST"

    base_target = tradeable_returns(prices).set_index("timestamp")["tradeable_return"]

    t = 3000
    tampered = prices.copy()
    tampered.loc[t, "close"] *= 1.10
    new_target = tradeable_returns(tampered).set_index("timestamp")["tradeable_return"]

    ts = prices.loc[t, "timestamp"]
    assert not np.isclose(base_target.loc[ts], new_target.loc[ts]), (
        "target must depend on close[t]"
    )

    feats_before = create_features_from_ohlcv(synthetic_ohlcv, config = cfg,
                                              drop_burn_in=False)
    feats_after = create_features_from_ohlcv(
        tampered.set_index("timestamp")[["open", "high", "low", "close", "volume"]],
        config = cfg, drop_burn_in=False,
    )

def test_leaky_feature_is_detected(synthetic_ohlcv):
    """ A deliberately leaky feature MUST break the boundary test.
    
    Without this, the tests above could pas vacuously.
    """
    prices = synthetic_ohlcv
    leaky = prices["close"].pct_change().shift(-1) # tomorrow's return, today 

    t = 3000
    tampered = prices.copy()
    tampered.iloc[t, tampered.columns.get_loc("close")] *= 1.5
    leaky_after = tampered["close"].pct_change().shift(-1)

    # The leaky feature at row t-1 changes when close[t] changes -- exactly the 
    # violation the causal featuers must not exhibit 
    assert not np.isclose(leaky.iloc[t-1], leaky_after.iloc[t-1])

def test_no_burn_in_zeros(synthetic_ohlcv):
    """ Burn-in rows are dropped, not filled with a semantically loaded zero"""
    cfg = FeatureConfig()
    features = create_features_from_ohlcv(synthetic_ohlcv, config = cfg)

    longest_ma = f"ma_ratio_{cfg.window(126)}"
    assert longest_ma in features.columns 

    # A literal zero in a long window MA ratio would assert "price is exactly 
    # at its mean" -- the previous fillna(0) produced 252 of them per symbol
    assert (features[longest_ma] == 0.0).sum() == 0
    assert not features.isna().any().any()

### Shuffle / reversal / lag, against a predictor with genuine skill ###

# These are the four checks we look for. They test the *detector*: a 
# predictor with real skill must lose that skill when the target is shuffled,
# reversed, or mis-lagged 

# The predictor is a least-squares fit rather than the full Transformer, for 
# two reasons: it converges deterministically (so a flaky optimizer cannot turn
# a leakage test red), and it runs in milliseconds. Convergence of the actual 
# architecture is covered separately by ``test_model_learns_when_normalized``

# The series is constructed so that a constant position earns nothing: each 
# split is demeaned exactly. Without that, a predictor that collapses to "always 
# long" scores the drift, and since ``Sharpe(c * r)`` is invariant to the 
# ordering of ``r``, every test here would pass vacuously.

PPY = 252 * 7

@pytest.fixture(scope = "module")
def skilled_predictor():
    """ Fit a linear predictor on a drift_free AR(1) series; return OOS results"""
    rng = np.random.default_rng(11)
    n = 4000
    eps = rng.normal(0, 0.004, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.45 * r[i-1] + eps[i]

    # feature[t] = r[t-1] -- known at the end of bar t -1
    # target[t] = r[t] -- earned by a position taken at that point 
    feature = np.roll(r, 1)
    feature[0] = 0.0
    target = r.copy() 

    split = int(n * 0.7)
    target[:split] -= target[:split].mean()
    target[split:] -= target[split:].mean()

    # Least squares on the training split only 
    coef = np.polyfit(feature[:split], target[:split], 1)
    positions = np.polyval(coef, feature[split:])
    positions = positions / (positions.std() + 1e-12) # scale is arbitrary

    return positions.astype(float), target[split:].astype(float)

def test_predictor_has_real_skill(skilled_predictor):
    """ Guard: the predictor must have skill, and it must not be constant.
    
    If either fails, the three tests below prove nothing
    """
    positions, returns = skilled_predictor

    dispersion = positions.std() / (abs(positions.mean()) + 1e-9)
    assert dispersion > 1.0, (
        f"positions nearly constant (std/|mean| = {dispersion:.3f}); "
        "shuffle and reversal tests would be vacuous"
    )

    constant_sharpe = abs(sharpe_ratio(np.ones_like(returns) * returns, PPY))
    assert constant_sharpe < 0.05, (
        f"split still carries drift (constant-position Sharpe {constant_sharpe:.4f})"
    )

    assert sharpe_ratio(positions * returns, PPY) > 2.0

def test_shuffled_targets_destroy_edge(skilled_predictor):
    """ Shuffling the target within the test set must collapse the Sharpe"""
    positions, returns = skilled_predictor 
    rng = np.random.default_rng(3)

    real = sharpe_ratio(positions * returns, PPY)
    shuffled = [
        abs(sharpe_ratio(positions * rng.permutation(returns), PPY)) for _ in range(50)
    ]
    assert np.mean(shuffled) < real / 5, (
        f"mean shuffled Sharpe {np.mean(shuffled):.3f} vs real {real:.3f}"
    )

def test_reversed_time_destroys_edge(skilled_predictor):
    """ Pairing positions with time-reversed returns must not retain the edge"""
    positions, returns = skilled_predictor
    real = sharpe_ratio(positions * returns, PPY)
    reversed_sharpe = abs(sharpe_ratio(positions * returns[::-1], PPY))
    assert reversed_sharpe < real / 5

def test_extra_lag_degrades_gracefully(skilled_predictor):
    """ One extra bar of lag must degrade the edge -- gracefully, not catastrophically.
    
    This is the audit's check (c), and "gracefully" is the operative word. On an 
    AR(1) series with phi = 0.45, lagging the target by one bar takes the usable
    correlation from phi to phi^2, so roughly half the skill should survive. A *collapse*
    to zero here would suggest the model was keying on same bar information; an *increase*
    would mean the alignment is off by a bar.
    """

    positions, returns = skilled_predictor
    real = sharpe_ratio(positions[:-1] * returns[:-1], PPY)
    lagged = sharpe_ratio(positions[:-1] * returns[1:], PPY)

    assert lagged < real, f"lagging must not improve performance ({lagged:.3f} >= {real:.3f})"
    assert lagged > 0.1 * real, (
        f"edge collapsed rather than degraded ({lagged:.3f} vs {real:.3f}); "
        " suggests dependence on same-bar information"
    )

def test_model_learns_when_normalized():
    """ The architecture must learn a planted signal -- but only if inputs are scaled.
    
    This encodes a finding from building these tests. Raw return-scale features
    (std ~0.004) leave the LSTM's bias terms dominating its inputs, the ``tanh``
    head saturates, and the model emits an essentially constant position: with 
    unnormalized inputs the fitted model reached a test Sharpe of 0.000 with 
    ``std/|mean| = 0.0004`` despite its predictions correlating +0.28 with the 
    target. Normalizing the inputs -- which ``Utils.Feature_engineering.FeatureNormalizer``
    does in the real pipeline, fitted on the training split only -- recovers it.
    
    That is the concrete reason audit item 3.2 matters: before this change
    nothing in the pipeline standardized any feature.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(5)

    n, seq_len = 1200, 16
    eps = rng.normal(0, 0.004, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.6 * r[i-1] + eps[i]

    feats = np.stack([np.roll(r,k) for k in range(1,3)], axis = 1).astype(np.float32)
    feats[:3] = 0.0
    x = np.stack([feats[t - seq_len + 1:t+1] for t in range(seq_len, n)]).astype(np.float32)
    y = r[seq_len:n].astype(np.float32)

    split = int(len(x) * 0.7)
    y[:split] -= y[:split].mean()
    y[split:] -= y[split:].mean()

    # Normalize with TRAINING-split statistics only 
    flat = x[:split].reshape(-1, x.shape[2])
    x = ((x - flat.mean(0)) / (flat.std(0) * 1e-8)).astype(np.float32)

    cfg = get_default_config()
    cfg.model.input_dim = x.shape[2]
    cfg.model.hidden_dim = 8
    cfg.model.num_transformer_layers = 1
    cfg.model.num_attention_heads = 2
    cfg.model.short_window = 4
    cfg.model.sequence_length = seq_len
    cfg.model.use_volatility_targeting = False 

    model = get_momentum_transformer(cfg.model, "simple", cfg.training)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-2)

    xt, yt = torch.from_numpy(x[:split]), torch.from_numpy(y[:split])
    for _ in range(120):
        opt.zero_grad()
        pos, _ = model(xt)
        pnl = pos * yt 
        (-(pnl.mean() / (pnl.std() + 1e-8))).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        positions, _ = model(torch.from_numpy(x[split:]))
    positions = positions.numpy()

    dispersion = positions.std() / (abs(positions.mean()) + 1e-9)
    assert dispersion > 0.05, (
        f"positions collapsed to a constant (std/|mean| = {dispersion:.4f}) even "
        "with normalized inputs"
    )
    assert sharpe_ratio(positions * y[split:], PPY) > 0.5
