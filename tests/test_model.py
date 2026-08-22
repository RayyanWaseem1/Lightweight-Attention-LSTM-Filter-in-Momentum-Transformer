"""Model architecture tests (Tier 2)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from Models.config import get_default_config
from Models.Ensemble_model import (
    REGIME_FEATURE_NAMES,
    DynamicWeightNetwork,
    RegimeFeatureExtractor,
    get_ensemble_model,
)
from Models.LSTM import LSTMMomentumEncoder
from Models.Momentum_transformer import (
    apply_volatility_target,
    count_parameters,
    get_momentum_transformer,
)
from Models.Transformer_layers import (
    InterpretableMultiheadAttention,
    PositionalEncoding,
    TransformerEncoderBlock,
    TransformerEncoder,
    build_causal_mask,
)

REGIME_COLUMNS = [
    "return_1", "rsi", "volatility_21", "momentum_35",
    "market_return_1", "market_volatility", "beta",
]


def make_config(n_features: int):
    cfg = get_default_config()
    cfg.model.input_dim = n_features
    cfg.model.hidden_dim = 16
    cfg.model.num_transformer_layers = 1
    cfg.model.num_attention_heads = 2
    cfg.model.short_window = 8
    cfg.model.sequence_length = 32
    return cfg


# ---------------------------------------------------------------------------
# 2.1 -- the Transformer must see more than the last short_window bars
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode,expected_len", [("strided", 4), ("full", 32), ("truncate", 8)]
)
def test_encoder_modes_produce_expected_sequence_length(mode, expected_len):
    encoder = LSTMMomentumEncoder(
        input_dim=5, hidden_dim=16, num_layers=1, short_window=8, mode=mode
    )
    x = torch.randn(3, 32, 5)
    sequence, _ = encoder(x, return_sequences=True)
    assert sequence.shape == (3, expected_len, 16)


def test_strided_mode_uses_the_whole_window():
    """The premise of the architecture: bars outside the last short_window matter.

    Under the old truncating encoder, changing anything before the final 63 bars
    left the output bit-identical -- 75% of every loaded window was discarded.
    """
    torch.manual_seed(0)
    cfg = make_config(5)
    x = torch.randn(2, 32, 5)

    strided = get_momentum_transformer(cfg.model, "simple", cfg.training)
    strided.eval()

    tampered = x.clone()
    tampered[:, :8, :] += 5.0  # perturb only the OLDEST block

    with torch.no_grad():
        base, _ = strided(x)
        after, _ = strided(tampered)
    assert not torch.allclose(base, after), (
        "strided encoder ignored the oldest block -- it is still truncating"
    )

    cfg.model.lstm_transformer_mode = "truncate"
    truncating = get_momentum_transformer(cfg.model, "simple", cfg.training)
    truncating.eval()
    with torch.no_grad():
        t_base, _ = truncating(x)
        t_after, _ = truncating(tampered)
    # Documents the old behaviour explicitly.
    assert torch.allclose(t_base, t_after)


# ---------------------------------------------------------------------------
# 2.2 -- positions must be bounded, and vol targeting must be wired up
# ---------------------------------------------------------------------------
def test_positions_are_bounded_by_tanh():
    cfg = make_config(5)
    cfg.model.use_volatility_targeting = False
    model = get_momentum_transformer(cfg.model, "enhanced", cfg.training)

    x = torch.randn(16, 32, 5) * 50.0  # deliberately extreme inputs
    positions, _ = model(x)

    assert positions.abs().max() <= 1.0 + 1e-6, "output is not a bounded position"


def test_volatility_targeting_scales_and_caps():
    raw = torch.tensor([1.0, 1.0, 1.0])
    vol = torch.tensor([0.30, 0.15, 0.0001])  # high, at target, ~zero

    scaled = apply_volatility_target(raw, vol, target_volatility=0.15, max_leverage=3.0)

    assert scaled[0] == pytest.approx(0.5, abs=1e-3)   # half size in double vol
    assert scaled[1] == pytest.approx(1.0, abs=1e-3)   # full size at target
    assert scaled[2] == pytest.approx(3.0, abs=1e-6)   # capped, not unbounded


def test_volatility_targeting_is_a_noop_without_vol():
    raw = torch.tensor([0.5, -0.5])
    assert torch.allclose(apply_volatility_target(raw, None, 0.15), raw)


# ---------------------------------------------------------------------------
# 2.13 / 2.14 -- causal masking
# ---------------------------------------------------------------------------
def test_causal_mask_convention_matches_pytorch():
    """True == disallowed, strictly upper triangular."""
    mask = build_causal_mask(4)
    assert mask.dtype == torch.bool
    assert not mask[0, 0] and not mask[3, 0]   # may attend to self and past
    assert mask[0, 1] and mask[0, 3]           # may not attend to the future
    assert mask.sum() == 6                     # n*(n-1)/2


def test_future_positions_cannot_affect_earlier_outputs():
    """The unit test audit item 2.14 asks for, on the interpretable attention."""
    torch.manual_seed(0)
    encoder = TransformerEncoder(
        d_model=8, num_layers=1, num_heads=2, dim_feedforward=16,
        dropout=0.0, use_positional_encoding=False, use_causal_mask=True,
    )
    encoder.eval()

    x = torch.randn(2, 6, 8)
    tampered = x.clone()
    tampered[:, 4:, :] += 10.0  # perturb only the future

    with torch.no_grad():
        base, _ = encoder(x)
        after, _ = encoder(tampered)

    # Positions before the perturbation must be untouched.
    assert torch.allclose(base[:, :4, :], after[:, :4, :], atol=1e-5)
    # And the perturbation must actually have done something later on.
    assert not torch.allclose(base[:, 4:, :], after[:, 4:, :])


def test_interpretable_attention_shares_one_value_head():
    """The TFT construction, which the old class claimed but did not implement."""
    attn = InterpretableMultiheadAttention(d_model=16, num_heads=4, dropout=0.0)

    # A shared value head projects to d_k, not d_model. The old implementation
    # used Linear(d_model, d_model) reshaped per head -- ordinary MHA.
    assert attn.value_projection.out_features == attn.d_k == 4
    assert attn.value_projection.out_features != attn.d_model

    x = torch.randn(2, 5, 16)
    out, weights = attn(x, return_attention=True)
    assert out.shape == (2, 5, 16)
    # Attention is averaged over heads, so it is [batch, seq, seq] not per-head.
    assert weights.shape == (2, 5, 5)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 5), atol=1e-5)


# ---------------------------------------------------------------------------
# 2.15 / 2.16 -- ablation parity and config plumbing
# ---------------------------------------------------------------------------
def test_ablation_arms_differ_only_by_the_attention_path():
    cfg = make_config(5)
    vanilla = get_momentum_transformer(cfg.model, "simple", cfg.training)
    attention = get_momentum_transformer(cfg.model, "enhanced", cfg.training)

    delta = count_parameters(attention) - count_parameters(vanilla)
    hidden = cfg.model.hidden_dim

    # Exactly the attention scorer + gate + its LayerNorm.
    scorer = (hidden * (hidden // 4) + hidden // 4) + ((hidden // 4) * 1 + 1)
    gate = hidden * 1 + 1
    layer_norm = 2 * hidden
    assert delta == scorer + gate + layer_norm
    assert delta < 0.15 * count_parameters(vanilla)


def test_factory_passes_every_configured_field_through():
    """The old factories hardcoded these, so Ray Tune searched dimensions that
    could not reach the model."""
    cfg = make_config(7)
    cfg.model.lstm_num_layers = 3
    cfg.model.lstm_dropout = 0.37
    cfg.model.short_window = 16
    cfg.model.feedforward_dim = 99

    model = get_momentum_transformer(cfg.model, "simple", cfg.training).model

    assert model.lstm.lstm.num_layers == 3
    assert model.lstm.lstm.dropout == pytest.approx(0.37)
    assert model.lstm.short_window == 16
    block = cast(TransformerEncoderBlock, model.transformer.layers[0])
    first_feedforward = cast(nn.Linear, block.feedforward[0])
    assert first_feedforward.out_features == 99


def test_all_models_return_a_uniform_tuple():
    cfg = make_config(5)
    x = torch.randn(2, 32, 5)

    for kind in ("simple", "enhanced"):
        out = get_momentum_transformer(cfg.model, kind, cfg.training)(x)
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == (2,)


# ---------------------------------------------------------------------------
# 2.17 -- positional encoding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d_model", [7, 8, 15, 16, 63])
def test_positional_encoding_handles_odd_d_model(d_model):
    pe = PositionalEncoding(d_model, dropout=0.0, max_len=32)
    x = torch.zeros(2, 10, d_model)
    assert pe(x).shape == (2, 10, d_model)


# ---------------------------------------------------------------------------
# 2.3 - 2.8 -- regime features
# ---------------------------------------------------------------------------
def test_regime_extractor_resolves_by_name_and_rejects_missing_columns():
    names = REGIME_COLUMNS + ["extra_a", "extra_b"]
    extractor = RegimeFeatureExtractor(names, lookback_short=8, lookback_long=24)
    assert extractor.idx["rsi"] == names.index("rsi")

    with pytest.raises(KeyError, match="cannot resolve"):
        RegimeFeatureExtractor(["return_1", "nope"], 8, 24)


def test_no_regime_feature_is_constant():
    """Four of the 21 were dead: regime_shift was identically zero and the
    volatility one-hots used daily thresholds on hourly returns."""
    torch.manual_seed(0)
    names = REGIME_COLUMNS + [f"f{i}" for i in range(3)]
    extractor = RegimeFeatureExtractor(names, lookback_short=8, lookback_long=24)

    x = torch.randn(64, 32, len(names)) * 0.01
    extractor.fit_thresholds(x[:, :, 0])
    features = extractor(x)

    assert features.shape == (64, len(REGIME_FEATURE_NAMES))
    variances = features.var(dim=0)
    dead = [REGIME_FEATURE_NAMES[i] for i in range(len(REGIME_FEATURE_NAMES))
            if variances[i] < 1e-12]
    assert not dead, f"constant regime features: {dead}"


def test_regime_shift_compares_distinct_windows():
    """It previously compared ``returns[-21:]`` with ``returns[-21:]`` -- itself."""
    names = REGIME_COLUMNS
    extractor = RegimeFeatureExtractor(names, lookback_short=4, lookback_long=16)

    x = torch.zeros(1, 20, len(names))
    x[0, :12, 0] = 0.001      # calm early period
    x[0, 12:, 0] = 0.05       # sharp recent shift

    # No fit_thresholds here: regime_shift does not depend on the volatility
    # terciles, and a single-sample batch cannot produce quantiles (the module
    # correctly refuses).
    features = extractor(x)
    regime_shift = features[0, REGIME_FEATURE_NAMES.index("regime_shift")]
    assert regime_shift > 1e-4


def test_trend_strength_keeps_its_sign():
    """abs(autocorr) made trending and mean-reverting regimes identical."""
    names = REGIME_COLUMNS
    extractor = RegimeFeatureExtractor(names, lookback_short=8, lookback_long=32)
    idx = REGIME_FEATURE_NAMES.index("trend_strength")

    trending = torch.zeros(1, 40, len(names))
    trending[0, :, 0] = torch.tensor(
        np.cumsum(np.full(40, 0.001)) + np.linspace(0, 0.001, 40), dtype=torch.float32
    )

    alternating = torch.zeros(1, 40, len(names))
    alternating[0, :, 0] = torch.tensor(
        [0.01 * (-1) ** i for i in range(40)], dtype=torch.float32
    )

    # trend_strength is independent of the volatility terciles.
    positive = extractor(trending)[0, idx]
    negative = extractor(alternating)[0, idx]

    assert positive > 0, "trending regime should show positive autocorrelation"
    assert negative < 0, "mean-reverting regime should show negative autocorrelation"


def test_regime_extractor_is_batched_and_order_independent():
    """A per-sample Python loop is gone; results must not depend on batching."""
    torch.manual_seed(1)
    names = REGIME_COLUMNS
    extractor = RegimeFeatureExtractor(names, lookback_short=8, lookback_long=24)

    x = torch.randn(12, 32, len(names)) * 0.01
    extractor.fit_thresholds(x[:, :, 0])

    whole = extractor(x)
    halves = torch.cat([extractor(x[:6]), extractor(x[6:])])
    assert torch.allclose(whole, halves, atol=1e-6)


def test_weight_network_normalises_its_input():
    """Without input normalisation a single large regime value saturates the
    sigmoid and pins the blend weight."""
    net = DynamicWeightNetwork(regime_feature_dim=21, hidden_dim=8, dropout=0.0)
    net.eval()

    normal = torch.randn(4, 21)
    extreme = normal.clone()
    extreme[:, 5] = 1e5  # the kind of value the old momentum ratio produced

    with torch.no_grad():
        out_extreme = net(extreme)

    assert torch.isfinite(out_extreme).all()
    assert (out_extreme > net.min_weight - 1e-6).all()
    assert (out_extreme < net.max_weight + 1e-6).all()


def test_weight_network_rejects_wrong_feature_count():
    """configTuned.py declared regime_feature_dim=10 against a 21-feature
    extractor and could not run; this now fails loudly."""
    net = DynamicWeightNetwork(regime_feature_dim=10, hidden_dim=8)
    with pytest.raises(ValueError, match="regime features"):
        net(torch.randn(2, 21))


def test_ensemble_weight_history_is_not_polluted_by_forward():
    """weight_history was mutated inside forward(), pooling train/val/test."""
    cfg = make_config(len(REGIME_COLUMNS) + 2)
    names = REGIME_COLUMNS + ["extra_a", "extra_b"]
    ensemble = get_ensemble_model(cfg, names)

    x = torch.randn(4, 32, len(names)) * 0.01

    ensemble(x)
    assert len(ensemble.weight_history) == 0, "forward() recorded without being asked"

    ensemble.record_weights(True)
    ensemble(x)
    assert len(ensemble.weight_history) == 4

    stats = ensemble.get_weight_statistics()
    assert stats["n_observations"] == 4


def test_ensemble_freeze_leaves_only_the_weight_network_trainable():
    cfg = make_config(len(REGIME_COLUMNS) + 2)
    names = REGIME_COLUMNS + ["extra_a", "extra_b"]
    ensemble = get_ensemble_model(cfg, names)

    ensemble.freeze_submodels()

    assert not any(p.requires_grad for p in ensemble.vanilla_model.parameters())
    assert not any(p.requires_grad for p in ensemble.attention_model.parameters())
    assert all(p.requires_grad for p in ensemble.weight_network.parameters())
