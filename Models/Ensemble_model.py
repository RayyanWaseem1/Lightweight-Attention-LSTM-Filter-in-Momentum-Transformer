""" Regime-weighted ensemble of the vanilla and attention Momentum Transformers"""

from __future__ import annotations 

from collections import deque 
from typing import Dict, List, Optional, Sequence, Tuple 

import numpy as np 
import torch
import torch.nn as nn

from .Momentum_transformer import (
    MomentumTransformerDualPath,
    MomentumTransformerSimple,
    model_kwargs_from_config,
)

EPS = 1e-8

# Feature columns the regime extractor needs, by name
REQUIRED_REGIME_COLUMNS = {
    "returns": "return_1",
    "rsi": "rsi",
    "volatility": "volatility_21",
    "momentum": "momentum_35",
    "market_return": "market_return_1",
    "market_volatility": "market_volatility",
    "beta": "beta",
}

REGIME_FEATURE_NAMES: List[str] = [
    # Stock-level
    "high_vol", "med_vol", "low_vol",
    "trend_strength", "vol_ratio", "mean_return", "abs_mean_return",
    "return_range", "skewness", "momentum_short",
    "rsi_current", "vol_feature_mean", "momentum_feature_current",

    # Cross-sectional
    "realtive_return", "relative_vol", "beta", "relative_strength",
    "market_vol_level",

    # Temporal
    "vol_persistence", "vol_trend", "regime_shift",
]

class RegimeFeatureExtractor(nn.Module):
    """ Batched extraction of 21 regime descriptors from a feature window"""

    def __init__(
        self,
        feature_names: Sequence[str],
        lookback_short: int = 21,
        lookback_long: int = 63,
        column_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long 
        self.feature_names = list(feature_names)

        column_map = column_map or REQUIRED_REGIME_COLUMNS
        lookup = {name: i for i, name in enumerate(self.feature_names)}

        missing = [col for col in column_map.values() if col not in lookup]
        if missing:
            raise KeyError(
                "RegimeFeatureExtractor cannot resolve required feature columns "
                f"{missing}. Available columns: {sorted(lookup)[:20]}..."
                "Regime featues are resolved by name; updated column_map if the "
                "feature set changed."
            )

        self.idx = {role: lookup[col] for role, col in column_map.items()}

        # Volatility terciles, fitted on the training split. Registered as
        # buffers so they move with the module and are saved in the state dict
        self.register_buffer("vol_q33", torch.tensor(float("nan")))
        self.register_buffer("vol_q67", torch.tensor(float("nan")))

    # threshold fitting
    @torch.no_grad()
    def fit_thresholds(self, train_returns: torch.Tensor) -> "RegimeFeatureExtractor":
        """ Fit volatility terciles from training-set short-window volatitilies.
        
        ``train_returns``: [n_samples, seq_len] of per-bar returns.
        """
        window = train_returns[:, -self.lookback_short:]
        vols = window.std(dim = 1)
        vols = vols[torch.isfinite(vols)]
        if vols.numel() < 3:
            raise ValueError("Not enough samples to fit volatility terciles")

        self.vol_q33 = torch.quantile(vols, 0.33).detach().clone()
        self.vol_q67 = torch.quantile(vols, 0.67).detach().clone()
        return self 

    @property
    def thresholds_fitted(self) -> bool:
        return bool(torch.isfinite(self.vol_q33) and torch.isfinite(self.vol_q67))

    # helpers 
    @staticmethod 
    def _skewness(w: torch.Tensor) -> torch.Tensor:
        mu = w.mean(dim = 1, keepdim = True)
        sd = w.std(dim = 1, keepdim = True)
        return (((w - mu) / (sd + EPS)) ** 3).mean(dim = 1)

    @staticmethod 
    def _lag1_autocorr(w: torch.Tensor) -> torch.Tensor:
        """ Signed lag-1 autocorrelation, batched"""
        centered = w - w.mean(dim = 1, keepdim = True)
        num = (centered[:, :-1] * centered[:, 1:]).sum(dim = 1)
        den = (centered ** 2).sum(dim = 1)
        return num / (den + EPS)

    # API
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ ``x``: [batch, seq_len, n_features] -> [batch, 21]."""
        long_w, short_w = self.lookback_long, self.lookback_short 

        returns = x[:, -long_w:, self.idx["returns"]]   # [B, long]
        short_returns = returns[:, -short_w:]   # [B, short]

        market = x[:, -long_w:, self.idx["market_return"]]
        market_short = market[:, -short_w:]

        # 1. Volatility
        vol_short = short_returns.std(dim = 1)
        vol_long = returns.std(dim = 1)
        vol_ratio = vol_short / (vol_long + EPS)

        if self.thresholds_fitted:
            q33, q67 = self.vol_q33, self.vol_q67
        else:
            # pre-fit fallback: use this batch's own quantiles. Only reachable
            # before fit_thresholds has run 
            finite = vol_short[torch.isfinite(vol_short)]
            q33 = torch.quantile(finite, 0.33) if finite.numel() > 2 else vol_short.mean()
            q67 = torch.quantile(finite, 0.67) if finite.numel() > 2 else vol_short.mean()

        high_vol = (vol_short > q67).float()
        med_vol = ((vol_short > q33) & (vol_short <= q67)).float() 
        low_vol = (vol_short <= q33).float() 

        # 2. Trend strength (signed)
        trend_strength = self._lag1_autocorr(returns)

        # 3. Distribution shape 
        mean_return = returns.mean(dim = 1)
        abs_mean_return = returns.abs().mean(dim = 1)
        return_range = returns.max(dim = 1).values - returns.min(dim = 1).values
        skewness = self._skewness(returns)

        # 4. Momentum as a cumulative return 
        momentum_short = torch.prod(1.0 + short_returns, dim = 1) - 1.0 

        # 5. Named auxiliary features 
        rsi_current = x[:, -1, self.idx["rsi"]]
        vol_feature_mean = x[:, -short_w:, self.idx["volatility"]].mean(dim = 1)
        momentum_feature_current = x[:, -1, self.idx["momentum"]]

        # 6. Cross-sectional, from this sample's own window 
        relative_return = short_returns.mean(dim = 1) - market_short.mean(dim = 1)
        market_vol = market_short.std(dim = 1)
        relative_vol = vol_short / (market_vol + EPS)
        beta = x[:, -1, self.idx["beta"]]
        relative_strength = short_returns.sum(dim = 1) - market_short.sum(dim = 1)
        market_vol_level = x[:, -short_w:, self.idx["market_volatility"]].mean(dim = 1)

        # 7. Temporal 
        step = max(1, short_w // 4)
        if returns.shape[1] >= short_w + step:
            sub = returns.unfold(1, short_w, step) # [B, n_sub, short]
            sub_vols = sub.std(dim = 2)
            vol_persistence = sub_vols.std(dim = 1)
            vol_trend = sub_vols[:, -1] - sub_vols[:, 0]
        else:
            vol_persistence = torch.zeros_like(vol_short)
            vol_trend = torch.zeros_like(vol_short)

        # regime_shift compares the recent window to the EARLIER part of long window 
        if returns.shape[1] > short_w:
            historical_mean = returns[:, :-short_w].mean(dim = 1)
        else: 
            historical_mean = torch.zeros_like(mean_return)
        regime_shift = (short_returns.mean(dim = 1) - historical_mean).abs()

        features = torch.stack(
            [
                high_vol, med_vol, low_vol,
                trend_strength, vol_ratio, mean_return, abs_mean_return,
                return_range, skewness, momentum_short,
                rsi_current, vol_feature_mean, momentum_feature_current,
                relative_return, relative_vol, beta, relative_strength,
                market_vol_level,
                vol_persistence, vol_trend, regime_shift,
            ],
            dim = 1,
        )
        return torch.nan_to_num(features, nan = 0.0, posinf = 0.0, neginf = 0.0)

class DynamicWeightNetwork(nn.Module):
    """ Maps regime descriptors to the attention model's blend weight"""

    def __init__(
        self,
        regime_feature_dim: int = 21,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        min_weight: float = 0.2,
        max_weight: float = 0.8,
    ):
        super().__init__()
        self.regime_feature_dim = regime_feature_dim 
        self.min_weight = min_weight 
        self.max_weight = max_weight 

        self.network = nn.Sequential(
            # LayerNorm on the input: regime descriptors live on wildly 
            # different scales (RSI ~50 next to mean returns ~1e-4), and 
            # without it a single large value saturates the sigmoud and pins 
            # the blend weight for that sample 
            nn.LayerNorm(regime_feature_dim),
            nn.Linear(regime_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, regime_features: torch.Tensor) -> torch.Tensor:
        if regime_features.shape[1] != self.regime_feature_dim:
            raise ValueError(
                f"DynamicWeightNetwork expect {self.regime_feature_dim} regime "
                f"features, got {regime_features.shape[1]}. "
                "Check EnsembleConfig.regime_feature_dim against "
                "len(REGIME_FEATURE_NAMES)."
            )
        raw = self.network(regime_features).squeeze(-1)
        return self.min_weight + (self.max_weight - self.min_weight) * raw 

class EnsembleMomentumTransformer(nn.Module):
    """ ``w * attention + (1 - w) * vanilla``, with ``w`` chosen per regime"""

    def __init__(
        self,
        feature_names: Sequence[str],
        model_kwargs: Dict,
        regime_feature_dim: int = 21,
        weight_hidden_dim: int = 32,
        weight_dropout: float = 0.2,
        lookback_short: int = 21,
        lookback_long: int = 63,
    ):
        super().__init__()

        if regime_feature_dim != len(REGIME_FEATURE_NAMES):
            raise ValueError(
                f"regime_feature_dim = {regime_feature_dim} but the extractor emits "
                f"{len(REGIME_FEATURE_NAMES)}. These must match "
            )

        self.vanilla_model: MomentumTransformerSimple = MomentumTransformerSimple(**model_kwargs)
        self.attention_model: MomentumTransformerDualPath = MomentumTransformerDualPath(**model_kwargs)

        self.regime_extractor = RegimeFeatureExtractor(
            feature_names = feature_names,
            lookback_short = lookback_short,
            lookback_long = lookback_long,
        )
        self.weight_network = DynamicWeightNetwork(
            regime_feature_dim = regime_feature_dim,
            hidden_dim = weight_hidden_dim,
            dropout = weight_dropout,
        )

        self.weight_history = deque(maxlen = 100_000)
        self._recording = False 

    # training-stage control 
    def freeze_submodels(self) -> "EnsembleMomentumTransformer":
        """ Freeze both sub-models so only the weight network trains. 
        
        Used by the two-stage schedule the README describes: train each arm 
        independently on the training split, freeze, then fit the blend on the 
        validation split. 
        """
        for param in self.vanilla_model.parameters():
            param.requires_grad = False 
        for param in self.attention_model.parameters():
            param.requires_grad = False
        self.vanilla_model.eval()
        self.attention_model.eval()
        return self

    def record_weights(self, enabled: bool = True) -> "EnsembleMomentumTransformer":
        """ Enable/disable blend-weight recording (off during training)"""
        self._recording = enabled 
        return self 

    def reset_weight_history(self) -> None:
        self.weight_history.clear() 

    # API 
    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        ex_ante_vol: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        vanilla_pred, _ = self.vanilla_model(x, ex_ante_vol = ex_ante_vol)
        attention_pred, _ = self.attention_model(x, ex_ante_vol = ex_ante_vol)

        regime_features = self.regime_extractor(x)
        attention_weight = self.weight_network(regime_features)
        vanilla_weight = 1.0 - attention_weight 

        ensemble_pred = vanilla_weight * vanilla_pred + attention_weight * attention_pred 

        # Recording is explicit and off by default
        if self._recording: 
            self.weight_history.extend(
                attention_weight.detach().cpu().numpy().tolist()
            )

        if return_components:
            metadata = {
                "vanilla_predictions": vanilla_pred.detach(),
                "attention_predictions": attention_pred.detach(),
                "attention_weights": attention_weight.detach(),
                "vanilla_weights": vanilla_weight.detach(),
                "mean_attention_weight": float(attention_weight.detach().mean()),
                "regime_features": regime_features.detach(),
            }

            return ensemble_pred, metadata
        return ensemble_pred, None 

    def get_weight_statistics(self) -> Dict[str, float]:
        if not self.weight_history:
            return {}
        weights = np.asarray(self.weight_history, dtype = float)
        return {
            "mean_attention_weight": float(np.mean(weights)),
            "std_attention_weight": float(np.std(weights)),
            "min_attention_weight": float(np.min(weights)),
            "max_attention_weight": float(np.max(weights)),
            "median_attention_weight": float(np.median(weights)),
            "attention_usage_pct": float(np.mean(weights > 0.5) * 100),
            "vanilla_usage_pct": float(np.mean(weights <= 0.5) * 100),
            "n_observations": int(weights.size),
        }

def get_ensemble_model(config, feature_names: Sequence[str]) -> EnsembleMomentumTransformer:
    """ Build the ensemble from a full ``Config`` plus the resolved feature order"""
    return EnsembleMomentumTransformer(
        feature_names = feature_names,
        model_kwargs = model_kwargs_from_config(config.model, config.training),
        regime_feature_dim=config.ensemble.regime_feature_dim,
        weight_hidden_dim=config.ensemble.weight_hidden_dim,
        weight_dropout=config.ensemble.weight_dropout,
        lookback_short=config.regime_detector.lookback_short,
        lookback_long=config.regime_detector.lookback_long,
    )
