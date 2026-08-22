""" Statistical market-regime labelling, used for performance attribution.

*Scope Note:
This module cnotains the descriptive regime labeller used to attribute
realized PnL to market conditions. The features that drive the ensemble's
blend weight live in ``Models.Ensemble_model.RegimeFeatureExtractor``

*Thresholds:
Regime cutoffs are **fitted from data** rather than hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass 
from enum import Enum 
from typing import Dict, List, Optional 

import numpy as np 

class MarketRegime(Enum):
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"
    NORMAL = "normal"

@dataclass
class RegimeMetrics:
    volatility: float
    trend_strength: float
    autocorrelation: float 
    skewness: float 
    kurtosis: float 
    regime: MarketRegime
    confidence: float 

class StatisticalRegimeDetector:
    """ Classify a return window into a discrete regime 
    
    Call :meth: `fit` on the training-period return series to calibrate the 
    volatility cutoffs; without it, the fallback thresholds are used and a 
    wawrning-worthy ``fitted`` flag stays False 
    """

    def __init__(
        self,
        volatility_threshold_low: float = 0.004,
        volatility_threshold_high: float = 0.010,
        trend_threshold: float = 0.15,
        crisis_quantile: float = 0.01,
    ):
        self.vol_low = volatility_threshold_low
        self.vol_high = volatility_threshold_high
        self.trend_threshold = trend_threshold
        self.crisis_quantile = crisis_quantile
        self.crisis_threshold = -0.01
        self.fitted = False 

    def fit(self, returns: np.ndarray, window: int = 20) -> "StatisticalRegimeDetector":
        """Calibrate cutofs from rolling volatility terciles of ``returns``"""
        returns = np.asarray(returns, dtype = float)
        returns = returns[np.isfinite(returns)]
        if returns.size < window * 3:
            return self 

        # Rolling std via stride tricks, then terciles of the distribution
        shape = (returns.size - window + 1, window)
        strides = (returns.strides[0], returns.strides[0])
        windows = np.lib.stride_tricks.as_strided(returns, shape = shape, strides = strides)
        vols = np.std(windows, axis = 1)

        self.vol_low = float(np.quantile(vols, 0.33))
        self.vol_high = float(np.quantile(vols, 0.67))
        self.crisis_threshold = float(np.quantile(returns, self.crisis_quantile))
        self.fitted = True 
        return self 

    def detect_regime(self, returns: np.ndarray, window: int = 20) -> RegimeMetrics:
        recent = np.asarray(returns, dtype = float)[-window:]
        recent = recent[np.isfinite(recent)]

        if recent.size < 2:
            return RegimeMetrics(
                volatility = 0.0,
                trend_strength = 0.0,
                autocorrelation = 0.0,
                skewness = 0.0,
                kurtosis = 0.0,
                regime = MarketRegime.NORMAL,
                confidence = 0.0,
            )

        volatility = float(np.std(recent))
        mean_return = float(np.mean(recent))

        with np.errstate(invalid = "ignore"):
            autocorr = float(np.corrcoef(recent[:-1], recent[1:])[0,1])
        if not np.isfinite(autocorr):
            autocorr = 0.0

        skewness = self._standardized_moment(recent, 3)
        kurtosis = self._standardized_moment(recent, 4) - 3.0

        regime, confidence = self._classify(volatility, mean_return, autocorr)
        return RegimeMetrics(
            volatility = volatility,
            trend_strength = autocorr,
            autocorrelation = autocorr,
            skewness = skewness,
            kurtosis = kurtosis,
            regime = regime, 
            confidence = confidence,
        )

    def _classify(self, volatility, mean_return, autocorr):
        if mean_return < self.crisis_threshold and volatility > self.vol_high:
            return MarketRegime.CRISIS, 0.9
        if volatility > self.vol_high:
            return MarketRegime.HIGH_VOLATILITY, 0.8
        if volatility < self.vol_low:
            return MarketRegime.LOW_VOLATILITY, 0.8
        if autocorr > self.trend_threshold:
            return (
                MarketRegime.TRENDING_UP if mean_return > 0 else MarketRegime.TRENDING_DOWN
            ), 0.7
        if autocorr < -self.trend_threshold:
            return MarketRegime.MEAN_REVERTING, 0.7
        return MarketRegime.NORMAL, 0.5

    @staticmethod
    def _standardized_moment(data: np.ndarray, order: int) -> float:
        if data.size < order:
            return 0
        std = np.std(data)
        if std < 1e-12:
            return 0.0
        return float(np.mean(((data - np.mean(data)) / std) ** order))

class RollingRegimeDetector:
    """ Maintains a bounded history of regime classifications."""
    def __init__(self, detector: StatisticalRegimeDetector, window_size: int = 100):
        self.detector = detector
        self.window_size = window_size 
        self.regime_history: List[RegimeMetrics] = []

    def update(self, returns: np.ndarray) -> RegimeMetrics:
        metrics = self.detector.detect_regime(returns)
        self.regime_history.append(metrics)
        if len(self.regime_history) > self.window_size:
            self.regime_history.pop(0)
        return metrics 

    def get_regime_distribution(self) -> Dict[MarketRegime, float]:
        """ Share of history spent in each regime.
            
         The previous version had both the ``total`` assignment and the 
        ``return`` **inside** the loop, so it always returned after the first 
        element with a single-key dict
        """
        if not self.regime_history:
            return {}

        counts: Dict[MarketRegime, int] = {}
        for metrics in self.regime_history:
            counts[metrics.regime] = counts.get(metrics.regime, 0) + 1

        total = len(self.regime_history)
        return {regime: count / total for regime, count in counts.items()}

    def get_current_regime_persistence(self) -> int:
        if not self.regime_history:
            return 0
        current = self.regime_history[-1].regime 
        count = 0 
        for metrics in reversed(self.regime_history):
            if metrics.regime != current:
                break 
            count += 1
        return count 

def get_regime_distribution(self) -> Dict[MarketRegime, float]:
    """ Share of history spent in each regime."""

    if not self.regime_history:
        return {}

    counts: Dict[MarketRegime, int] = {}
    for metrics in self.regime_history:
        counts[metrics.regime] = counts.get(metrics.regime, 0) + 1

    total = len(self.regime_history)
    return {regime: count / total for regime, count in counts.items()}

def get_current_regime_persistence(self) -> int:
    if not self.regime_history:
        return 0
    current = self.regime_history[-1].regime
    count = 0
    for metrics in reversed(self.regime_history):
        if metrics.regime != current:
            break 
        count += 1
    return count 

def label_regimes(
    returns: np.ndarray,
    lookback: int = 252,
    detector: Optional[StatisticalRegimeDetector] = None,
) -> tuple:
    """ Label each bar with a regime id, using only trailing information"""

    returns = np.asarray(returns, dtype = float)
    detector = detector or StatisticalRegimeDetector().fit(returns[:lookback])

    labels: List[int] = []
    regime_to_id: Dict[MarketRegime, int] = {}
    names: Dict[str, str] = {}

    for i in range(len(returns)):
        if i < lookback:
            regime = MarketRegime.NORMAL
        else:
            regime = detector.detect_regime(returns[i - lookback:i]).regime

        if regime not in regime_to_id:
            regime_to_id[regime] = len(regime_to_id)
            names[str(regime_to_id[regime])] = regime.name.replace("_", " ").title()
        labels.append(regime_to_id[regime])

    return np.array(labels), names
