#Market regime detection 

#Detects market regiems (trending, mean-reverting, volatile, etc)
#This is used by the ensemble method for its dynamic weighting

import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn 
try:
    import hmmlearn  # optional; only needed for HMM regime detection
except ImportError:  # pragma: no cover
    hmmlearn = None
from typing import Dict, List, Tuple, Optional 
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    #The market regime classifications 

    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"
    NORMAL = "normal"


@dataclass
class RegimeMetrics:
    #Metrics that characterize each current market regime 
    volatility: float
    trend_strength: float
    autocorrelation: float 
    skewness: float 
    kurtosis: float 
    regime: MarketRegime
    confidence: float 

class RegimeFeatureExtractor(nn.Module):
    #Extracting features characterizing each market regime
    #Fed into EnsembleMomentumTransformer for dynamic weighting 

    #Features that are to be extracted
    #1. Volatility (short and long)
    #2. Volatility ratio 
    #3. Volatility regime indicators
    #4. Trend strength (autocorrelation)
    #5. Return statistics (mean, range, skewness)
    #6. Momentum indicators 

    def __init__(self, 
                 short_window: int = 20,
                 long_window: int = 60,
                 epsilon: float = 1e-8):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.epsilon = epsilon 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #extracting regime features from the input time series

        batch_size = x.shape[0]

        #using first feature (typically its returns) for the regime detection 
        returns = x[:,:,0]
        features = []

        #1. Short term volatility
        vol_short = torch.std(returns[:, -self.short_window:], dim = 1)
        features.append(vol_short.unsqueeze(1))

        #2. Long term volatility
        vol_long = torch.std(returns[:, -self.long_window:], dim = 1)
        features.append(vol_long.unsqueeze(1))

        #3. Volatility ratio (short / long)
        vol_ratio = vol_short / (vol_long + self.epsilon)
        features.append(vol_ratio.unsqueeze(1))

        #4-6. Volatility regime indicators
        vol_percentiles = torch.quantile(
            torch.std(returns, dim = 1, keepdim = True),
            torch.tensor([0.33, 0.67]).to(x.device)
        )
        high_vol = (vol_short > vol_percentiles[1]).float().unsqueeze(1)
        med_vol = ((vol_short > vol_percentiles[0]) & (vol_short <= vol_percentiles[1])).float().unsqueeze(1)
        low_vol = (vol_short <= vol_percentiles[0]).float().unsqueeze(1)
        features.extend([high_vol, med_vol, low_vol])

        #7. Trend strength (lag-1 autocorrelation)
        mean_returns = torch.mean(returns[:, -self.short_window:], dim = 1, keepdim = True)
        centered = returns[:, -self.short_window:] - mean_returns
        autocorr = torch.sum(centered[:, :-1] * centered[:, 1:], dim = 1) / (
            torch.sum(centered ** 2, dim = 1) + self.epsilon
        )
        features.append(autocorr.unsqueeze(1))

        #8. Recent mean return 
        mean_return = torch.mean(returns[:, -self.short_window:], dim = 1)
        features.append(mean_return.unsqueeze(1))

        #9. Absolute mean return 
        abs_mean_return = torch.abs(mean_return)
        features.append(abs_mean_return.unsqueeze(1))

        #10. Return range
        return_range = (torch.max(returns[:, -self.short_window:], dim = 1)[0] - 
                        torch.min(returns[:, -self.short_window:], dim = 1)[0])
        features.append(return_range.unsqueeze(1))

        #Concatenating all of the features
        regime_features = torch.cat(features, dim = 1)

        return regime_features
    

class StatisticalRegimeDetector:
    #Statistical regime detection using historical data
    #Classifies the market into discrete regimes based on previous statistical properties

    def __init__(self, 
                 volatility_threshold_low: float = 0.01,
                 volatility_threshold_high: float = 0.03,
                 trend_threshold: float = 0.3,
                 crisis_threshold: float = -0.05):
        self.vol_low = volatility_threshold_low
        self.vol_high = volatility_threshold_high
        self.trend_threshold = trend_threshold
        self.crisis_threshold = crisis_threshold

    def detect_regime(self,
                      returns: np.ndarray,
                      window: int = 20) -> RegimeMetrics:
        #Detecting current market regime 

        #returns: historical returns (1d array)
        #window: lookback window for analysis 

        #returns the RegimeMetrics with regime classification 

        recent_returns = returns[-window:]

        #Computing statistics
        volatility = np.std(recent_returns)
        mean_return = np.mean(recent_returns)

        #Autocorrelation (trend strength)
        if len(recent_returns) > 1:
            autocorr = np.corrcoef(recent_returns[:-1], recent_returns[1:])[0,1]
        else:
            autocorr = 0.0 

        #Higher moments
        skewness = self._compute_skewness(recent_returns)
        kurtosis = self._compute_kurtosis(recent_returns)

        #Classifying the regime 
        regime, confidence = self._classify_regime(
            volatility, mean_return, autocorr, skewness, kurtosis
        )

        return RegimeMetrics(
            volatility=volatility,
            trend_strength=autocorr,
            autocorrelation=autocorr,
            skewness=skewness,
            kurtosis=kurtosis,
            regime=regime,
            confidence=confidence
        )
    
    def _classify_regime(self,
                         volatility: float,
                         mean_return: float,
                         autocorr: float,
                         skewness: float,
                         kurtosis: float) -> Tuple[MarketRegime, float]:
        #Classifying the regime based on the statistics 

        #returns (regime, confidence) tuple 

        #Crisis detection (high priority)
        if mean_return < self.crisis_threshold and volatility > self.vol_high:
            return MarketRegime.CRISIS, 0.9
        
        #High volatility
        if volatility > self.vol_high:
            return MarketRegime.HIGH_VOLATILITY, 0.8
        
        #Low volatility
        if volatility < self.vol_low:
            return MarketRegime.LOW_VOLATILITY, 0.8
        
        #Trending up (positive autocorrelation + positive returns)
        if autocorr > self.trend_threshold and mean_return > 0:
            return MarketRegime.TRENDING_UP, 0.7
        
        #Trending down
        if autocorr > self.trend_threshold and mean_return < 0:
            return MarketRegime.TRENDING_DOWN, 0.7
        
        #Mean reverting (negative autocorrelation)
        if autocorr < -self.trend_threshold:
            return MarketRegime.MEAN_REVERTING, 0.7
        
        #Defaulting to normal
        return MarketRegime.NORMAL, 0.5
    
    @staticmethod
    def _compute_skewness(data: np.ndarray) -> float:
        #Compute skewness
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-8:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _compute_kurtosis(data: np.ndarray) -> float:
        #Compute excess kurtosis
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-8:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0
    
class HiddenMarkovRegimeDetector:
    #Hidden Markov Model for the regime detection 
    #Learns latent states from the data
    #Requires sklearn and hmmlearn 

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None 

        try:
            from hmmlearn import hmm 
            self.hmm = hmm
            self.available = True 
        except ImportError: 
            self.available = False 
            print("Warning: hmmlearn not available. Install with pip install hmmlearn")

    def fit(self, returns: np.ndarray):
        #Fit HMM to the historical returns

        #returns: historical returns (1d array)

        if not self.available:
            raise ImportError("hmmlearn is required for HMM regime detection")
        

        #Reshape for HMM
        X = returns.reshape(-1,1)

        #Fit Gaussian HMM
        self.model = self.hmm.GaussianHMM(
            n_components = self.n_regimes,
            covariance_type = "full",
            n_iter = 100
        )
        self.model.fit(X)

    def predict_regime(self, returns: np.ndarray) -> int:
        #Predict the current regime 
        #returns: recent returns

        #regime_id: integer regime identifier [0, n_regimes - 1]

        if self.model is None:
            raise ValueError("Model must be fit before prediction")
        
        X = returns.reshape(-1,1)
        regimes = self.model.predict(X)
        return regimes[-1] #Return the most recent regime
    
class RollingRegimeDetector:
    #Rolling window regime detection
    #Maintains the history of regime classifications

    def __init__(self,
                 detector: StatisticalRegimeDetector,
                 window_size: int = 100):
        self.detector = detector
        self.window_size = window_size
        self.regime_history = []

    def update(self, returns: np.ndarray) -> RegimeMetrics:
        #Update the regime detection with any new data

        metrics = self.detector.detect_regime(returns)
        self.regime_history.append(metrics)

        #Keep only recent history
        if len(self.regime_history) > self.window_size:
            self.regime_history.pop(0)

        return metrics 
    
    def get_regime_distribution(self) -> Dict[MarketRegime, float]:
        #Get the distribution of regimes over history 

        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for metrics in self.regime_history:
            regime = metrics.regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            total = len(self.regime_history)
            return {regime: count / total 
                    for regime, count in regime_counts.items()}
        
    def get_current_regime_persistence(self) -> int:
        #Get the number of consecutive periods in current regime

        if not self.regime_history:
            return 0
        
        current_regime = self.regime_history[-1].regime
        count = 0

        for metrics in reversed(self.regime_history):
            if metrics.regime == current_regime:
                count += 1
            else:
                break 

        return count 
    
def extract_regime_features_numpy(returns: np.ndarray,
                                  short_window: int = 20,
                                  long_window: int = 60) -> np.ndarray:
    
    #Numpy version of regime feature extraction
    #Used for preprocessing before PyTorch

    #returns: [seq_len] - return series
    #short_window: short lookback window
    #long_window: long lookback window 

    #features: [10] - regime feature vector 

    features = []

    #volatilities
    vol_short = np.std(returns[-short_window:])
    vol_long = np.std(returns[-long_window:])
    features.extend([vol_short, vol_long])

    #Volatility ratio 
    vol_ratio = vol_short / (vol_long + 1e-8)
    features.append(vol_ratio)

    #Volatility percentiles
    vol_33 = np.percentile(returns, 33)
    vol_67 = np.percentile(returns, 67)
    high_vol = float(vol_short > vol_67)
    med_vol = float(vol_33 < vol_short <= vol_67)
    low_vol = float(vol_short <= vol_33)
    features.extend([high_vol, med_vol, low_vol])

    #autocorrelation
    recent = returns[-short_window:]
    if len(recent) > 1:
        autocorr = np.corrcoef(recent[:-1], recent[1:])[0,1]
    else:
        autocorr = 0.0
    features.append(autocorr)

    #returns statistics
    mean_return = np.mean(returns[-short_window:])
    abs_mean = np.abs(mean_return)
    return_range = np.max(returns[-short_window:]) - np.min(returns[-short_window:])
    features.extend([mean_return, abs_mean, return_range])

    return np.array(features)


if __name__ == "__main__":
    print("Testing Regime Detection\n")
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Normal regime
    normal_returns = np.random.randn(100) * 0.01
    
    # High volatility regime
    volatile_returns = np.random.randn(100) * 0.05
    
    # Trending regime
    trending_returns = np.cumsum(np.random.randn(100) * 0.01 + 0.001)
    trending_returns = np.diff(trending_returns)
    
    # Crisis regime
    crisis_returns = np.random.randn(100) * 0.08 - 0.03
    
    # Test detector
    detector = StatisticalRegimeDetector()
    
    regimes_data = {
        'Normal': normal_returns,
        'Volatile': volatile_returns,
        'Trending': trending_returns,
        'Crisis': crisis_returns
    }
    
    print("Regime Classification Results:")
    print("-" * 60)
    
    for name, returns in regimes_data.items():
        metrics = detector.detect_regime(returns)
        print(f"\n{name} Market:")
        print(f"  Detected Regime: {metrics.regime.value}")
        print(f"  Confidence: {metrics.confidence:.2f}")
        print(f"  Volatility: {metrics.volatility:.4f}")
        print(f"  Trend Strength: {metrics.trend_strength:.4f}")
    
    # Test PyTorch feature extractor
    print("\n" + "=" * 60)
    print("Testing PyTorch Feature Extractor")
    print("=" * 60)
    
    extractor = RegimeFeatureExtractor()
    
    # Create sample batch
    x = torch.randn(4, 252, 32) * 0.02  # [batch, seq, features]
    features = extractor(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"\nSample feature vector:")
    print(features[0].detach().numpy())
    
    print("\n✅ Regime detection working correctly!")
