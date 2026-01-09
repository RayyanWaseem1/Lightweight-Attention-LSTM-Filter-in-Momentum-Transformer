#Ensemble Model with Dynamic Weighting
#Combines vanilla and attention-enhanced models with learned weights

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque

from .Momentum_transformer import MomentumTransformerSimple, MomentumTransformerDualPath

class RegimeFeatureExtractor(nn.Module):
    #Extracting regime features from input for weight network

    def __init__(self, lookback_short: int = 21, lookback_long: int = 63):
        super().__init__()
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def forward(self, x: torch.Tensor, market_data: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        #Extract the statistical features from input time series 

        #x: [batch, seq_len, input_dim]

        #Features: [batch, num_features]

        batch_size = x.shape[0]
        features_list = []

        for i in range(batch_size):
            x_sample = x[i, -self.lookback_long:, :] #[lookback_long, input_dim]

            ### STOCK LEVEL FEATURES ###
            returns = x_sample[:, 0] #[lookback_long]
            rsi = x_sample[:, 1] if x_sample.shape[1] > 1 else returns #Adding RSI
            volatility_feature = x_sample[:, 10] if x_sample.shape[1] > 10 else returns #Adding vol
            momentum_feature = x_sample[:, 15] if x_sample.shape[1] > 15 else returns #Adding momentum

            #1. Volatility features 
            vol_short = torch.std(returns[-self.lookback_short:])
            vol_long = torch.std(returns) if len(returns) >= self.lookback_long else vol_short
            vol_ratio = vol_short / (vol_long + 1e-8)

            #volatility regime indicator
            high_vol = float(vol_short > 0.03)
            med_vol = float(0.01 <= vol_short <= 0.03)
            low_vol = float(vol_short < 0.01)

            #2. Trend strength (autocorrelation)
            if len(returns) > 1:
                returns_np = returns.cpu().numpy()
                autocorr = np.corrcoef(returns_np[:-1], returns_np[1:])[0,1]
                trend_strength = abs(autocorr) if not np.isnan(autocorr) else 0.5
            else:
                trend_strength = 0.5

            #3. Statistical features
            mean_returns = torch.mean(returns)
            abs_mean_returns = torch.mean(torch.abs(returns))
            return_range = torch.max(returns) - torch.min(returns)
            skewness = self._compute_skewness(returns)

            #4. Momentum Indicators
            momentum_short = returns[-1] / (returns[-self.lookback_short] + 1e-8) -1 if len(returns) >= self.lookback_short else 0.0
            momentum_long = returns[-1] / (returns[0] + 1e-8) - 1

            #5. Additional Features
            rsi_current = float(rsi[-1])
            vol_feature_mean = float(torch.mean(volatility_feature[-self.lookback_short:]))
            momentum_feature_current = float(momentum_feature[-1])

            ### CROSS SECTIONAL FEATURES ###
            #Stock vs Current Market Context
            if market_data is not None and 'market_returns' in market_data:
                market_returns = market_data['market_returns'][-self.lookback_long:]

                #1. Relative Return (stock vs market)
                market_mean = torch.mean(market_returns[-self.lookback_short:])
                stock_mean = torch.mean(returns[-self.lookback_short:])
                relative_return = float(stock_mean - market_mean)

                #2. Relative Volatility (stock/market ratio)
                market_vol = torch.std(market_returns[-self.lookback_short:])
                relative_vol = float(vol_short / (market_vol + 1e-8))

                #3. Beta (Correlated with market)
                stock_np = returns[-self.lookback_short:].cpu().numpy()
                market_np = market_returns[-self.lookback_short:].cpu().numpy()
                if len(stock_np) > 1 and len(market_np) > 1:
                    corr = np.corrcoef(stock_np, market_np)[0,1]
                    beta = corr if not np.isnan(corr) else 1.0
                else:
                    beta = 1.0

                #4. Relative Strength (cumulative outperformance)
                cum_stock = torch.sum(returns[-self.lookback_short:])
                cum_market = torch.sum(market_returns[-self.lookback_short:])
                relative_strength = float(cum_stock - cum_market)

                #5. Market volatility level (is the market in a high volatility regime?)
                market_vol_level = float(market_vol)
            else:
                #Defaults if there is no market data
                relative_return = 0.0
                relative_vol = 1.0
                beta = 1.0
                relative_strength = 0.0
                market_vol_level = 0.02 

            ### TEMPORAL FEATURES ###
            #Time context: Early vs Late in the regime

            #1. Volatility Persistence (how long has the volatility been high?)
            vol_series = []
            for j in range(len(returns)-1, max(0, len(returns)-self.lookback_long), -5):
                if j >= self.lookback_short:
                    v = torch.std(returns[max(0, j-self.lookback_short):j+1])
                    vol_series.append(float(v))

            if len(vol_series) > 1:
                vol_persistence = float(np.std(vol_series)) #High = recently spiked, Low = persistent spike
            else:
                vol_persistence = 0.0

            #2. Volatility Trend (is the volatility increasing or decresing?)
            if len(vol_series) > 1:
                vol_trend = vol_series[0] - vol_series[-1] #Positive = increasing vol
            else:
                vol_trend = 0.0

            #3. Regime Shift Indicator (did the statistics change recently?)
            if len(returns) >= self.lookback_long:
                recent_mean = torch.mean(returns[-self.lookback_short:])
                historical_mean = torch.mean(returns[-self.lookback_short:])
                regime_shift = float(abs(recent_mean - historical_mean))
            else:
                regime_shift = 0.0

            


            #COMBINING ALL FEATURES INTO A VECTOR
            feature_vector = torch.tensor([
                #Stock specific features
                high_vol, med_vol, low_vol, 
                trend_strength, 
                float(vol_ratio), 
                float(mean_returns), 
                float(abs_mean_returns), 
                float(return_range), 
                float(skewness), 
                float(momentum_short),
                rsi_current,
                vol_feature_mean,
                momentum_feature_current,

                #Cross sectional features
                relative_return, #stock return vs market
                relative_vol, #stock vol/market vol (key: 2.5 = crisis, 1.2 = normal)
                beta, #Correlation with the market
                relative_strength, #Cumulative outperformance
                market_vol_level, #Absolute market volatility

                #Temporal features
                vol_persistence, #volatility stability
                vol_trend, #vol increasing/decreasing
                regime_shift, #recent regime change
            ], dtype = torch.float32, device = x.device) 

            features_list.append(feature_vector)

        return torch.stack(features_list) #[batch, 21]
    
    def _compute_skewness(self, x: torch.Tensor) -> float:
        #computing skew of a tensor
        mean = torch.mean(x)
        std = torch.std(x)
        if std < 1e-8:
            return 0.0
        
        skew = torch.mean(((x-mean) / std) ** 3)
        return float(skew)
    

class DynamicWeightNetwork(nn.Module):
    #A neural network that learns how to weight the vanilla vs attention model based on the current regime

    def __init__(self,
                 regime_feature_dim: int = 21,
                 hidden_dim: int = 32,
                 dropout: float = 0.2,
                 min_weight: float = 0.2,
                 max_weight: float = 0.8):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.network = nn.Sequential(
            nn.Linear(regime_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() #Output in [0,1]
        )

    def forward(self, regime_features: torch.Tensor) -> torch.Tensor:
        raw_weight = self.network(regime_features).squeeze(-1)

        #Constrain the output to [0.2,0.8] instead of [0,1]
        attention_weight = self.min_weight + (self.max_weight - self.min_weight) * raw_weight

        return attention_weight
    

class EnsembleMomentumTransformer(nn.Module):
    #Ensemble of vanilla and attention-enhanced Momentum Transformer
    #Has dynamic weighting based on current market regime

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_transformer_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 regime_feature_dim: int = 21,
                 weight_hidden_dim: int = 32):
        super().__init__()

        #Vanilla Model
        self.vanilla_model = MomentumTransformerSimple(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        #Attention enhanced model
        self.attention_model = MomentumTransformerDualPath(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            num_transformer_layers = num_transformer_layers,
            num_heads = num_heads,
            dropout = dropout, 
            use_lstm_attention= True
        )

        #Regime feature extractor
        self.regime_extractor = RegimeFeatureExtractor(
            lookback_short=21,
            lookback_long=63
        )

        #Dynamic weight network
        self.weight_network = DynamicWeightNetwork(
            regime_feature_dim = regime_feature_dim,
            hidden_dim = weight_hidden_dim,
            dropout = dropout
        )

        #Statistics tracking
        self.weight_history = deque(maxlen = 1000)

    def forward(self,
                x: torch.Tensor,
                return_components: bool = False,
                market_data: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        
        #x: [batch, seq_len, input_dim]
        #return_components: Whether to return the individual predictions and weights
        #market data: optional dict with market context

        #ensemble_positions: [batch] - weighted combination
        #metadata: dict with individual predictions and weights - optional

        #Getting predictions from both models
        vanilla_pred = self.vanilla_model(x)
        attention_pred = self.attention_model(x)[0]

        #Extracting regime features WITH market context
        regime_features = self.regime_extractor(x, market_data = market_data)

        #computing dynamic weights
        attention_weight = self.weight_network(regime_features)
        vanilla_weight = 1.0 - attention_weight

        #Weighted combination
        ensemble_pred = vanilla_weight * vanilla_pred + attention_weight * attention_pred

        #tracking weights
        self.weight_history.extend(attention_weight.detach().cpu().numpy().tolist())

        if return_components:
            metadata = {
                "vanilla_predictions": vanilla_pred.detach(),
                "attention_predictions": attention_pred.detach(),
                "vanilla_weights": vanilla_weight.detach(),
                "attention_weights": attention_weight.detach(),
                "mean_attention_weight": float(torch.mean(attention_weight)),
                "regime_features": regime_features.detach()
            }
            return ensemble_pred, metadata
        return ensemble_pred, None
    
    def get_weight_statistics(self) -> Dict[str, float]:
        #Getting statisticsa bout weight distribution 

        #Returns dict with weight statistics

        if not self.weight_history:
            return {}
        
        weights = np.array(self.weight_history)

        return {
            "mean_attention_weight": float(np.mean(weights)),
            "std_attention_weight": float(np.std(weights)),
            "min_attention_weight": float(np.min(weights)),
            "max_attention_weight": float(np.max(weights)),
            "median_attention_weight": float(np.median(weights)),
            "attention_usage_pct": float(np.mean(weights > 0.5) * 100),
            "vanilla_usage_pct": float(np.mean(weights < 0.5) * 100)
        }
    
def get_ensemble_model(config):
    #factory function to create the ensemble model

    return EnsembleMomentumTransformer(
        input_dim = config.model.input_dim,
        hidden_dim = config.model.hidden_dim,
        num_transformer_layers=config.model.num_transformer_layers,
        num_heads = config.model.num_attention_heads,
        dropout = config.model.transformer_dropout,
        regime_feature_dim = config.ensemble.regime_feature_dim,
        weight_hidden_dim=config.ensemble.weight_hidden_dim
    )

if __name__ == "__main__":
    from Models.config import get_production_config

    config = get_production_config()
    config.model.input_dim = 32

    ensemble = get_ensemble_model(config)

    batch_size = 16
    seq_len = 252
    x = torch.randn(batch_size, seq_len, config.model.input_dim)

    positions, metadata = ensemble(x, return_components = True)

    print(f"Input shape: {x.shape}")
    print(f"Ensemble positions shape: {positions.shape}")
    print(f"Position range: [{positions.min():.3f}, {positions.max():.3f}]")
    print(f"\nVanilla predictions range: [{metadata['vanilla_predictions'].min():.3f}, {metadata['vanilla_predictions'].max():.3f}]")
    print(f"Attention predictions range: [{metadata['attention_predictions'].min():.3f}, {metadata['attention_predictions'].max():.3f}]")
    print(f"\nMean attention weight: {metadata['mean_attention_weight']:.3f}")
    print(f"Attention weight range: [{metadata['attention_weights'].min():.3f}, {metadata['attention_weights'].max():.3f}]")
    
    #Count parameters
    num_params = sum(p.numel() for p in ensemble.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    #Weight statistics
    weight_stats = ensemble.get_weight_statistics()
    print("\nWeight Statistics:")
    for key, value in weight_stats.items():
        print(f"  {key}: {value:.4f}")
