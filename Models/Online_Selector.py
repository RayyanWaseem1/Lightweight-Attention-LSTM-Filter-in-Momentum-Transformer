"""
Docstring for Models.Online_Selector
Online Performance Monitoring for Alpaca deployment
Meta layer that monitors both models' performance and selects the better one
Uses actual realized returns to make the adaptive/dynamic decisions
"""

import torch
import torch.nn as nn
import numpy as np 
from typing import Dict, Tuple, Optional, List
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from .config import ANNUALIZATION_SQRT
from .Momentum_transformer import MomentumTransformerSimple, MomentumTransformerDualPath
from .Ensemble_model import EnsembleMomentumTransformer

@dataclass
class PerformanceMetrics:
    #container for model performance metrics 
    sharpe_ratio: float
    mean_return: float
    std_return: float
    max_drawdown: float 
    win_rate: float 
    num_observations: int
    last_updated: datetime 

class RollingPerformanceTracker:
    """Rolling performance of one model's realised P&L.

    Renamed from ``PerformanceTracker``: ``Utils.Training`` has a class of that
    name tracking per-epoch training history, which is a different thing.

    Two bugs fixed here:

    * ``update`` accumulated the raw **asset** return rather than
      ``position * return``, so ``max_drawdown`` described the instrument, not
      the model being evaluated.
    * ``cumulative_returns`` was an unbounded list while ``returns`` and
      ``positions`` were ``deque(maxlen=lookback_window)``, so drawdown was
      measured over all history while Sharpe was measured over the last 50
      bars.  Both now share one window.
    """

    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
        self.returns = deque(maxlen=lookback_window)
        self.positions = deque(maxlen=lookback_window)
        self.pnl = deque(maxlen=lookback_window)

    def update(self, position: float, realized_return: float) -> None:
        """Record one bar: the position taken and the return it earned."""
        self.positions.append(position)
        self.returns.append(realized_return)
        # P&L, not the asset return.
        self.pnl.append(position * realized_return)

    def get_metrics(self) -> Optional[PerformanceMetrics]:
        if len(self.pnl) < 10:
            return None

        pnl = np.asarray(self.pnl, dtype=float)

        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl))
        sharpe = 0.0 if std_pnl < 1e-8 else mean_pnl / std_pnl * ANNUALIZATION_SQRT

        win_rate = float(np.mean(pnl > 0))

        # Drawdown over the same window the other statistics use.
        equity = np.cumprod(1.0 + pnl)
        running_max = np.maximum.accumulate(equity)
        max_drawdown = float(np.min((equity - running_max) / running_max))

        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            mean_return=mean_pnl,
            std_return=std_pnl,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_observations=len(self.pnl),
            last_updated=datetime.now(),
        )


# Backwards-compatible alias for callers that imported the old name.
PerformanceTracker = RollingPerformanceTracker
    
class OnlineModelSelector:
    #Monitoring both model's performance and dynamically selecting the better one 
    #Acts like a meta layer on top of any models 

    def __init__(self,
                 vanilla_model: nn.Module,
                 attention_model: nn.Module,
                 lookback_window: int = 50,
                 switch_threshold: float = 0.2,
                 min_observations: int = 20,
                 switch_cooldown_period: int = 10):
        
        #Vanilla model: Vanilla Momentum Transformer
        #Attention_model: Attention-enhanced Momentum Transformer
        #Lookback_window: Rolling window for performance calculation
        #Switch_threshold: Sharpe difference required to trigger switch
        #Min_observations: Minimum observations before considering switch
        #Switch_cooldown_period: Days to wait after switch before allowing another 

        self.vanilla_model = vanilla_model
        self.attention_model = attention_model

        self.lookback_window = lookback_window
        self.switch_threshold = switch_threshold
        self.min_observations = min_observations
        self.switch_cooldown_period = switch_cooldown_period

        #Performance tracking 
        self.vanilla_tracker = RollingPerformanceTracker(lookback_window)
        self.attention_tracker = RollingPerformanceTracker(lookback_window)

        #State
        self.current_model = "vanilla" #Starting off conservative and according to the paper
        self.switch_cooldown = 0
        self.last_vanilla_pred = None 
        self.last_attention_pred = None 

        #History
        self.switch_history = []
        self.performance_history = []

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        #Generating prediction using the currently selected model 

        #x: [batch, seq_len, input_dim]

        #positions: [batch]
        #metadata: dict with model information and both predictions

        #Get predictions from both models (for tracking)
        with torch.no_grad():
            self.vanilla_model.eval()
            self.attention_model.eval() 

            # Every model now returns a uniform (position, info) tuple.
            vanilla_pred, _ = self.vanilla_model(x)
            attention_pred, _ = self.attention_model(x)

        #Storing for later performance update
        self.last_vanilla_pred = vanilla_pred.detach().cpu() 
        self.last_attention_pred = attention_pred.detach().cpu() 

        # Read the CURRENT selection. Re-selecting here made predict()
        # non-idempotent: calling it twice on the same input could return
        # different answers. Selection now happens in update_performance(),
        # when new realised P&L actually arrives.
        model_name = self.current_model
        selection_metadata = self._selection_metadata()

        if model_name == "vanilla":
            final_pred = vanilla_pred
        else:
            final_pred = attention_pred

        metadata = {
            "model_used" : model_name,
            "vanilla_predictions": vanilla_pred.detach().cpu(),
            "attention_predictions": attention_pred.detach().cpu(),
            **selection_metadata
        }

        return final_pred, metadata
    
    def update_performance(self, realized_returns: np.ndarray) -> Dict:
        #Updating performance trackers with the realized returns
        #Call this after each trading period with the actual PnL 

        #realized_returns: [batch] array of realized returns 

        #Dict with updated performance metrics

        if self.last_vanilla_pred is None or self.last_attention_pred is None:
            return {}
        
        #NumPy conversion 
        vanilla_positions = self.last_vanilla_pred.numpy()
        attention_positions = self.last_attention_pred.numpy() 

        #Update trackers for each sample in the batch 
        for i in range(len(realized_returns)):
            self.vanilla_tracker.update(
                float(vanilla_positions[i]),
                float(realized_returns[i])
            )
            self.attention_tracker.update(
                float(attention_positions[i]),
                float(realized_returns[i])
            )

        #Decrementing the cooldown period
        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1

        # Re-evaluate the selection now that new realised P&L has arrived.
        # This is the only place self.current_model changes.
        self._select_model()

        #Getting current metrics
        vanilla_metrics = self.vanilla_tracker.get_metrics()
        attention_metrics = self.attention_tracker.get_metrics()

        #store history
        self.performance_history.append({
            "timestamp": datetime.now(),
            "vanilla_metrics": vanilla_metrics,
            "attention_metrics": attention_metrics,
            "current_model": self.current_model
        })

        metadata = {
            "vanilla_sharpe": vanilla_metrics.sharpe_ratio if vanilla_metrics else None,
            "attention_sharpe": attention_metrics.sharpe_ratio if attention_metrics else None,
            "vanilla_return": vanilla_metrics.mean_return if vanilla_metrics else None,
            "attention_return": attention_metrics.mean_return if attention_metrics else None,
            "current_model": self.current_model,
            "switch_cooldown": self.switch_cooldown
        }

        return metadata
    
    def _selection_metadata(self) -> Dict:
        """Read-only view of the current selection state.

        Used by ``predict`` so that generating a prediction never mutates the
        selector.
        """
        vanilla_metrics = self.vanilla_tracker.get_metrics()
        attention_metrics = self.attention_tracker.get_metrics()
        return {
            "vanilla_sharpe": vanilla_metrics.sharpe_ratio if vanilla_metrics else None,
            "attention_sharpe": attention_metrics.sharpe_ratio if attention_metrics else None,
            "current_model": self.current_model,
            "switch_cooldown": self.switch_cooldown,
        }

    def _select_model(self) -> Tuple[str, Dict]:
        #Selecting the model based on the recent performance

        #model_name: vanilla or attention
        #metadata: dict with reasoning for selection

        vanilla_metrics = self.vanilla_tracker.get_metrics()
        attention_metrics = self.attention_tracker.get_metrics() 

        metadata = {
            "vanilla_sharpe": vanilla_metrics.sharpe_ratio if vanilla_metrics else None,
            "attention_sharpe": attention_metrics.sharpe_ratio if attention_metrics else None,
            "current_model": self.current_model,
            "switch_cooldown": self.switch_cooldown,
            "n_observations": len(self.vanilla_tracker.returns)
        }

        #Need enough data and an expired cooldown to consider switching again 
        if (vanilla_metrics is None or
            attention_metrics is None or 
            vanilla_metrics.num_observations < self.min_observations or 
            self.switch_cooldown > 0):
            return self.current_model, metadata 
        
        #Calculating the differrence in performance
        sharpe_diff = attention_metrics.sharpe_ratio - vanilla_metrics.sharpe_ratio 
        metadata["sharpe_difference"] = sharpe_diff

        #switch logic with hysteresis
        switched = False
        reason = None 

        if self.current_model == "vanilla":
            #Switching to attention if significantly better
            if sharpe_diff > self.switch_threshold:
                self.current_model = "attention"
                switched = True 
                reason = f"Attention model is outperforming the vanilla model by {sharpe_diff:.3f} Sharpe"
        else: #current model == "attention"
            #switching back to vanilla if the attention model is underperforming
            if sharpe_diff < -self.switch_threshold:
                self.current_model = "vanilla"
                switched = True
                reason = f"Vanilla model is outperforming the attention model by {-sharpe_diff:.3f} Sharpe"

        if switched:
            self.switch_cooldown = self.switch_cooldown_period
            self.switch_history.append({
                "timestamp": datetime.now(),
                "from_model": "attention" if self.current_model == "vanilla" else "vanilla",
                "to_model": self.current_model,
                "sharpe_diff": sharpe_diff,
                "reason": reason
            })
            metadata["switched"] = True
            metadata["switch_reason"] = reason 

        return self.current_model, metadata 
    
    def get_performance_summary(self) -> Dict:
        #getting the comprehensive performance summary 

        #Dict with detailed performance statistics 

        vanilla_metrics = self.vanilla_tracker.get_metrics()
        attention_metrics = self.attention_tracker.get_metrics() 

        summary = {
            "current_model": self.current_model,
            "num_switches": len(self.switch_history),
            "switch_cooldown": self.switch_cooldown,
        }

        if vanilla_metrics: 
            summary["vanilla"] = {
                "sharpe_ratio": vanilla_metrics.sharpe_ratio,
                "mean_return": vanilla_metrics.mean_return,
                "std_return": vanilla_metrics.std_return,
                "max_drawdown": vanilla_metrics.max_drawdown,
                "win_rate": vanilla_metrics.win_rate,
                "num_observations": vanilla_metrics.num_observations
            }
        if attention_metrics:
            summary["attention"] = {
                "sharpe_ratio": attention_metrics.sharpe_ratio,
                "mean_return": attention_metrics.mean_return,
                "std_return": attention_metrics.std_return,
                "max_drawdown": attention_metrics.max_drawdown,
                "win_rate": attention_metrics.win_rate,
                "num_observations": attention_metrics.num_observations
            }
        if vanilla_metrics and attention_metrics:
            summary["performance_diff"] = {
                "sharpe_diff": attention_metrics.sharpe_ratio - vanilla_metrics.sharpe_ratio,
                "return_diff": attention_metrics.mean_return - vanilla_metrics.mean_return
            }

        #Recent switches
        if self.switch_history:
            summary["recent_switches"] = self.switch_history[-5:] #last 5 switches
        return summary 
    
class HybridSelector(nn.Module):
    #Hybrid model that combines the ensemble strategy with the online monitoring
    #Essentially uses the ensemble model but validates with the online performance monitoring
    def __init__(self,
                 ensemble_model: EnsembleMomentumTransformer,
                 vanilla_model: nn.Module,
                 attention_model: nn.Module,
                 lookback_window: int = 50,
                 switch_threshold: float = 0.3,
                 min_observations: int = 30):
        super().__init__() 

        self.ensemble_model = ensemble_model

        #Online selector as a meta layer
        self.online_selector = OnlineModelSelector(
            vanilla_model=vanilla_model,
            attention_model=attention_model,
            lookback_window=lookback_window,
            switch_threshold = switch_threshold,
            min_observations=min_observations,
            switch_cooldown_period=15 #Longer cooldown for override
        )

        self.use_ensemble = True #Defaulting to ensemble 

    def forward(self, x: torch.Tensor, return_all: bool = False):
        #Forward pass with the hybrid selection 

        #x: [batch, seq_len, input_dim]
        #return_all: whether to return all of the predictions

        #positions: [batch]
        #metadata: dict with all of the model info 

        #Getting ensemble predictions
        ensemble_pred, ensemble_meta = self.ensemble_model(x, return_components=True)

        #Getting online selector's choice (for the validation)
        _, online_meta = self.online_selector.predict(x)

        #Using ensemble by default, but tracking online selector 
        final_pred = ensemble_pred 

        metadata = {
            "ensemble_prediction": ensemble_pred.detach(),
            "ensemble_mean_weight": ensemble_meta["mean_attention_weight"],
            "online_recommended_model": online_meta["model_used"],
            "online_vanilla_sharpe": online_meta.get("vanilla_sharpe"),
            "online_attention_sharpe": online_meta.get("attention_sharpe"),
            "using_ensemble": True
        }

        if return_all:
            metadata.update(ensemble_meta)
            metadata.update(online_meta)

        return final_pred, metadata
    
    def update_performance(self, realized_returns: np.ndarray):
        #Updating online selector with realized returns 
        return self.online_selector.update_performance(realized_returns)
    def get_full_summary(self) -> Dict:
        #Comprehensive summary of both strategies
        return {
            "ensemble_weights": self.ensemble_model.get_weight_statistics(),
            "online_performance": self.online_selector.get_performance_summary()
        }
    

if __name__ == "__main__":
    #testing online selector 

    from Models.config import get_production_config

    config = get_production_config()
    config.model.input_dim = 32

    #creating the models
    vanilla_model = MomentumTransformerSimple(
        input_dim = config.model.input_dim,
        hidden_dim=config.model.hidden_dim
    )

    attention_model = MomentumTransformerDualPath(
        input_dim = config.model.input_dim,
        hidden_dim=config.model.hidden_dim
    )

    #Creating the online selector
    selector = OnlineModelSelector(
        vanilla_model=vanilla_model,
        attention_model=attention_model,
        lookback_window=50,
        switch_threshold=0.2
    )

    print("Testing the online model selector \n")

    #Simulate the trading loop
    for day in range(60):
        x = torch.randn(8,252, config.model.input_dim)

        #get prediction
        positions, metadata = selector.predict(x)

        #simulate realized returns (random for testing)
        realized_returns = np.random.randn(8) * 0.01

        #Updating the performance
        perf_meta = selector.update_performance(realized_returns)

        if day % 10 == 0:
            print(f"\n Day {day}:")
            print(f" Current Model: {metadata["model_used"]}")
            if perf_meta.get("vanilla_sharpe"):
                print(f" Vanilla Sharpe: {perf_meta["vanilla_sharpe"]:.3f}")
                print(f" Attention Sharpe: {perf_meta["attention_sharpe"]:.3f}")

    #final summary 
    print("\n" + "="*50)
    print("Final performance summary")
    print("="*50)

    summary = selector.get_performance_summary()

    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k,v in value.items():
                print(f" {k}: {v}")
        else:
            print(f"{key}: {value}")