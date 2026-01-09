#Online Performance Monitoring
#Meta layer that monitors both models' performance and selects the better one
#Uses actual realized returns to make the adaptive/dynamic decisions

import torch
import torch.nn as nn
import numpy as np 
from typing import Dict, Tuple, Optional, List
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from Models.Momentum_transformer import MomentumTransformerSimple, MomentumTransformerDualPath
from Models.Ensemble_model import EnsembleMomentumTransformer

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

class PerformanceTracker:
    #Tracking and computing performance metrics for a model 

    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
        self.returns = deque(maxlen = lookback_window)
        self.positions = deque(maxlen = lookback_window)
        self.cumulative_returns = []

    def update(self, position: float, realized_return: float) -> None:
        #Update with new positions and realized returns 

        #position: predicted position
        #realized_return: actual realized return 

        self.positions.append(position)
        self.returns.append(realized_return)

        #Tracking cumulative for drawdown calculation
        if self.cumulative_returns:
            self.cumulative_returns.append(
                self.cumulative_returns[-1] + realized_return
            )
        else:
            self.cumulative_returns.append(realized_return)

    def get_metrics(self) -> Optional[PerformanceMetrics]:
        #Computing current performance metrics

        #Returns: PerformanceMetrics or None if the data is insufficient    
        if len(self.returns) < 10:
            return None 
        
        returns_array = np.array(self.returns)
        positions_array = np.array(self.positions)

        #PnL = Position * return
        pnl = positions_array * returns_array

        #Annualized Sharpe Ratio 
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl) 

        if std_pnl < 1e-8:
            sharpe = 0.0
        else:
            sharpe = mean_pnl/std_pnl * np.sqrt(252)

        #Other metrics
        mean_return = np.mean(pnl)
        std_return = std_pnl 

        #Win rate
        win_rate = np.mean(pnl > 0) if len(pnl) > 0 else 0.0

        #Max drawdown
        if self.cumulative_returns:
            cumulative = np.array(self.cumulative_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0.0 

        return PerformanceMetrics(
            sharpe_ratio = sharpe,
            mean_return = mean_return,
            std_return = std_return,
            max_drawdown = max_drawdown,
            win_rate = win_rate,
            num_observations = len(self.returns),
            last_updated = datetime.now()
        )
    
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
        self.vanilla_tracker = PerformanceTracker(lookback_window)
        self.attention_tracker = PerformanceTracker(lookback_window)

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

            vanilla_pred = self.vanilla_model(x)

            #handling different return signatures
            if isinstance(self.attention_model, MomentumTransformerDualPath):
                attention_pred, _ = self.attention_model(x, return_attention = False)
            else:
                attention_pred = self.attention_model(x)

        #Storing for later performance update
        self.last_vanilla_pred = vanilla_pred.detach().cpu() 
        self.last_attention_pred = attention_pred.detach().cpu() 

        #Select which model
        model_name, selection_metadata = self._select_model() 

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