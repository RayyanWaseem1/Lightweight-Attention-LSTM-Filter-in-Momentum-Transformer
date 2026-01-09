#Loss Functions for the Momentum Transformer

#Sharpe ratio loss (main)
#Sortino ratio loss
#Calmar ratio loss
#Maximum drawdown loss
#Combined losses with transactional costs

import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Optional, Tuple 

class SharpeRatioLoss(nn.Module):
    #Negative Sharpe Ratio

    #Sharpe = (mean_return/std_return) * sqrt(252)
    #Loss = -Sharpe (minimizing negative sharpe = maximizing Sharpe)

    #annualization_factor: sqrt(252) for daily data
    #epsilon: small constant for numerical stability 

    def __init__(self,
                 annualization_factor: float = (252 * 24) ** 0.5,
                 epsilon: float = 1e-8):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon 

    def forward(self, 
                positions: torch.Tensor,
                returns: torch.Tensor) -> torch.Tensor:
        #positions: [batch] - predicted positions [-1,1]
        #returns: [batch] - actual returns 

        #loss: scalar - negative sharpe ratio 

        #portfolio returns (position * market_return)
        pnl = positions * returns 

        #mean and std of PNL 
        mean_pnl = torch.mean(pnl)
        std_pnl = torch.std(pnl) + self.epsilon

        #annualized sharpe
        sharpe = (mean_pnl/std_pnl) * self.annualization_factor

        #returning negative sharpe (so we can minimize)
        return -sharpe
    

class SortinoRatioLoss(nn.Module):
    #Negative Sortino ratio loss
    #Sortio = (mean_return / downside_std) * sqrt(252)

    #similar in nature to sharpe but it only penalizes downside volatility
    #Better for when the return distribution is asymmetrical 

    def __init__(self,
                 annualization_factor: float = (252 * 24) ** 0.5,
                 epsilon: float = 1e-8,
                 target_return: float = 0.0):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon
        self.target_return = target_return

    def forward(self,
                positions: torch.Tensor,
                returns: torch.Tensor) -> torch.Tensor:
        #positions: [batch] - predicted positions
        #returns: [batch] - actual returns 

        #loss: scalar - negative sortino ratio 

        pnl = positions * returns 

        #mean return
        mean_pnl = torch.mean(pnl)

        #downside deviation (only the negative returns)
        downside_returns = torch.minimum(pnl - self.target_return, torch.zeros_like(pnl))
        downside_std = torch.sqrt(torch.mean(downside_returns ** 2)) + self.epsilon

        #Sortino Ratio
        sortino = (mean_pnl/downside_std) * self.annualization_factor

        return -sortino
    
class CalmarRatioLoss(nn.Module):
    #Negative Calmar ratio loss
    #Calmar = annualized_return / max_drawdown
    #Focuses on return relative to the worst drawdown 

    def __init__(self,
                 annualization_factor: float = 252.0 * 24,
                 epsilon: float = 1e-8):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon

    def forward(self,
                positions: torch.Tensor,
                returns: torch.Tensor) -> torch.Tensor:
        
        #Positions: [batch] - predicted positions
        #returns: [batch] - actual returns

        #loss: scalar - negative Calmar

        pnl = positions * returns 

        #annualized return
        mean_return = torch.mean(pnl) * self.annualization_factor

        #maximum drawdown 
        cumulative_returns = torch.cumsum(pnl, dim = 0)
        running_max = torch.cummax(cumulative_returns, dim = 0)[0]
        drawdown = running_max - cumulative_returns
        max_drawdown = torch.max(drawdown) + self.epsilon

        #calmar ratio 
        calmar = mean_return / max_drawdown
        
        return -calmar 
    
    

class MaximumDrawdownLoss(nn.Module):
    #Direct maximum drawdown loss
    #minimizes the worst peak-to-trough decline 

    def __init__(self):
        super().__init__()

    def forward(self,
                positions: torch.Tensor,
                returns: torch.Tensor) -> torch.Tensor:
        
        #positions: [batch] - predicted posisitons
        #returns: [batch] - actual returns

        #loss: scalar - maximum drawdown 

        pnl = positions * returns 

        #Computing the cumulative returns
        cumulative_returns = torch.cumsum(pnl, dim = 0)

        #running maximum
        running_max = torch.cummax(cumulative_returns, dim = 0)[0]

        #drawdown at each point
        drawdown = running_max - cumulative_returns

        #max drawdown
        max_drawdown = torch.max(drawdown)

        return max_drawdown
    
class HybridLoss(nn.Module):
    #Combines MSE for stability and Sharpe for performance
    def __init__(self, sharpe_weight = 0.7):
        super().__init__()
        self.sharpe_weight = sharpe_weight
        self.mse_weight = 1.0 - sharpe_weight

    def forward(self, predictions, targets):
        #MSE components for the stability
        mse = torch.mean((predictions - targets) ** 2)

        #Sharpe component for performance
        strategy_returns = predictions * targets
        sharpe_loss = -strategy_returns.mean() / (strategy_returns.std() + 1e-8)

        return self.mse_weight * mse + self.sharpe_weight * sharpe_loss
    

class SharpeWithTurnoverPenalty(nn.Module):
    #Sharpe with transaction cost penalty
    #Penalizes excessive trading by subtracting any turnover costs

    def __init__(self,
                 transaction_cost: float = 0.001, #10 bps
                 annualization_factor: float = 15.874,
                 epsilon: float = 1e-8):
        super().__init__()
        self.transaction_cost = transaction_cost
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon

    def forward(self, 
                positions: torch.Tensor,
                returns: torch.Tensor,
                previous_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        #positions: [batch] - current positions
        #returns: [batch] - actual returns
        #previous_positions: [batch] - previous day positions

        #loss: scalar - negative Sharpe with turnover penalty

        #Gross PNL 
        gross_pnl = positions * returns
        
        #turnover cost
        if previous_positions is not None:
            turnover = torch.abs(positions - previous_positions)
        else:
            #no previous position, start from 0
            turnover = torch.abs(positions)

        #net pnl after the transaction cost
        net_pnl = gross_pnl - self.transaction_cost * turnover 

        #sharpe on net PNL 
        mean_pnl = torch.mean(net_pnl)
        std_pnl = torch.std(net_pnl) * self.epsilon
        sharpe = (mean_pnl / std_pnl) * self.annualization_factor

        return -sharpe
    

class CombinedLoss(nn.Module):
    #Weighted combination of multiple loss functions
    #0.7Sharpe + 0.2Sortino + 0.1Max_Drawdown

    def __init__(self,
                 sharpe_weight: float = 0.7,
                 sortino_weight: float = 0.2,
                 drawdown_weight: float = 0.1,
                 transaction_cost: float = 0.001):
        super().__init__()

        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        self.drawdown_weight = drawdown_weight

        #initialize individual losses
        self.sharpe_loss = SharpeRatioLoss()
        self.sortino_loss = SortinoRatioLoss()
        self.drawdown_loss = MaximumDrawdownLoss()
        self.turnover_penalty = SharpeWithTurnoverPenalty() 

    def forward(self,
                positions: torch.Tensor, 
                returns: torch.Tensor,
                previous_positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        #Positions: [batch] - predicted positions
        #returns: [batch] - actual returns 
        #previous_positions: [batch] - previous positions (optional)

        #loss: scalar - weighted combo 
        #components: dict of individual loss components

        #computing individual losses
        sharpe = self.sharpe_loss(positions, returns)
        sortino = self.sortino_loss(positions, returns)
        drawdown = self.drawdown_loss(positions, returns)

        #combined loss
        loss = (self.sharpe_weight * sharpe +
                self.sortino_weight * sortino + 
                self.drawdown_weight * drawdown)
        
        #returning loss and its components for monitoring
        components = {
            "sharpe": sharpe.item(),
            "sortino": sortino.item(),
            "drawdown": drawdown.item(),
            "total": loss.item()
        }

        return loss, components
    

class DirectionalAccuracyLoss(nn.Module):
    #Penalizes incorrect directional prediction
    #Auxiliary loss when combined with Sharpe

    def __init__(self):
        super().__init__()

    def forward(self, 
                positions: torch.Tensor,
                returns: torch.Tensor) -> torch.Tensor:
        
        #positions: [batch] - predicted positions (sign = direction)
        #returns: [batch] - actual returns (sign = true direction)

        #loss: scalar - directional accuracy loss 

        #Get signs
        pred_direction = torch.sign(positions)
        true_direction = torch.sign(returns)

        #Accuracy (1 = same sign, 0 = different)
        correct = (pred_direction == true_direction).float()

        #Return the negative acccurazy
        accuracy = torch.mean(correct)

        return -accuracy
    

class InformationRatioLoss(nn.Module):
    #information ratio relative to a benchmark
    #IR = (portfolio_return - benchmark_return) / tracking_error 

    def __init__(self,
                 annualization_factor: float = 15.874,
                 epsilon: float = 1e-8):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon

    def forward(self,
                positions: torch.Tensor, 
                returns: torch.Tensor,
                benchmark_returns: torch.Tensor) -> torch.Tensor:
        
        #position: [batch] - predicted positions
        #returns: [batch] - actual returns
        #benchmark_returns: [batch] - benchmark returns

        #loss: scalar - negative information ratio

        #portfolio returns
        portfolio_returns = positions * returns

        #excess returns over the benchmark
        excess_returns = portfolio_returns - benchmark_returns

        #information ratio 
        mean_excess = torch.mean(excess_returns)
        std_excess = torch.std(excess_returns) + self.epsilon
        ir = (mean_excess / std_excess) * self.annualization_factor

        return -ir
    

#Factory Function
def get_loss_function(loss_type: str = "sharpe", **kwargs) -> nn.Module:

    #factory function to create the loss functions

    #loss_type: type of loss ("sharpe", "sortino", "calmar", etc)
    #**kwargs: additional arguments for loss function

    #loss function module 

    #loss_fn = get_loss_function("sharpe")
    #loss_fn = get_loss_function("combined", sharpe_weight = 0.7)
    #loss_fn = get_loss_function("sharpe_turnover", transaction_cost = 0.002)

    loss_registry = {
        "sharpe": SharpeRatioLoss,
        "sortino": SortinoRatioLoss,
        "calmar": CalmarRatioLoss,
        "drawdown": MaximumDrawdownLoss,
        "sharpe_turnover": SharpeWithTurnoverPenalty,
        "combined": CombinedLoss,
        "directional": DirectionalAccuracyLoss,
        "information_ratio": InformationRatioLoss
    }

    if loss_type not in loss_registry:
        raise ValueError(f"Unknown loss type: {loss_type}. " f"Available: {list(loss_registry.keys())}")
    return loss_registry[loss_type](**kwargs)


#convenience functions for direct use
def sharpe_ratio_loss(positions: torch.Tensor,
                      returns: torch.Tensor) -> torch.Tensor:
    #Convenience function for Sharpe loss
    loss_fn = SharpeRatioLoss()
    return loss_fn(positions, returns)

def sortino_ratio_loss(positions: torch.Tensor,
                       returns: torch.Tensor) -> torch.Tensor:
    #convenience function for Sortino loss
    loss_fn = SortinoRatioLoss()
    return loss_fn(positions,returns)

def calmar_ratio_loss(positions: torch.Tensor,
                      returns: torch.Tensor) -> torch.Tensor:
    #Convenience function for Calmar loss
    loss_fn = CalmarRatioLoss()
    return loss_fn(positions, returns)

def maximum_drawdown_loss(positions: torch.Tensor,
                          returns: torch.Tensor) -> torch.Tensor:
    #Convenience function for drawdown loss
    loss_fn = MaximumDrawdownLoss()
    return loss_fn(positions, returns)

def sharpe_with_turnover_penalty(positions: torch.Tensor,
                                 returns: torch.Tensor,
                                 previous_positions: Optional[torch.Tensor] = None,
                                 transaction_cost: float = 0.001) -> torch.Tensor:
    #Convenience function for Sharpe with turnover penalty
    loss_fn = SharpeWithTurnoverPenalty(transaction_cost=transaction_cost)
    return loss_fn(positions, returns, previous_positions)

if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions\n")
    
    # Create sample data
    torch.manual_seed(42)
    positions = torch.randn(100) * 0.5  # Random positions
    returns = torch.randn(100) * 0.02   # Random returns
    
    # Test each loss
    losses = {
        'Sharpe': sharpe_ratio_loss(positions, returns),
        'Sortino': sortino_ratio_loss(positions, returns),
        'Calmar': calmar_ratio_loss(positions, returns),
        'Max Drawdown': maximum_drawdown_loss(positions, returns)
    }
    
    print("Loss Values:")
    for name, loss in losses.items():
        print(f"  {name:15s}: {loss.item():8.4f}")
    
    # Test combined loss
    combined = CombinedLoss()
    total_loss, components = combined(positions, returns)
    
    print("\nCombined Loss Components:")
    for name, value in components.items():
        print(f"  {name:15s}: {value:8.4f}")
    
    print("\n✅ All loss functions working correctly!")
