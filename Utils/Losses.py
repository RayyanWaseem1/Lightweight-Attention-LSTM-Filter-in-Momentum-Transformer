#Loss Functions for the Momentum Transformer

#Sharpe ratio loss (main)
#Sortino ratio loss
#Calmar ratio loss
#Maximum drawdown loss
#Combined losses with transactional costs

from __future__ import annotations

from typing import Dict, Optional, Tuple 

import torch
import torch.nn as nn 

from Models.config import ANNUALIZATION_SQRT, PERIODS_PER_YEAR

DEFAULT_EPS = 1e-8

class SharpeRatioLoss(nn.Module):
    """ Negative annualized Sharpe of `` positions * returns ``"""

    def __init__(
        self,
        annualization_factor: float = ANNUALIZATION_SQRT,
        epsilon: float = DEFAULT_EPS
    ):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon 

    def forward(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        pnl = positions.reshape(-1) * returns.reshape(-1)
        sharpe = pnl.mean() / (pnl.std() + self.epsilon) * self.annualization_factor
        return -sharpe 

class SortinoRatioLoss(nn.Module):
    """ Negative annualized Sortino; downside deviation about ``target_return``"""

    def __init__(
        self,
        annualization_factor: float = ANNUALIZATION_SQRT,
        epsilon: float = DEFAULT_EPS,
        target_return: float = 0.0,
    ):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon 
        self.target_return = target_return 

    def forward(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        pnl = positions.reshape(-1) * returns.reshape(-1)
        downside = torch.clamp(pnl - self.target_return, max = 0.0)
        downside_dev = torch.sqrt(torch.mean(downside**2)) + self.epsilon
        sortino = (pnl.mean() - self.target_return) / downside_dev 
        return -sortino * self.annualization_factor 

class _PathDependentLoss(nn.Module):
    """ Base for objectives that require time-ordered samples"""

    def __init__(self, sequential: bool = False):
        super().__init__()
        self.sequential = sequential 

    def _check(self):
        if not self.sequential:
            raise RuntimeError(
                f"{type(self).__name__} is path-dependent: cumsum/cummax over a "
                "shuffled batch is not a drawdown. Build the DataLoader with "
                "Utils.Training.SequentialBlockSampler (contiguous per-symbol "
                "blocks) and construct this loss with sequential = True."
            )

class MaximumDrawdownLoss(_PathDependentLoss):
    """ Maximum peak-to-trough decline of the cumulative PnL """

    def forward(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        self._check()
        pnl = positions.reshape(-1) * returns.reshape(-1)
        cumulative = torch.cumsum(pnl, dim = 0)
        running_max = torch.cummax(cumulative, dim = 0)[0]
        return torch.max(running_max - cumulative)

class NormalizedDrawdownLoss(_PathDependentLoss):
    """ Max drawdown scaled by the random-walk length ``std * sqrt(n)``
    
    Dimensionless, so it can be mixed with Sharpe/Sortino under weights that
    mean what they look like.
    """

    def __init__(self, sequential: bool = False, epsilon: float = DEFAULT_EPS):
        super().__init__(sequential)
        self.epsilon = epsilon 

    def forward(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        self._check()
        pnl = positions.reshape(-1) * returns.reshape(-1)
        cumulative = torch.cumsum(pnl, dim = 0)
        running_max = torch.cummax(cumulative, dim = 0)[0]
        max_dd = torch.max(running_max - cumulative)
        scale = pnl.std() * torch.sqrt(torch.tensor(float(pnl.numel()))) + self.epsilon
        return max_dd / scale 

class CalmarRatioLoss(_PathDependentLoss):
    """ Negative Calmar: annualized return over maximum drawdown """

    def __init__(
        self,
        annualization_factor: float = float(PERIODS_PER_YEAR),
        epsilon: float = DEFAULT_EPS,
        sequential: bool = False,
    ):
        super().__init__(sequential)
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon 

    def forward(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        self._check()
        pnl = positions.reshape(-1) * returns.reshape(-1)
        annual_return = pnl.mean() * self.annualization_factor

        cumulative = torch.cumsum(pnl, dim = 0)
        running_max = torch.cummax(cumulative, dim = 0)[0]
        max_dd = torch.max(running_max - cumulative) + self.epsilon
        return -(annual_return / max_dd)

class SharpeWithTurnoverPenalty(nn.Module):
    """ Sharpe on PnL net of turnover cost.
    
    ``net_pnl = position * return - transaction_cost * [position - prev_position]
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        annualization_factor: float = ANNUALIZATION_SQRT,
        epsilon: float = DEFAULT_EPS,
    ):
        super().__init__()
        self.transaction_cost = transaction_cost
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon 

    def forward(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor,
        previous_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positions = positions.reshape(-1)
        returns = returns.reshape(-1)

        gross_pnl = positions * returns 
        if previous_positions is None:
            turnover = positions.abs() # entering from flat
        else: 
            turnover = (positions - previous_positions.reshape(-1)).abs()

        net_pnl = gross_pnl - self.transaction_cost * turnover
        # NOTE: '+' epsilon. the previous '*' inflated this by 1e8
        sharpe = net_pnl.mean() / (net_pnl.std() + self.epsilon)
        return -sharpe * self.annualization_factor

class SoftDirectionalLoss(nn.Module):
    """ Differnetiable stand-in for directional accuracy.
    
    ``-mean(tanh(k * position) * sign(return))`` -- rewards agreeing with the 
    realized sign, with a gradient that actually exists
    """

    def __init__(self, sharpness: float = 10.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        soft_sign = torch.tanh(self.sharpness * positions.reshape(-1))
        return -(soft_sign * torch.sign(returns.reshape(-1))).mean() 

class InformationRatioLoss(nn.Module):
    """ Negative information ratio against a benchmark return series """

    def __init__(
        self,
        annualization_factor: float = ANNUALIZATION_SQRT,
        epsilon: float = DEFAULT_EPS
    ):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon 

    def forward(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor,
        benchmark_returns: torch.Tensor,
    ) -> torch.Tensor:
        excess = positions.reshape(-1) * returns.reshape(-1) - benchmark_returns.reshape(-1)
        ir = excess.mean() / (excess.std() + self.epsilon)
        return -ir * self.annualization_factor

class CombinedLoss(nn.Module):
    """ Weighted blend of Sharpe, Sortino, normalized drawdown and turnover cost.
    
    All four components are dimensionless and O(1), so the weights are 
    interpretable. Set ``sequential = True`` (and use a sequential sampler) to 
    enable the drawdown term; otherwise its weight is redistributed and the
    drawdown component is reported as ``0.0``.
    """

    def __init__(
        self,
        sharpe_weight: float = 0.7,
        sortino_weight: float = 0.2,
        drawdown_weight: float = 0.1,
        turnover_weight: float = 0.0,
        transaction_cost: float = 0.001,
        sequential: bool = False,
        annualization_factor: float = ANNUALIZATION_SQRT,
    ):
        super().__init__()
        self.sequential = sequential
        self.turnover_weight = turnover_weight

        if not sequential and drawdown_weight > 0:
            # Redistribute rather than silently computing a meaningless term.
            total = sharpe_weight + sortino_weight
            sharpe_weight += drawdown_weight * sharpe_weight / total
            sortino_weight += drawdown_weight * sortino_weight / total 
            drawdown_weight = 0.0

        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight 
        self.drawdown_weight = drawdown_weight 

        # Components are kept unannualized relative to one another by sharing
        # one factor, so the weights compare like with like 
        self.sharpe_loss = SharpeRatioLoss(annualization_factor = annualization_factor)
        self.sortino_loss = SortinoRatioLoss(annualization_factor = annualization_factor)
        self.drawdown_loss = NormalizedDrawdownLoss(sequential=sequential)
        self.turnover_penalty = SharpeWithTurnoverPenalty(
            transaction_cost = transaction_cost, annualization_factor = annualization_factor
        )

    def forward(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor,
        previous_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        sharpe = self.sharpe_loss(positions, returns)
        sortino = self.sortino_loss(positions, returns)

        loss = self.sharpe_weight * sharpe + self.sortino_weight * sortino 
        components: Dict[str, float] = {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
        }

        if self.drawdown_weight > 0:
            drawdown = self.drawdown_loss(positions, returns)
            loss = loss + self.drawdown_weight * drawdown 
            components["drawdown"] = float(drawdown)
        else:
            components["drawdown"] = 0.0

        # previous_positions is now actually used
        if self.turnover_weight > 0:
            turnover_term = self.turnover_penalty(positions, returns, previous_positions)
            loss = loss + self.turnover_weight * turnover_term
            components["turnover"] = float(turnover_term)

        components["total"] = float(loss)
        return loss, components

LOSS_REGISTRY = {
    "sharpe": SharpeRatioLoss,
    "sortino": SortinoRatioLoss,
    "calmar": CalmarRatioLoss,
    "drawdown": MaximumDrawdownLoss,
    "normalized_drawdown": NormalizedDrawdownLoss,
    "sharpe_turnover": SharpeWithTurnoverPenalty,
    "combined": CombinedLoss,
    "soft_directional": SoftDirectionalLoss,
    "information_ratio": InformationRatioLoss,
}

def get_loss_function(loss_type: str = "sharpe", **kwargs) -> nn.Module:
    """ Build a loss by name.
    
    ``directional`` is deliberately absent: it had no gradient and is now
    ``Utils.Metrics.directional_accuracy``
    """

    if loss_type == "directional":
        raise ValueError(
            "DirectionalAccuracyLoss has zero gradient and is not trainable. "
            "Use 'soft_directional' for a differentiable surrogate, or "
            "Utils.Metrics.directional_accuracy to report it as a metric"
        )
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Available: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[loss_type](**kwargs)
