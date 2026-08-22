""" Training loop, early stopping, samplers and evaluations

Fixes relative to the previous version 

* **Early stopping in ``max`` mode was inverted**
    `` score > (best - min_delta) `` makes improvement *easier* to declare as 
    `` min_delta`` gros, defeating the purpose. It is now ``best + min_delta``
* **Early stopping was called with a negated loss**
    `` early_stopping(-val_loss, epoch)`` under ``mod = 'min'`` means "stop when
    ``-val_loss`` stops decreasing", i.e. stop when validation loss stops 
    *increasings* -- training halted precisel when the model started improving. 
    Selection is now on ``val_sharpe`` with ``mode = 'max'``, matching the
    best-checkpoint criterion, which previously disagreed with it
* **Sharpe is computed at the epoch level**, as the README always claimed.
    Predictions and returns are accumulated across, ``accumulation_steps``
    mini-batches (or the whole epoch) and a single Sharpe is backpropagated,
    instead of one noisy per-batch ratio whose mean across batches is not the 
    Sharpe of anything 
* **Turnover is computed on time-ordered positions** ``np.diff`` over the 
    concatenation of shuffled batches is not turnover.
* **A dead branch is gone** -- ``std_pnl = std + 1e-8`` followed by 
    ``if std_pnl < 1e-8`` was unreachable.
* ``SequentialBlockSampler`` provides the contiguous per-symbol batches that 
    path-dependent losses require
"""

from __future__ import annotations

import time 
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple 

import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Sampler 

from Models.config import ANNUALIZATION_SQRT, PERIODS_PER_YEAR
from Utils import Metrics 

class _Scheduler(Protocol):
    def step(self, *args, **kwargs) -> None:
        ...

@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float 
    val_loss: float 
    train_sharpe: float 
    val_sharpe: float 
    learning_rate: float 
    time_elapsed: float

class EarlyStopping:
    """ Stop when the monitored score stops improving 
    
    ``mode = 'max'`` is the default because every selection criterion in 
    this repo is a Sharpe ratio
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience 
        self.min_delta = min_delta 
        self.mode = mode 
        self.counter = 0 
        self.best_score: Optional[float] = None 
        self.best_epoch = 0 
        self.early_stop = False 

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None or not np.isfinite(self.best_score):
            self.best_score = score 
            self.best_epoch = epoch 
            return False 

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            # '+' min_delta: a larger tolerance must make improvement HARDER
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score 
            self.best_epoch = epoch 
            self.counter = 0 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True 
                return True 
        return False 

class GradientClipper:
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm 

    def __call__(self, parameters):
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)

class SequentialBlockSampler(Sampler[List[int]]):
    """ Yield batches of contiguous, same-symbol indices in time order.
    
    Required by any path-dependent objective (drawdown, Calmar): a cumulative
    maximum over a random permutation of unrelated timestamps is meaningless.
    Block *order* is shuffled between epochs; order *within* a block is not
    """

    def __init__(
        self,
        group_ids: Sequence,
        batch_size: int,
        shuffle_blocks: bool = True,
        seed: int = 0,
    ):
        self.batch_size = batch_size 
        self.shuffle_blocks = shuffle_blocks
        self.seed = seed 
        self.epochs = 0 

        groups: Dict[object, List[int]] = {}
        for idx, gid in enumerate(group_ids):
            groups.setdefault(gid, []).append(idx)

        self.blocks: List[List[int]] = []
        for indices in groups.values():
            indices.sort()
            for start in range(0, len(indices), batch_size):
                block = indices[start:start + batch_size]
                if len(block) > 1:
                    self.blocks.append(block)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch 

    def __iter__(self) -> Iterator[List[int]]:
        order = list(range(len(self.blocks)))
        if self.shuffle_blocks: 
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)
        for i in order:
            yield self.blocks[i]

    def __len__(self) -> int:
        return len(self.blocks)

### Metric Helpers ###

def compute_sharpe_ratio(
    positions: torch.Tensor,
    returns: torch.Tensor,
    annualization_factor: float = ANNUALIZATION_SQRT,
) -> float:
    """Annualized Sharpe of ``positions * returns`` (detached, for reporting)"""
    pnl = (positions.reshape(-1) * returns.reshape(-1)).detach()
    std = pnl.std()
    if not torch.isfinite(std) or std.item() == 0.0:
        return 0.0
    return float(pnl.mean() / std * annualization_factor)

def _unpack_batch(batch, device):
    """Accept ``(x,y)`` or ``(x,y,ex_ante_vol)`` batches"""
    if len(batch) == 3:
        x, y, vol = batch
        return x.to(device), y.to(device), vol.to(device)
    x, y = batch 
    return x.to(device), y.to(device), None 

### Train / Evaluate ###

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clipper: Optional[GradientClipper] = None,
    accumulation_steps: Optional[int] = 8,
) -> Tuple[float, float]:
    """ One training epoch with epoch-level (accumulated) Sharpe
    
    ``accumulation_steps`` mini-batches are concatenated *keeping the graph*
    and a single objective is backpropagated over the pooled sample. With 
    ``batch_size = 256`` and ``accumulation_steps = 8`` each estimate sees 2,048
    observations instead of 256, cutting the standard error of the ratio by ~2.8x.
    
    ``accumulation_steps = None`` pools the entire epoch, it is only feasible when the epoch
    fits in memory
    """

    model.train() 

    losses: List[float] = []
    pending_positions: List[torch.Tensor] = []
    pending_returns: List[torch.Tensor] = []
    epoch_positions: List[torch.Tensor] = []
    epoch_returns: List[torch.Tensor] = []

    def flush() -> None:
        if not pending_positions:
            return 
        positions = torch.cat(pending_positions)
        returns = torch.cat(pending_returns)

        optimizer.zero_grad()
        result = criterion(positions, returns)
        loss = result[0] if isinstance(result, tuple) else result
        loss.backward()
        if gradient_clipper is not None:
            gradient_clipper(model.parameters())
        optimizer.step()

        losses.append(float(loss.detach()))
        pending_positions.clear()
        pending_returns.clear()

    for step, batch in enumerate(dataloader, start = 1):
        x, returns, vol = _unpack_batch(batch, device)

        positions, _ = model(x, ex_ante_vol = vol)
        positions = positions.reshape(-1)
        returns = returns.reshape(-1)

        pending_positions.append(positions)
        pending_returns.append(returns)
        epoch_positions.append(positions.detach().cpu())
        epoch_returns.append(returns.detach().cpu())

        if accumulation_steps is not None and step % accumulation_steps == 0:
            flush() 

    flush() # trailing partial group (or the whole epoch when steps is None)

    all_positions = torch.cat(epoch_positions)
    all_returns = torch.cat(epoch_returns)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, compute_sharpe_ratio(all_positions, all_returns)

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """ Evaluate, pooling the whole split before computing loss and Sharpe"""
    model.eval()

    all_positions, all_returns = [], []
    for batch in dataloader:
        x, returns, vol = _unpack_batch(batch, device)
        positions, _ = model(x, ex_ante_vol = vol)
        all_positions.append(positions.reshape(-1).cpu())
        all_returns.append(returns.reshape(-1).cpu())

    if not all_positions:
        return 0.0, 0.0

    positions = torch.cat(all_positions)
    returns = torch.cat(all_returns)

    result = criterion(positions, returns)
    loss = result[0] if isinstance(result, tuple) else result 

    return float(loss), compute_sharpe_ratio(positions, returns)

class PerformanceTracker:
    """ Per-epoch training history"""

    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_sharpe": [],
            "val_sharpe": [], "learning_rate": [], "epoch_time": [],
        }

    def update(self, metrics: TrainingMetrics) -> None:
        self.history["train_loss"].append(metrics.train_loss)
        self.history["val_loss"].append(metrics.val_loss)
        self.history["train_sharpe"].append(metrics.train_sharpe)
        self.history["val_sharpe"].append(metrics.val_sharpe)
        self.history["learning_rate"].append(metrics.learning_rate)
        self.history["epoch_time"].append(metrics.time_elapsed)

    def get_best_epoch(self, metric: str = "val_sharpe", mode: str = "max") -> int:
        values = self.history[metric]
        if not values:
            return 0
        return int(np.argmax(values) if mode == "max" else np.argmin(values))

    def get_summary(self) -> Dict:
        if not self.history["train_loss"]:
            return {}
        best = self.get_best_epoch()
        return {
            "best_epoch": best,
            "best_val_sharpe": self.history["val_sharpe"][best],
            "best_val_loss": self.history["val_loss"][best],
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "total_epochs": len(self.history["train_loss"]),
        }

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_Scheduler] = None,
        num_epochs: int = 100,
        device: Optional[torch.device] = None,
        early_stopping: Optional[EarlyStopping] = None,
        gradient_clip_norm: Optional[float] = 1.0,
        accumulation_steps: Optional[int] = 8,
        save_best_path: Optional[str] = None,
        validation_fn: Optional[Callable[[nn.Module], float]] = None,
        verbose: bool = True,
) -> Tuple[PerformanceTracker, Dict]:
    """ Full training loop 
    
    ``validation_fn`` lets the caller supply a validation score computed the 
    same way the backtest is -- forming the cross-sectional portfolio at each 
    validation timestamp == instead of a proxy the strategy never trades. When
    given, it is the early-stopping and checkpoint criterion
    """

    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    clipper = GradientClipper(gradient_clip_norm) if gradient_clip_norm else None
    tracker = PerformanceTracker()
    best_val_sharpe = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0 

    if verbose:
        print(f" Training on {device} | {num_epochs} epochs | " f"{len(train_loader)} train batches | accumulation = {accumulation_steps}")

    for epoch in range(num_epochs):
        started = time.time()

        train_loss, train_sharpe = train_epoch(
            model, train_loader, criterion, optimizer, device, clipper, accumulation_steps
        )
        val_loss, val_sharpe_proxy = evaluate_model(model, val_loader, criterion, device)

        # Prefer the backtest-consistent validation score when available
        val_sharpe = validation_fn(model) if validation_fn is not None else val_sharpe_proxy

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_sharpe)
            else:
                scheduler.step()

        tracker.update(
            TrainingMetrics(
                epoch = epoch,
                train_loss = train_loss,
                val_loss = val_loss,
                train_sharpe = train_sharpe,
                val_sharpe = val_sharpe,
                learning_rate = optimizer.param_groups[0]["lr"],
                time_elapsed = time.time() - started,
            )
        )

        if np.isfinite(val_sharpe) and val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_epoch = epoch 
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if save_best_path:
                torch.save(
                    {"epoch": epoch, "model_state_dict": best_state,
                    "val_sharpe": val_sharpe},
                    save_best_path,
                )
            
        if verbose:
            print(f" Epoch {epoch + 1:3d} / {num_epochs} | "
            f"loss {train_loss:8.4f} | train Sharpe {train_sharpe:6.3f} | "
            f"val Sharpe {val_sharpe:6.3f} | {time.time() - started:5.1f}s")

        # Early stopping on the SAME criteiron as checkpoint selection
        if early_stopping is not None and early_stopping(val_sharpe, epoch):
            if verbose:
                print(f" Early stopping at epoch {epoch + 1} "
                f"(best epoch {early_stopping.best_epoch + 1})")
            break 

    model.load_state_dict(best_state)
    return tracker, {"best_epoch": best_epoch, "best_val_sharpe": float(best_val_sharpe)}


def compute_metrics(
    positions: torch.Tensor,
    returns: torch.Tensor,
    timestamps: Optional[Sequence[object]] = None,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """ Evaluation metrics for a positions/returns pair
    
    Delegates to ``Utils.Metrics`` so there is exactly one definition of each 
    ratio. ``timestamps`` is required for a meaningful turnover figure: it 
    orders the positions in time. Without it, turnover is omitted rather than 
    computed over an arbitrary permutation
    """

    import pandas as pd

    pos = positions.detach().cpu().numpy().reshape(-1)
    rets = returns.detach().cpu().numpy().reshape(-1)
    pnl = pos * rets 

    index = pd.DatetimeIndex(list(timestamps)) if timestamps is not None else None

    if index is not None:
        series = pd.Series(pnl, index = index).sort_index()
    else:
        series = pd.Series(pnl)

    summary = Metrics.performance_summary(series, periods_per_year)

    if index is not None:
        ordered = pd.Series(pos, index = index).sort_index()
        summary["avg_turnover"] = float(np.abs(np.diff(ordered.to_numpy())).mean())

    summary["avg_position"] = float(np.abs(pos).mean())
    return summary 

def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    print(f"\n{title}")
    print("=" * 62)
    for key in (
        "sharpe_ratio", "daily_sharpe", "sortino_ratio", "calmar_ratio",
        "annualized_return", "volatility", "max_drawdown", "win_rate",
        "profit_factor", "avg_position", "avg_turnover", "n_periods",
    ):
        if key in metrics:
            print(f"{key:22s} {metrics[key]:12.4f}")
    print("=" * 62)
