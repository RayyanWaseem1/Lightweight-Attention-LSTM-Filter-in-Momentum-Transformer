#Training utils for the Momentum Transformer
#Training loops, early stopping, learning rate scheduling, and evals

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import time
from pathlib import Path 

@dataclass
class TrainingMetrics:
    #Container for the training metrics
    epoch: int
    train_loss: float
    val_loss: float
    train_sharpe: float
    val_sharpe: float
    learning_rate: float
    time_elapsed: float 


class EarlyStopping:
    #Early stopping to help prevent any overfitting
    #Monitors val loss and stops training if no improvement

    def __init__(self,
                 patience: int = 15,
                 min_delta: float = 0.0,
                 mode: str = "min"):
        #patience: number of epochs to wait for improvement
        #min_delta: minimum change to qualify as an actual improvement
        #mode: "min" for loss, "max" for metrics such as Sharpe

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode 
        self.counter = 0
        self.best_score = None
        self.early_stop = False 
        self.best_epoch = 0 

    def __call__(self, score: float, epoch: int) -> bool:
        #Check if the training should stop 

        #score: current val score
        #epoch: current epoch number

        #returns true if should stop, false otherwise

        if self.best_score is None:
            self.best_score = score 
            self.best_epoch = epoch
            return False 
        
        #Checking for any improvement 
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else: #mode == "max"
            improved = score > (self.best_score - self.min_delta)

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
    #Gradient Clipping 
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def __call__(self, parameters):
        #Clips the gradient
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)


class PerformanceTracker:
    #Tracking the training performance
    def __init__(self):
        self.history = {
            "train_loss":[],
            "val_loss": [],
            "train_sharpe":[],
            "val_sharpe": [],
            "learning_rate": [],
            "epoch_time": []
        }

    def update(self, metrics: TrainingMetrics):
        #Adding metrics for the current epoch
        self.history["train_loss"].append(metrics.train_loss)
        self.history["val_loss"].append(metrics.val_loss)
        self.history["train_sharpe"].append(metrics.train_sharpe)
        self.history["val_sharpe"].append(metrics.val_sharpe)
        self.history["learning_rate"].append(metrics.learning_rate)
        self.history["epoch_time"].append(metrics.time_elapsed)

    def get_best_epoch(self, metric: str = "val_sharpe", mode: str = "max") -> int: 
        #Getting the best epoch with the best validation metric
        values = self.history[metric]
        if mode == "max":
            return int(np.argmax(values))
        else:
            return int(np.argmin(values))
        
    def get_summary(self) -> Dict:
        #Summary Statistics
        best_epoch = self.get_best_epoch("val_sharpe", "max")
        return {
            "best_epoch": best_epoch,
            "best_val_sharpe": self.history["val_sharpe"][best_epoch],
            "best_val_loss": self.history["val_loss"][best_epoch],
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "total_epochs": len(self.history["train_loss"])
        }
    
def compute_sharpe_ratio(positions: torch.Tensor,
                         returns: torch.Tensor,
                         annualization_factor: float = (252 * 24) ** 0.5) -> float:
    #Computing sharpe from the positions and returns (hourly default)

    pnl = positions * returns
    mean_pnl = torch.mean(pnl)
    std_pnl = torch.std(pnl) + 1e-8

    if std_pnl < 1e-8:
        return 0.0
    
    sharpe = (mean_pnl / std_pnl) * annualization_factor
    return sharpe.item() 

def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float]:
    #Evaluating the model on the dataset 

    #Model: model we are evaluating 
    #Dataloader: dataloader with evaluation data
    #criterion: loss function
    #device: device to run on 

    #returns (avg_loss, sharpe_ratio) as a tuple

    model.eval()
    total_loss = 0.0
    all_positions = []
    all_returns = []

    with torch.no_grad():
        for batch in dataloader:
            x, returns = batch
            x = x.to(device)
            returns = returns.to(device)

            #Forward pass (models may return (positions, aux))
            out = model(x)
            positions = out[0] if isinstance(out, tuple) else out
            loss = criterion(positions, returns)

            total_loss += loss.item() 
            all_positions.append(positions.cpu())
            all_returns.append(returns.cpu())

    #Compute metrics
    avg_loss = total_loss / len(dataloader)

    all_positions = torch.cat(all_positions)
    all_returns = torch.cat(all_returns)
    sharpe = compute_sharpe_ratio(all_positions,all_returns)

    return avg_loss, sharpe 

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                gradient_clipper: Optional[GradientClipper] = None) -> Tuple[float, float]:
    #Training for one epoch 

    #Model: Model to train
    #Dataloader: Training data
    #Criterion: Loss function
    #Optimizer: Optimizer
    #Device: Device to run on 
    #Gradient_clipper: Optional Gradient clipper 

    #returns (avg_loss, sharpe_ratio) as a tuple 

    model.train()
    total_loss = 0.0
    all_positions = []
    all_returns = []

    for batch in dataloader:
        x, returns = batch
        x = x.to(device)
        returns = returns.to(device)

        #forward pass
        optimizer.zero_grad()
        out = model(x)
        positions = out[0] if isinstance(out, tuple) else out
        loss = criterion(positions, returns)

        #backward pass
        loss.backward() 

        #gradient clipping
        if gradient_clipper is not None:
            gradient_clipper(model.parameters())

        optimizer.step()

        #tracking the metrics
        total_loss += loss.item() 
        all_positions.append(positions.detach().cpu())
        all_returns.append(returns.cpu())

    #compute metrics
    avg_loss = total_loss / len(dataloader)

    all_positions = torch.cat(all_positions)
    all_returns = torch.cat(all_returns)
    sharpe = compute_sharpe_ratio(all_positions,all_returns)

    return avg_loss, sharpe 

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                num_epochs: int = 100,
                device: Optional[torch.device] = None,
                early_stopping: Optional[EarlyStopping] = None,
                gradient_clip_norm: Optional[float] = 1.0,
                save_best_path: Optional[str] = None,
                verbose: bool = True) -> PerformanceTracker:
    """
    Complete training loop
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        early_stopping: Optional early stopping
        gradient_clip_norm: Maximum gradient norm
        save_best_path: Path to save best model
        verbose: Whether to print progress
        
    Returns:
        PerformanceTracker with training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Setup utilities
    gradient_clipper = GradientClipper(gradient_clip_norm) if gradient_clip_norm else None
    tracker = PerformanceTracker()
    best_val_sharpe = -np.inf
    
    if verbose:
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print("=" * 80)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_sharpe = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clipper
        )
        
        # Validate
        val_loss, val_sharpe = evaluate_model(
            model, val_loader, criterion, device
        )
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_sharpe)  # Use Sharpe for plateau detection
            else:
                scheduler.step()
        
        # Track metrics
        epoch_time = time.time() - epoch_start
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_sharpe=train_sharpe,
            val_sharpe=val_sharpe,
            learning_rate=current_lr,
            time_elapsed=epoch_time
        )
        tracker.update(metrics)
        
        # Save best model
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            if save_best_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_sharpe': val_sharpe,
                }, save_best_path)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:7.4f} | "
                  f"Val Loss: {val_loss:7.4f} | "
                  f"Train Sharpe: {train_sharpe:6.3f} | "
                  f"Val Sharpe: {val_sharpe:6.3f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:5.2f}s")
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(-val_loss, epoch):  # Minimize loss
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best epoch: {early_stopping.best_epoch}")
                break
    
    if verbose:
        print("=" * 80)
        print("Training Complete")
        print("=" * 80)
        summary = tracker.get_summary()
        print(f"Best epoch: {summary['best_epoch']}")
        print(f"Best val Sharpe: {summary['best_val_sharpe']:.4f}")
        print(f"Best val loss: {summary['best_val_loss']:.4f}")
        print("=" * 80)
    
    return tracker


def compute_metrics(positions: torch.Tensor,
                   returns: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        positions: [batch] - predicted positions
        returns: [batch] - actual returns
        
    Returns:
        Dictionary of metrics
    """
    pnl = positions * returns
    
    # Convert to numpy
    pnl_np = pnl.cpu().numpy() if isinstance(pnl, torch.Tensor) else pnl
    positions_np = positions.cpu().numpy() if isinstance(positions, torch.Tensor) else positions
    
    # Sharpe ratio (hourly annualization)
    annualization_factor = np.sqrt(252 * 24)
    mean_pnl = np.mean(pnl_np)
    std_pnl = np.std(pnl_np)
    sharpe = (mean_pnl / (std_pnl + 1e-8)) * annualization_factor
    
    # Sortino ratio (downside deviation)
    downside_returns = np.minimum(pnl_np, 0)
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    sortino = (mean_pnl / (downside_std + 1e-8)) * annualization_factor
    
    # Maximum drawdown
    cumulative = np.cumsum(pnl_np)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown)
    
    # Calmar ratio
    annualized_return = mean_pnl * (252 * 24)
    calmar = annualized_return / (max_drawdown + 1e-8)
    
    # Win rate
    win_rate = np.mean(pnl_np > 0)
    
    # Average win/loss
    wins = pnl_np[pnl_np > 0]
    losses = pnl_np[pnl_np < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    # Profit factor
    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = total_wins / (total_losses + 1e-8)
    
    # Turnover (average absolute position change)
    position_changes = np.abs(np.diff(positions_np))
    avg_turnover = np.mean(position_changes)
    
    return {
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'max_drawdown': float(max_drawdown),
        'annualized_return': float(annualized_return),
        'volatility': float(std_pnl * annualization_factor),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'avg_turnover': float(avg_turnover)
    }


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics"""
    print(f"\n{title}")
    print("=" * 60)
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:8.4f}")
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:8.4f}")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:8.4f}")
    print(f"Annualized Return:   {metrics['annualized_return']:8.4f}")
    print(f"Volatility:          {metrics['volatility']:8.4f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']:8.4f}")
    print(f"Win Rate:            {metrics['win_rate']:8.4f}")
    print(f"Avg Win:             {metrics['avg_win']:8.4f}")
    print(f"Avg Loss:            {metrics['avg_loss']:8.4f}")
    print(f"Profit Factor:       {metrics['profit_factor']:8.4f}")
    print(f"Avg Turnover:        {metrics['avg_turnover']:8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    print("Testing Training Utilities\n")
    
    # Create dummy model and data
    from Models.Momentum_transformer import MomentumTransformerSimple
    from Models.config import get_default_config
    from torch.utils.data import TensorDataset
    
    config = get_default_config()
    config.model.input_dim = 10
    config.model.hidden_dim = 32
    
    model = MomentumTransformerSimple(config.model)
    
    # Dummy data
    X_train = torch.randn(500, 252, 10)
    y_train = torch.randn(500) * 0.02
    X_val = torch.randn(100, 252, 10)
    y_val = torch.randn(100) * 0.02
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Setup training
    from Utils.Losses import SharpeRatioLoss
    
    criterion = SharpeRatioLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=10, mode='min')
    
    # Train
    print("Training model for 20 epochs...")
    tracker = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=20,
        early_stopping=early_stopping,
        gradient_clip_norm=1.0,
        verbose=True
    )
    
    # Evaluate
    print("\nEvaluating model...")
    val_loss, val_sharpe = evaluate_model(model, val_loader, criterion, torch.device('cpu'))
    
    # Compute detailed metrics
    all_positions = []
    all_returns = []
    
    model.eval()
    with torch.no_grad():
        for x, returns in val_loader:
            out = model(x)
            positions = out[0] if isinstance(out, tuple) else out
            all_positions.append(positions)
            all_returns.append(returns)
    
    all_positions = torch.cat(all_positions)
    all_returns = torch.cat(all_returns)
    
    metrics = compute_metrics(all_positions, all_returns)
    print_metrics(metrics, "Validation Metrics")
    
    print("\n✅ Training utilities working correctly!")
