#A comprehensive backtesting framework for the Momentum Transformer

#Features:
#Walk forward analysis
#Regime based testing
#Transaction cost modeling
#Statistical validation
#Performance attribution
#Out of sample testing

import numpy as np 
import pandas as pd
import torch 
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt 
from pathlib import Path 

@dataclass
class BacktestConfig: 
    #Backtesting Configuration

    #Transaction costs
    transaction_cost: float = 0.001 #10 bps
    slippage: float = 0.0005 #5 bps

    #Position constraints
    max_position: float = 1.0
    min_position: float = -1.0

    #Risk management
    max_drawdown_stop: Optional[float] = None #Stop trading if DD exceeds
    daily_var_limit: Optional[float] = None #Daily VaR limit 

    #Rebalancing
    rebalance_threshold: float = 0.01 #only trade if the position change > threshold

    #Data
    initial_capital: float = 10000.0
    compound_returns: bool = True 

@dataclass
class BacktestResults:
    #Results from the backtest

    #Equity curve
    dates: pd.DatetimeIndex
    equity: np.ndarray
    positions: np.ndarray
    returns: np.ndarray

    #Trades
    trades: pd.DataFrame

    #Metrics
    metrics: Dict[str, float]

    #Regime-specific metrics
    regime_metrics: Optional[Dict[str, Dict[str, float]]] = None 

    #Additional info
    metadata: Optional[Dict] = None 

class Backtester:
    #Comprehensive backtesting engine
    #Simulates realistic trading with:
    #Transaction costs
    #Slippage
    #Position constraints
    #Risk management

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset() 

    def reset(self):
        #resetting backtest state
        self.equity = [self.config.initial_capital]
        self.positions = []
        self.returns = []
        self.trades = []
        self.current_position = 0.0
        self.peak_equity = self.config.initial_capital

    def run_backtest(self,
                     model: torch.nn.Module,
                     test_data: torch.utils.data.DataLoader,
                     dates: pd.DatetimeIndex,
                     market_returns: np.ndarray,
                     regimes: Optional[np.ndarray] = None,
                     pre_normalized_predictions: Optional[np.ndarray] = None) -> BacktestResults:
        #Run the complete backtest

        #model: trained model
        #test_data: test data loader
        #dates: dates for the test period 
        #market_returns: actual market returns
        #regimes: optional regime labels for analysis 

        #returns the BacktestResults with the comprehensive metrics

        self.reset()

        model.eval()
        all_predictions = []
        device = next(model.parameters()).device

        #Use pre-normalized predictions if provided, otherwise generate and normalize
        if pre_normalized_predictions is not None:
            # Use pre-normalized predictions (already clipped to [-1, 1])
            all_predictions = pre_normalized_predictions
            print("✓ Using pre-normalized predictions (already safe for position sizing)")
        else:
            # Generate predictions from model
            all_predictions = []
            with torch.no_grad():
                for batch_x, _ in test_data:
                    batch_x = batch_x.to(device)
                    out = model(batch_x)
                    predictions = out[0] if isinstance(out, tuple) else out
                    all_predictions.extend(predictions.cpu().numpy())
            
            all_predictions = np.array(all_predictions)
            
            #CRITICAL: Normalize predictions to prevent overleveraging
            print("\nWARNING: Predictions not pre-normalized. Normalizing now...")
            print(f"   Original range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")
            
            # Step 1: Scale by standard deviation if too large
            original_std = all_predictions.std()
            if original_std > 0.5:
                scaling_factor = original_std * 2
                all_predictions = all_predictions / scaling_factor
                print(f"   Scaled down by factor of {scaling_factor:.3f}")
            
            # Step 2: Clip to safe range [-1, 1]
            all_predictions = np.clip(all_predictions, -1.0, 1.0)
            print(f"   Normalized range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")
            print("   ✓ Predictions normalized and safe for position sizing")

        #Ensure we have matching lengths
        n_samples = min(len(all_predictions), len(market_returns), len(dates))
        predictions = all_predictions[:n_samples]
        returns = market_returns[:n_samples]
        dates = dates[:n_samples]
        regimes = regimes[:n_samples] if regimes is not None else None

        #Simulate trading
        raw_returns = []
        for i in range(n_samples):
            #proposed position
            target_position = np.clip(
                predictions[i],
                self.config.min_position,
                self.config.max_position
            )

            #Check rebalance threshold
            position_change = abs(target_position - self.current_position)

            if position_change > self.config.rebalance_threshold:
                #execute the trade
                trade_cost = self._calculate_trade_cost(position_change)

                #record the trade
                self.trades.append({
                    "date": dates[i],
                    "from_position": self.current_position,
                    "to_position": target_position,
                    "cost": trade_cost
                })

                self.current_position = target_position

            #calculate the return 
            gross_return = self.current_position * returns[i]

            #apply costs if we traded
            if position_change > self.config.rebalance_threshold:
                net_return = gross_return - trade_cost
            else:
                net_return = gross_return

            raw_returns.append(gross_return)

            #update the equity
            if self.config.compound_returns:
                new_equity = self.equity[-1] * (1 + net_return)
            else:
                new_equity = self.equity[-1] + (self.config.initial_capital * net_return)

            self.equity.append(new_equity)
            self.returns.append(net_return)
            self.positions.append(self.current_position)

            #update peak for drawdown
            self.peak_equity = max(self.peak_equity, new_equity)

            #check the risk limits
            if self.config.max_drawdown_stop is not None:
                current_dd = (self.peak_equity - new_equity) / self.peak_equity
                if current_dd > self.config.max_drawdown_stop:
                    print(f"Max drawdown limit hit at {dates[i]}: {current_dd:.2%}")
                    break 

        #convert to arrays
        equity = np.array(self.equity[1:]) #skip initial capital
        positions = np.array(self.positions)
        returns_array = np.array(self.returns)

        #calculating metrics
        metrics = self._calculate_metrics(returns_array, equity, positions)

        #pre-cost metrics for diagnostics (use hourly annualization)
        periods_per_year = 252 * 24
        raw_returns = np.array(raw_returns)
        metrics["sharpe_pre_cost"] = (np.mean(raw_returns) / (np.std(raw_returns) + 1e-8)) * np.sqrt(periods_per_year)
        metrics["turnover"] = np.mean(np.abs(np.diff(positions))) if len(positions) > 1 else 0.0

        #Regime specific metrics
        regime_metrics = None
        if regimes is not None:
            regime_metrics = self._calculate_regime_metrics(
                returns_array,
                regimes
            )

        #Creating trades dataframe
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame() 

        return BacktestResults(
            dates = dates[:len(equity)],
            equity = equity,
            positions = positions,
            returns = returns_array,
            trades = trades_df,
            metrics = metrics,
            regime_metrics= regime_metrics,
            metadata = {
                "config": self.config,
                "n_trades": len(self.trades),
                "avg_position": np.mean(np.abs(positions)),
                "position_turnover": np.mean(np.abs(np.diff(positions)))
            }
        )
    def _calculate_trade_cost(self, position_change: float) -> float:
        #Calculate the cost of trade including the slippage
        transaction_cost = self.config.transaction_cost * position_change
        slippage = self.config.slippage * position_change
        return transaction_cost + slippage 
    
    def _calculate_metrics(self,
                           returns: np.ndarray,
                           equity: np.ndarray,
                           positions: np.ndarray) -> Dict[str, float]:
        #Calculating the comprehensive performance metrics 

        #basic stats (hourly data)
        total_return = (equity[-1] / self.config.initial_capital) - 1
        n_periods = len(returns)
        periods_per_year = 252 * 24
        n_years = n_periods / periods_per_year

        #annualized return
        if self.config.compound_returns:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annualized_return = np.mean(returns) * periods_per_year 

        #volatility
        vol = np.std(returns)
        volatility = vol * np.sqrt(periods_per_year)

        #sharpe ratio
        sharpe = (np.mean(returns) / (vol + 1e-8)) * np.sqrt(periods_per_year)

        #Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else vol
        denom_sortino = downside_std if downside_std > 1e-8 else 1e-8
        sortino = (np.mean(returns) / denom_sortino) * np.sqrt(periods_per_year)

        #Maximize drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdown)

        #Calmar ratio 
        calmar = annualized_return / (max_drawdown + 1e-8)

        #win rate
        win_rate = np.mean(returns > 0)

        #Average win/.oss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        #Profit factor
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
        profit_factor = total_wins / (total_losses + 1e-8)

        #Recovery factor
        recovery_factor = total_return / (max_drawdown + 1e-8)

        #Average position and turnover 
        avg_position = np.mean(np.abs(positions))
        turnover = np.mean(np.abs(np.diff(positions)))

        #Longest winning/losing streak
        def longest_streak(arr):
            if len(arr) == 0:
                return 0 
            max_streak = current = 1
            for i in range(1, len(arr)):
                if arr[i] == arr[i-1]:
                    current += 1
                    max_streak = max(max_streak, current)
                else:
                    current = 1
            return max_streak
        
        win_streak = longest_streak(returns > 0)
        loss_streak = longest_streak(returns < 0)

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "recovery_factor": recovery_factor,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_position": avg_position,
            "turnover": turnover,
            "win_streak": win_streak,
            "loss_streak": loss_streak,
            "n_periods": n_periods
        }
    
    def _calculate_regime_metrics(self,
                                  returns: np.ndarray,
                                  regimes: np.ndarray) -> Dict[str, Dict[str, float]]:
        #Calculating metrics for each regime
        regime_metrics = {}

        unique_regimes = np.unique(regimes)

        for regime in unique_regimes:
            mask = regimes == regime 
            regime_returns = returns[mask]

            if len(regime_returns) > 0:
                regime_metrics[str(regime)] = {
                    "n_periods": len(regime_returns),
                    "mean_return": np.mean(regime_returns),
                    "std_return": np.std(regime_returns),
                    "sharpe": (np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)) * np.sqrt(252 * 24),
                    "win_rate": np.mean(regime_returns > 0),
                    "total_return": np.prod(1 + regime_returns) - 1
                }
        return regime_metrics
    

class WalkForwardAnalyzer:
    #Walk forward analysis
    #trains on rolling windows and tests on out of sample data

    def __init__(self,
                 train_window: int = 756, #3 years
                 test_window: int = 63, #1 quarter,
                 step_size: int = 21): #1 month
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def run_analysis(self,
                     features: np.ndarray,
                     returns: np.ndarray,
                     dates: pd.DatetimeIndex,
                     model_factory: Callable,
                     training_fn: Callable,
                     config: BacktestConfig) -> Dict:
            #Run walk forward analysis 
            
            
            #Features: Feature matrix
            #Returns: Return series
            #Dates: Dates 
            #Model_factory: Function that creates a new model
            #Training_fn: Function that trains model
            #config: Backtest Config

            #Returns dictionary with results for each window 

            results = []

            n_samples = len(features)
            start_idx = 0

            window_num = 0

            while start_idx + self.train_window + self.test_window <= n_samples:
                print(f"\n Window {window_num + 1}:")
                print(f" Train: {dates[start_idx]} to {dates[start_idx + self.train_window - 1]}")

                #Define train and test periods
                train_end = start_idx + self.train_window
                test_end = min(train_end + self.test_window, n_samples)

                print(f" Test: {dates[train_end]} to {dates[test_end - 1]}")

                #Extract data
                train_features = features[start_idx:train_end]
                train_returns = returns[start_idx:train_end]

                test_features = features[train_end:test_end]
                test_returns = returns[train_end:test_end]
                test_dates = dates[train_end:test_end]

                #create and train the model
                model = model_factory()
                trained_model = training_fn(model, train_features, train_returns)

                #backtest on the test period 
                backtester = Backtester(config) 

                #Generate Predictions
                from data.Dataset import RollingWindowDataset
                from torch.utils.data import DataLoader

                seq_len = min(252, len(test_features) - 1)
                if seq_len < 1:
                    print(" Skipping window due to insufficient test data")
                    start_idx += self.step_size
                    window_num += 1
                    continue

                test_dataset = RollingWindowDataset(
                    test_features,
                    test_returns,
                    sequence_length = seq_len
                )

                test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

                #Run backtest
                backtest_results = backtester.run_backtest(
                    trained_model,
                    test_loader,
                    test_dates,
                    test_returns
                )

                results.append({
                    "window": window_num,
                    "train_start": dates[start_idx],
                    "train_end": dates[train_end - 1],
                    "test_start": dates[train_end],
                    "test_end": dates[test_end - 1],
                    "metrics": backtest_results.metrics,
                    "equity": backtest_results.equity,
                    "returns": backtest_results.returns
                })

                print(f" Sharpe: {backtest_results.metrics['sharpe_ratio']:.3f}")
                print(f" Return: {backtest_results.metrics['total_return']:.2%}")

                #Move on to the next window
                start_idx += self.step_size
                window_num += 1

            #aggregate the results
            all_returns = np.concatenate([r["returns"] for r in results])
            all_equity = np.concatenate([r["equity"] for r in results])

            aggregate_metrics = self._aggregate_metrics(results)

            return {
                "windows": results,
                "aggregate_metrics": aggregate_metrics,
                "all_returns": all_returns,
                "all_equity": all_equity,
                "n_windows": len(results)
            }
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        #Aggregate the metrics across all windows
        all_sharpes = [r["metrics"]["sharpe_ratio"] for r in results]
        all_returns = [r["metrics"]["total_return"] for r in results]
        all_drawdowns = [r["metrics"]["max_drawdown"] for r in results]
        all_win_rates = [r["metrics"]["win_rate"] for r in results]

        return {
            "mean_sharpe": np.mean(all_sharpes),
            "std_sharpe": np.std(all_sharpes),
            "min_sharpe": np.min(all_sharpes),
            "max_sharpe": np.max(all_sharpes),
            "mean_return": np.mean(all_returns),
            "std_return": np.std(all_returns),
            "mean_drawdown": np.mean(all_drawdowns),
            "max_drawdown": np.max(all_drawdowns),
            "mean_win_rate": np.mean(all_win_rates),
            "consistent_windows": np.sum(np.array(all_sharpes) > 0),
            "total_windows": len(results)
        }

def print_backtest_results(results: BacktestResults, title: str = "Backtest Results"):
    """Pretty print backtest results"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    
    print("\nPerformance Metrics:")
    print("-" * 80)
    print(f"Total Return:        {results.metrics['total_return']:8.2%}")
    print(f"Annualized Return:   {results.metrics['annualized_return']:8.2%}")
    print(f"Volatility:          {results.metrics['volatility']:8.2%}")
    print(f"Sharpe Ratio:        {results.metrics['sharpe_ratio']:8.3f}")
    print(f"Sortino Ratio:       {results.metrics['sortino_ratio']:8.3f}")
    print(f"Calmar Ratio:        {results.metrics['calmar_ratio']:8.3f}")
    print(f"Max Drawdown:        {results.metrics['max_drawdown']:8.2%}")
    print(f"Recovery Factor:     {results.metrics['recovery_factor']:8.3f}")
    
    print("\nTrading Statistics:")
    print("-" * 80)
    print(f"Win Rate:            {results.metrics['win_rate']:8.2%}")
    print(f"Profit Factor:       {results.metrics['profit_factor']:8.3f}")
    print(f"Avg Win:             {results.metrics['avg_win']:8.4f}")
    print(f"Avg Loss:            {results.metrics['avg_loss']:8.4f}")
    print(f"Win Streak:          {results.metrics['win_streak']:8.0f}")
    print(f"Loss Streak:         {results.metrics['loss_streak']:8.0f}")
    
    print("\nPosition Statistics:")
    print("-" * 80)
    print(f"Avg Position:        {results.metrics['avg_position']:8.2%}")
    print(f"Turnover:            {results.metrics['turnover']:8.4f}")
    print(f"Number of Trades:    {results.metadata['n_trades']:8.0f}")
    
    if results.regime_metrics:
        print("\nRegime-Specific Performance:")
        print("-" * 80)
        for regime, metrics in results.regime_metrics.items():
            print(f"\nRegime {regime}:")
            print(f"  Periods:     {metrics['n_periods']:6.0f}")
            print(f"  Sharpe:      {metrics['sharpe']:6.3f}")
            print(f"  Win Rate:    {metrics['win_rate']:6.2%}")
            print(f"  Total Ret:   {metrics['total_return']:6.2%}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("Testing Backtesting Framework\n")
    
    # This would normally use real data and models
    # Here we show the structure
    
    print("✅ Backtesting framework ready!")
    print("\nFeatures:")
    print("  - Realistic transaction costs and slippage")
    print("  - Position constraints and risk limits")
    print("  - Walk-forward analysis")
    print("  - Regime-based performance analysis")
    print("  - Comprehensive metrics")
