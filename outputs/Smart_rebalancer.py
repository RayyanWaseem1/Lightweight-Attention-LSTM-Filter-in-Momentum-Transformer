"""
Smart Rebalancing Engine - Minimize Turnover & Transaction Costs

FEATURES:
✅ Keeps positions with strong signals (>0.3)
✅ Exits positions with weak signals (<0.2)
✅ Adds new positions only if significantly stronger
✅ Considers transaction costs in decisions
✅ Tracks which model (vanilla/attention) used per stock
✅ Position-aware decision making

REBALANCING LOGIC:
1. Check current positions
2. Evaluate each against new signals
3. Exit if:
   - Signal < 0.2 (too weak)
   - Signal changed direction
   - Stop loss hit (-5%)
4. Keep if:
   - Signal > 0.3 (still strong)
   - Signal 0.2-0.3 and profitable
5. Add new if:
   - Portfolio has < max_positions
   - Signal > 0.4 AND > weakest_current + 0.1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from outputs.Position_tracker import Position, PositionTracker


@dataclass
class RebalanceAction:
    """Action to take for a symbol"""
    symbol: str
    action: str  # 'KEEP', 'EXIT', 'ADD', 'REDUCE', 'INCREASE'
    reason: str
    signal: float
    target_weight: Optional[float] = None
    model_used: Optional[str] = None  # 'vanilla' or 'attention'


@dataclass
class RebalanceDecision:
    """Complete rebalancing decision"""
    actions: List[RebalanceAction]
    estimated_trades: int
    estimated_cost_pct: float
    turnover_pct: float
    
    def get_exits(self) -> List[RebalanceAction]:
        return [a for a in self.actions if a.action == 'EXIT']
    
    def get_keeps(self) -> List[RebalanceAction]:
        return [a for a in self.actions if a.action == 'KEEP']
    
    def get_adds(self) -> List[RebalanceAction]:
        return [a for a in self.actions if a.action == 'ADD']


class SmartRebalancer:
    """
    Smart rebalancing engine with turnover minimization
    
    Key Features:
    - Position-aware decision making
    - Transaction cost optimization
    - Regime-aware model tracking
    - Configurable thresholds
    """
    
    def __init__(self,
                 keep_threshold: float = 0.3,
                 exit_threshold: float = 0.2,
                 add_threshold: float = 0.4,
                 min_improvement: float = 0.1,
                 max_positions: int = 10,
                 max_position_weight: float = 0.15,
                 transaction_cost_bps: float = 10.0,
                 stop_loss_pct: float = -0.05):
        """
        Initialize smart rebalancer
        
        Args:
            keep_threshold: Keep position if |signal| > this
            exit_threshold: Exit position if |signal| < this
            add_threshold: Only add new if |signal| > this
            min_improvement: New must be this much better than weakest
            max_positions: Maximum number of positions
            max_position_weight: Maximum weight per position (15%)
            transaction_cost_bps: Transaction cost in basis points
            stop_loss_pct: Stop loss threshold (-5%)
        """
        self.keep_threshold = keep_threshold
        self.exit_threshold = exit_threshold
        self.add_threshold = add_threshold
        self.min_improvement = min_improvement
        self.max_positions = max_positions
        self.max_position_weight = max_position_weight
        self.transaction_cost = transaction_cost_bps / 10000
        self.stop_loss_pct = stop_loss_pct
    
    def rebalance(self,
                  current_positions: Dict[str, Position],
                  new_signals: Dict[str, float],
                  regime_usage: Optional[Dict[str, Dict]] = None) -> RebalanceDecision:
        """
        Determine rebalancing actions
        
        Args:
            current_positions: Dict of {symbol: Position}
            new_signals: Dict of {symbol: signal_strength}
            regime_usage: Optional dict of regime model usage
        
        Returns:
            RebalanceDecision with all actions
        """
        actions = []
        
        print(f"\n{'='*100}")
        print(f"SMART REBALANCING ANALYSIS")
        print(f"{'='*100}")
        print(f"Current positions: {len(current_positions)}")
        print(f"New signals available: {len(new_signals)}")
        print(f"\n{'Symbol':<8} {'Action':<8} {'Entry':>8} {'Current':>8} {'P&L':>8} {'Age':>6} {'Reason':<40}")
        print(f"{'-'*100}")
        
        # Step 1: Evaluate current positions
        positions_to_keep = []
        
        for symbol, position in current_positions.items():
            current_signal = new_signals.get(symbol, 0)
            
            action, reason = self._evaluate_position(position, current_signal)
            
            # Determine which model used
            model_used = None
            if regime_usage and symbol in regime_usage:
                if regime_usage[symbol]['attention'] > regime_usage[symbol]['vanilla']:
                    model_used = 'attention'
                else:
                    model_used = 'vanilla'
            
            rebal_action = RebalanceAction(
                symbol=symbol,
                action=action,
                reason=reason,
                signal=current_signal,
                model_used=model_used
            )
            actions.append(rebal_action)
            
            if action == 'KEEP':
                positions_to_keep.append((symbol, abs(current_signal)))
            
            # Print
            print(f"{symbol:<8} {action:<8} "
                  f"{position.entry_signal:>+7.2f} {current_signal:>+7.2f} "
                  f"{position.unrealized_pnl_pct:>7.1%} "
                  f"{position.age_days:>5.1f}d {reason:<40}")
        
        # Step 2: Consider new candidates
        current_symbols = set(current_positions.keys())
        available_slots = self.max_positions - len(positions_to_keep)
        
        if available_slots > 0:
            print(f"\n{'NEW CANDIDATES':<20} {'Signal':<8} {'Decision':<10} {'Reason':<50}")
            print(f"{'-'*100}")
            
            # Filter and sort candidates
            candidates = [
                (symbol, signal) for symbol, signal in new_signals.items()
                if symbol not in current_symbols and abs(signal) >= self.add_threshold
            ]
            candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get weakest current signal
            if positions_to_keep:
                weakest_current = min(positions_to_keep, key=lambda x: x[1])[1]
            else:
                weakest_current = 0
            
            # Evaluate candidates
            for symbol, signal in candidates[:min(available_slots + 5, len(candidates))]:
                # Determine model used
                model_used = None
                if regime_usage and symbol in regime_usage:
                    if regime_usage[symbol]['attention'] > regime_usage[symbol]['vanilla']:
                        model_used = 'attention'
                    else:
                        model_used = 'vanilla'
                
                # Only add if significantly stronger
                if abs(signal) > weakest_current + self.min_improvement:
                    if len([a for a in actions if a.action == 'ADD']) < available_slots:
                        rebal_action = RebalanceAction(
                            symbol=symbol,
                            action='ADD',
                            reason=f'Strong signal, >{weakest_current:.2f}+{self.min_improvement:.2f}',
                            signal=signal,
                            model_used=model_used
                        )
                        actions.append(rebal_action)
                        print(f"{symbol:<20} {signal:>+7.2f} {'ADD':<10} {rebal_action.reason:<50}")
                    else:
                        print(f"{symbol:<20} {signal:>+7.2f} {'SKIP':<10} Portfolio full")
                else:
                    print(f"{symbol:<20} {signal:>+7.2f} {'SKIP':<10} Not strong enough vs weakest ({weakest_current:.2f})")
        
        # Step 3: Calculate target weights
        self._calculate_target_weights(actions, new_signals)
        
        # Step 4: Estimate costs
        num_exits = len([a for a in actions if a.action == 'EXIT'])
        num_adds = len([a for a in actions if a.action == 'ADD'])
        estimated_trades = num_exits + num_adds
        estimated_cost_pct = estimated_trades * self.transaction_cost
        
        # Calculate turnover
        num_positions = len(current_positions)
        if num_positions > 0:
            turnover_pct = estimated_trades / (num_positions * 2)  # Divide by 2 because each change is 2 trades
        else:
            turnover_pct = 0
        
        decision = RebalanceDecision(
            actions=actions,
            estimated_trades=estimated_trades,
            estimated_cost_pct=estimated_cost_pct,
            turnover_pct=turnover_pct
        )
        
        # Print summary
        print(f"\n{'='*100}")
        print(f"REBALANCING SUMMARY")
        print(f"{'='*100}")
        print(f"Positions to KEEP: {len(decision.get_keeps())}")
        print(f"Positions to EXIT: {len(decision.get_exits())}")
        print(f"Positions to ADD:  {len(decision.get_adds())}")
        print(f"Estimated trades:  {estimated_trades}")
        print(f"Estimated cost:    {estimated_cost_pct:.2%}")
        print(f"Turnover:          {turnover_pct:.1%}")
        print(f"{'='*100}\n")
        
        return decision
    
    def _evaluate_position(self, position: Position, current_signal: float) -> Tuple[str, str]:
        """
        Evaluate whether to keep or exit a position
        
        Returns:
            (action, reason) tuple
        """
        entry_signal = position.entry_signal
        
        # RULE 1: Stop loss
        if position.unrealized_pnl_pct < self.stop_loss_pct:
            return ('EXIT', f'Stop loss ({position.unrealized_pnl_pct:.1%})')
        
        # RULE 2: Signal reversed direction
        if np.sign(entry_signal) != np.sign(current_signal):
            return ('EXIT', 'Signal reversed direction')
        
        # RULE 3: Signal too weak
        if abs(current_signal) < self.exit_threshold:
            return ('EXIT', f'Signal too weak (<{self.exit_threshold})')
        
        # RULE 4: Signal still strong
        if abs(current_signal) >= self.keep_threshold:
            signal_change = entry_signal - current_signal
            if abs(signal_change) > 0.1:
                return ('KEEP', f'Signal still strong but weakened {signal_change:+.2f}')
            else:
                return ('KEEP', f'Signal still strong (>{self.keep_threshold})')
        
        # RULE 5: Gray zone (exit_threshold < signal < keep_threshold)
        # Consider P&L
        
        if position.unrealized_pnl_pct > 0.02:  # Profitable by 2%+
            return ('KEEP', f'Profitable ({position.unrealized_pnl_pct:+.1%}), signal borderline')
        
        if position.unrealized_pnl_pct < -0.02:  # Losing 2%+
            return ('EXIT', f'Losing ({position.unrealized_pnl_pct:+.1%}), signal borderline')
        
        # Default: Keep in gray zone
        return ('KEEP', 'Gray zone, signal borderline')
    
    def _calculate_target_weights(self, actions: List[RebalanceAction], signals: Dict[str, float]):
        """Calculate target portfolio weights"""
        # Get positions to hold
        positions_to_hold = [
            (action.symbol, signals[action.symbol])
            for action in actions
            if action.action in ['KEEP', 'ADD']
        ]
        
        if not positions_to_hold:
            return
        
        # Normalize to weights
        total_abs_signal = sum(abs(signal) for _, signal in positions_to_hold)
        
        for action in actions:
            if action.action in ['KEEP', 'ADD']:
                weight = abs(signals[action.symbol]) / total_abs_signal
                # Cap at max position weight
                weight = min(weight, self.max_position_weight)
                action.target_weight = weight


def example_usage():
    """Example of smart rebalancing"""
    
    # Create positions
    current_positions = {
        'AAPL': Position('AAPL', 'LONG', 100, 180.0, 182.5, 
                        datetime.now() - pd.Timedelta(hours=48), 0.73, 0.15),
        'MSFT': Position('MSFT', 'LONG', 50, 380.0, 385.0,
                        datetime.now() - pd.Timedelta(hours=36), 0.65, 0.13),
        'TSLA': Position('TSLA', 'SHORT', 30, 250.0, 245.0,
                        datetime.now() - pd.Timedelta(hours=24), -0.58, 0.12),
        'GOOGL': Position('GOOGL', 'LONG', 80, 140.0, 142.0,
                         datetime.now() - pd.Timedelta(hours=72), 0.52, 0.11),
        'BA': Position('BA', 'SHORT', 25, 180.0, 182.0,
                      datetime.now() - pd.Timedelta(hours=48), -0.32, 0.07),
    }
    
    # New signals
    new_signals = {
        'AAPL': 0.71,   # Slightly weaker
        'MSFT': 0.68,   # Slightly stronger
        'TSLA': -0.55,  # Slightly weaker
        'GOOGL': 0.54,  # Slightly stronger
        'BA': -0.29,    # Much weaker (below 0.3!)
        # New candidates
        'META': 0.51,
        'AMZN': 0.59,
        'NVDA': 0.48,
    }
    
    # Regime usage
    regime_usage = {
        'AAPL': {'attention': 0.8, 'vanilla': 0.2},
        'MSFT': {'attention': 0.7, 'vanilla': 0.3},
        'META': {'attention': 0.9, 'vanilla': 0.1},
        'AMZN': {'attention': 0.6, 'vanilla': 0.4},
    }
    
    # Rebalance
    rebalancer = SmartRebalancer(
        keep_threshold=0.3,
        exit_threshold=0.2,
        add_threshold=0.4,
        max_positions=10
    )
    
    decision = rebalancer.rebalance(current_positions, new_signals, regime_usage)
    
    # Show final portfolio
    print(f"\nFINAL PORTFOLIO:")
    print(f"{'-'*100}")
    print(f"{'Symbol':<8} {'Signal':>8} {'Weight':>8} {'Model':<12}")
    print(f"{'-'*100}")
    
    for action in decision.actions:
        if action.action in ['KEEP', 'ADD']:
            model = action.model_used or 'unknown'
            print(f"{action.symbol:<8} {action.signal:>+7.2f} {action.target_weight*100:>7.1f}% {model:<12}")
    
    return decision


if __name__ == "__main__":
    decision = example_usage()