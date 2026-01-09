"""
Position Tracking System - Database & Monitoring
This is for Alpaca Sandbox deployment
Again not updated, will get to it later. 
Should track all of the portfolio positions with:
    -entry/exit prices and times
    -unrealized/realized P&L
    -position age and signals
    -trade history
    -performance metrics
Will get to it later

"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Position:
    """Current portfolio position"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    entry_signal: float
    portfolio_weight: float
    
    # Calculated properties
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.direction == 'LONG':
            return (self.current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - self.current_price) / self.entry_price
    
    @property
    def unrealized_pnl_dollars(self) -> float:
        if self.direction == 'LONG':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def current_value(self) -> float:
        return self.current_price * self.quantity
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.entry_time).total_seconds() / 3600
    
    @property
    def age_days(self) -> float:
        return self.age_hours / 24


@dataclass
class Trade:
    """Completed trade record"""
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    entry_signal: float
    exit_signal: float
    realized_pnl_pct: float
    realized_pnl_dollars: float
    commission: float = 0.0
    
    @property
    def holding_period_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600
    
    @property
    def holding_period_days(self) -> float:
        return self.holding_period_hours / 24


class PositionTracker:
    """
    Tracks positions and trades in SQLite database
    
    Features:
    - Add/remove positions
    - Update current prices
    - Record trades
    - Query history
    - Performance analytics
    """
    
    def __init__(self, db_path: str = "trading_positions.db"):
        """Initialize position tracker with database"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Positions table (current positions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                direction TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_signal REAL NOT NULL,
                portfolio_weight REAL NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        
        # Trades table (completed trades)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_signal REAL NOT NULL,
                exit_signal REAL NOT NULL,
                realized_pnl_pct REAL NOT NULL,
                realized_pnl_dollars REAL NOT NULL,
                commission REAL DEFAULT 0,
                notes TEXT
            )
        """)
        
        # Signal history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal REAL NOT NULL,
                price REAL NOT NULL
            )
        """)
        
        # Portfolio value history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_history (
                timestamp TEXT PRIMARY KEY,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                num_positions INTEGER NOT NULL,
                unrealized_pnl REAL NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_position(self, position: Position):
        """Add new position to tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO positions 
            (symbol, direction, quantity, entry_price, current_price, 
             entry_time, entry_signal, portfolio_weight, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.symbol,
            position.direction,
            position.quantity,
            position.entry_price,
            position.current_price,
            position.entry_time.isoformat(),
            position.entry_signal,
            position.portfolio_weight,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Added position: {position.direction} {position.symbol} @ ${position.entry_price:.2f}")
    
    def remove_position(self, symbol: str, exit_price: float, exit_signal: float, notes: str = ""):
        """Remove position and record trade"""
        position = self.get_position(symbol)
        if position is None:
            print(f"⚠️  No position found for {symbol}")
            return
        
        # Calculate realized P&L
        if position.direction == 'LONG':
            realized_pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            realized_pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        realized_pnl_dollars = realized_pnl_pct * position.entry_price * position.quantity
        
        # Record trade
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades 
            (symbol, direction, quantity, entry_price, exit_price,
             entry_time, exit_time, entry_signal, exit_signal,
             realized_pnl_pct, realized_pnl_dollars, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            position.direction,
            position.quantity,
            position.entry_price,
            exit_price,
            position.entry_time.isoformat(),
            datetime.now().isoformat(),
            position.entry_signal,
            exit_signal,
            realized_pnl_pct,
            realized_pnl_dollars,
            notes
        ))
        
        # Remove from positions
        cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Closed position: {position.direction} {symbol} @ ${exit_price:.2f} | P&L: {realized_pnl_pct:+.2%} (${realized_pnl_dollars:+,.2f})")
    
    def update_current_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for symbol, price in prices.items():
            cursor.execute("""
                UPDATE positions 
                SET current_price = ?, last_updated = ?
                WHERE symbol = ?
            """, (price, datetime.now().isoformat(), symbol))
        
        conn.commit()
        conn.close()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return Position(
            symbol=row[0],
            direction=row[1],
            quantity=row[2],
            entry_price=row[3],
            current_price=row[4],
            entry_time=datetime.fromisoformat(row[5]),
            entry_signal=row[6],
            portfolio_weight=row[7]
        )
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM positions")
        rows = cursor.fetchall()
        conn.close()
        
        positions = {}
        for row in rows:
            position = Position(
                symbol=row[0],
                direction=row[1],
                quantity=row[2],
                entry_price=row[3],
                current_price=row[4],
                entry_time=datetime.fromisoformat(row[5]),
                entry_signal=row[6],
                portfolio_weight=row[7]
            )
            positions[row[0]] = position
        
        return positions
    
    def record_signal(self, symbol: str, signal: float, price: float):
        """Record signal history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO signal_history (symbol, timestamp, signal, price)
            VALUES (?, ?, ?, ?)
        """, (symbol, datetime.now().isoformat(), signal, price))
        
        conn.commit()
        conn.close()
    
    def get_signal_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get signal history for symbol"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        df = pd.read_sql_query("""
            SELECT timestamp, signal, price
            FROM signal_history
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp
        """, conn, params=(symbol, cutoff))
        
        conn.close()
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    def get_trade_history(self, days: Optional[int] = None) -> pd.DataFrame:
        """Get trade history"""
        conn = sqlite3.connect(self.db_path)
        
        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query = """
                SELECT * FROM trades 
                WHERE exit_time > ?
                ORDER BY exit_time DESC
            """
            df = pd.read_sql_query(query, conn, params=(cutoff,))
        else:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY exit_time DESC", conn)
        
        conn.close()
        
        if len(df) > 0:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['holding_period_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
            df['holding_period_days'] = df['holding_period_hours'] / 24
        
        return df
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance metrics"""
        trades = self.get_trade_history()
        positions = self.get_all_positions()
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl_pct': 0,
                'total_pnl_dollars': 0,
                'avg_holding_period_days': 0,
                'current_positions': len(positions),
                'unrealized_pnl': 0
            }
        
        # Realized performance
        win_rate = (trades['realized_pnl_pct'] > 0).sum() / len(trades)
        avg_pnl_pct = trades['realized_pnl_pct'].mean()
        total_pnl_dollars = trades['realized_pnl_dollars'].sum()
        avg_holding_period = trades['holding_period_days'].mean()
        
        # Unrealized P&L
        unrealized_pnl = sum(p.unrealized_pnl_dollars for p in positions.values())
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_pnl_pct': avg_pnl_pct,
            'total_pnl_dollars': total_pnl_dollars,
            'avg_holding_period_days': avg_holding_period,
            'current_positions': len(positions),
            'unrealized_pnl': unrealized_pnl
        }
    
    def print_positions_summary(self):
        """Print formatted summary of current positions"""
        positions = self.get_all_positions()
        
        if len(positions) == 0:
            print("\n📊 No current positions")
            return
        
        print(f"\n{'='*100}")
        print(f"📊 CURRENT POSITIONS ({len(positions)})")
        print(f"{'='*100}")
        print(f"{'Symbol':<8} {'Dir':<6} {'Qty':>8} {'Entry':>10} {'Current':>10} {'P&L %':>10} {'P&L $':>12} {'Age':>10} {'Signal':>8}")
        print(f"{'-'*100}")
        
        total_pnl = 0
        for symbol, pos in sorted(positions.items()):
            total_pnl += pos.unrealized_pnl_dollars
            
            print(f"{symbol:<8} {pos.direction:<6} {pos.quantity:>8.2f} "
                  f"${pos.entry_price:>9.2f} ${pos.current_price:>9.2f} "
                  f"{pos.unrealized_pnl_pct:>9.2%} ${pos.unrealized_pnl_dollars:>11.2f} "
                  f"{pos.age_days:>9.1f}d {pos.entry_signal:>+7.2f}")
        
        print(f"{'-'*100}")
        print(f"{'Total Unrealized P&L:':<80} ${total_pnl:>11.2f}")
        print(f"{'='*100}\n")


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("POSITION TRACKING SYSTEM DEMO")
    print("=" * 80)
    
    # Create tracker
    tracker = PositionTracker("demo_trading.db")
    
    # Add some positions
    print("\n[Adding Positions]")
    
    positions_to_add = [
        Position('AAPL', 'LONG', 100, 180.0, 182.5, datetime.now() - timedelta(hours=48), 0.73, 0.15),
        Position('MSFT', 'LONG', 50, 380.0, 385.0, datetime.now() - timedelta(hours=36), 0.65, 0.13),
        Position('TSLA', 'SHORT', 30, 250.0, 245.0, datetime.now() - timedelta(hours=24), -0.58, 0.12),
    ]
    
    for pos in positions_to_add:
        tracker.add_position(pos)
    
    # Show positions
    tracker.print_positions_summary()
    
    # Update prices
    print("\n[Updating Prices]")
    tracker.update_current_prices({
        'AAPL': 183.0,
        'MSFT': 384.5,
        'TSLA': 246.0
    })
    
    tracker.print_positions_summary()
    
    # Close a position
    print("\n[Closing Position]")
    tracker.remove_position('AAPL', 183.0, 0.68, "Signal weakened")
    
    tracker.print_positions_summary()
    
    # Show performance
    print("\n[Performance Summary]")
    perf = tracker.get_performance_summary()
    print(f"Total Trades:     {perf['total_trades']}")
    print(f"Win Rate:         {perf['win_rate']:.1%}")
    print(f"Avg P&L:          {perf['avg_pnl_pct']:+.2%}")
    print(f"Total P&L:        ${perf['total_pnl_dollars']:+,.2f}")
    print(f"Unrealized P&L:   ${perf['unrealized_pnl']:+,.2f}")
    
    # Show trade history
    print("\n[Trade History]")
    trades = tracker.get_trade_history()
    print(trades[['symbol', 'direction', 'realized_pnl_pct', 'realized_pnl_dollars', 'holding_period_days']])
    
    print("\n✅ Position tracking system working!")