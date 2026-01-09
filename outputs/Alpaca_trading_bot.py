"""
Alpaca Trading Bot - Live Trading Integration

Eventually will try and connect trading strategy to Alpaca Sandbox. 
###This code below is not finished and was an early iteration. Will update later
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import schedule

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    print("⚠️  Alpaca SDK not installed. Install with: pip install alpaca-py")
    print("⚠️  Running in simulation mode only")

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from outputs.Position_tracker import Position, PositionTracker
from outputs.Smart_rebalancer import SmartRebalancer, RebalanceDecision
from outputs.Feature_engineering_positions import (
    create_features_with_positions, 
    Position as FeaturePosition
)
from Models.config import get_production_config
from Models.Ensemble_model import get_ensemble_model


class AlpacaTradingBot:
    """
    Automated trading bot using Alpaca API
    
    Runs daily at market open, generates signals, and rebalances portfolio
    """
    
    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 paper: bool = True,
                 model_path: str = "models/best_model.pt",
                 max_positions: int = 10,
                 max_position_weight: float = 0.15,
                 transaction_cost_bps: float = 10.0,
                 dry_run: bool = False,
                 universe_symbols: Optional[List[str]] = None):
        """
        Initialize Alpaca trading bot
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: Use paper trading (True) or live (False)
            model_path: Path to trained model
            max_positions: Maximum number of positions
            max_position_weight: Max weight per position
            transaction_cost_bps: Transaction costs in basis points
            dry_run: If True, don't execute actual trades
            universe_symbols: List of symbols to trade (None = use all)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.model_path = Path(model_path)
        self.dry_run = dry_run
        self.universe_symbols = universe_symbols or self._get_default_universe()
        
        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self._init_alpaca_clients()
        
        # Initialize systems
        self.position_tracker = PositionTracker("alpaca_positions.db")
        self.rebalancer = SmartRebalancer(
            max_positions=max_positions,
            max_position_weight=max_position_weight,
            transaction_cost_bps=transaction_cost_bps
        )
        
        # Load model
        self.model = None
        self.config = None
        self._load_model()
        
        print(f"\n{'='*100}")
        print(f"ALPACA TRADING BOT INITIALIZED")
        print(f"{'='*100}")
        print(f"Mode:           {'PAPER' if paper else 'LIVE'}")
        print(f"Dry Run:        {dry_run}")
        print(f"Max Positions:  {max_positions}")
        print(f"Universe:       {len(self.universe_symbols)} symbols")
        print(f"Model:          {self.model_path}")
        print(f"{'='*100}\n")
    
    def _get_default_universe(self) -> List[str]:
        """Get default trading universe"""
        # Top 100 liquid stocks
        return [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
            'ORCL', 'ADBE', 'NFLX', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC',
            # Finance
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB',
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'DHR', 'ABT', 'MRK', 'LLY', 'BMY',
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'COST', 'PG',
            # Industrials
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'UNP',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
            # Communications
            'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'NFLX', 'CHTR', 'ATVI', 'EA', 'TTWO',
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'DOW', 'DD', 'PPG', 'NUE',
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'PEG'
        ]
    
    def _init_alpaca_clients(self):
        """Initialize Alpaca API clients"""
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper
            )
            
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            
            # Test connection
            account = self.trading_client.get_account()
            print(f"✓ Connected to Alpaca")
            print(f"  Account: {account.account_number}")
            print(f"  Cash: ${float(account.cash):,.2f}")
            print(f"  Equity: ${float(account.equity):,.2f}")
            
        except Exception as e:
            print(f"⚠️  Failed to connect to Alpaca: {e}")
            print(f"⚠️  Running in simulation mode")
            self.trading_client = None
            self.data_client = None
    
    def _load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            print(f"⚠️  Model not found at {self.model_path}")
            print(f"⚠️  Please train model first or provide correct path")
            return
        
        try:
            self.config = get_production_config()
            self.model = get_ensemble_model(self.config)
            
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✓ Loaded model from {self.model_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            self.model = None
    
    def fetch_market_data(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data from Alpaca
        
        Args:
            symbols: List of symbols
            days: Number of days of historical data
        
        Returns:
            Dict of {symbol: DataFrame with OHLCV}
        """
        if self.data_client is None:
            print("⚠️  No data client available, using simulated data")
            return self._get_simulated_data(symbols, days)
        
        print(f"\n[Fetching Market Data]")
        print(f"  Symbols: {len(symbols)}")
        print(f"  Period: {days} days")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = {}
        
        # Fetch in batches of 10 to avoid rate limits
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Hour,
                    start=start_date,
                    end=end_date
                )
                
                bars = self.data_client.get_stock_bars(request)
                
                for symbol in batch:
                    if symbol in bars.data:
                        df = bars.data[symbol].df
                        df = df.rename(columns={
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'volume': 'volume'
                        })
                        df = df.reset_index()
                        data[symbol] = df
                        print(f"  ✓ {symbol}: {len(df)} bars")
                
                time.sleep(0.2)  # Rate limit protection
                
            except Exception as e:
                print(f"  ⚠️  Failed to fetch {batch}: {e}")
        
        print(f"✓ Fetched data for {len(data)} symbols\n")
        return data
    
    def _get_simulated_data(self, symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """Generate simulated data for testing"""
        data = {}
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            price = 100 + np.random.randn() * 50
            prices = [price]
            
            for _ in range(len(dates) - 1):
                price *= (1 + np.random.normal(0, 0.01))
                prices.append(price)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * 1.005 for p in prices],
                'low': [p * 0.995 for p in prices],
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            })
            
            data[symbol] = df
        
        return data
    
    def generate_signals(self) -> Dict[str, float]:
        """
        Generate trading signals for all symbols
        
        Returns:
            Dict of {symbol: signal} where signal is -1 to +1
        """
        if self.model is None:
            print("⚠️  No model available")
            return {}
        
        print(f"\n[Generating Signals]")
        
        # Fetch data
        market_data = self.fetch_market_data(self.universe_symbols, days=30)
        
        # Get current positions
        current_positions = self.position_tracker.get_all_positions()
        
        signals = {}
        
        for symbol, df in market_data.items():
            try:
                # Get position if exists
                position = current_positions.get(symbol)
                
                # Convert to FeaturePosition for feature engineering
                feature_position = None
                if position:
                    feature_position = FeaturePosition(
                        symbol=position.symbol,
                        direction=position.direction,
                        quantity=position.quantity,
                        entry_price=position.entry_price,
                        current_price=position.current_price,
                        entry_time=position.entry_time,
                        entry_signal=position.entry_signal,
                        portfolio_weight=position.portfolio_weight
                    )
                
                # Create features
                features = create_features_with_positions(
                    df,
                    symbol=symbol,
                    position=feature_position
                )
                
                # Generate signal using model
                with torch.no_grad():
                    # Take last sequence
                    feature_cols = [c for c in features.columns if c not in ['symbol', 'timestamp']]
                    X = features[feature_cols].values[-252:]  # Last 252 hours
                    
                    if len(X) < 252:
                        continue
                    
                    X_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
                    
                    prediction, metadata = self.model(X_tensor)
                    signal = prediction.item()
                    
                    signals[symbol] = signal
                    
                    # Record signal
                    current_price = df['close'].iloc[-1]
                    self.position_tracker.record_signal(symbol, signal, current_price)
            
            except Exception as e:
                print(f"  ⚠️  Failed to generate signal for {symbol}: {e}")
                continue
        
        print(f"✓ Generated {len(signals)} signals")
        print(f"  Signal range: [{min(signals.values()):.3f}, {max(signals.values()):.3f}]")
        
        return signals
    
    def run_daily_rebalance(self):
        """Run daily rebalancing routine"""
        print(f"\n{'='*100}")
        print(f"DAILY REBALANCE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}\n")
        
        # 1. Get current positions from tracker
        current_positions = self.position_tracker.get_all_positions()
        print(f"Current positions: {len(current_positions)}")
        
        # 2. Update current prices from Alpaca
        if current_positions and self.trading_client:
            symbols = list(current_positions.keys())
            latest_prices = self._get_latest_prices(symbols)
            self.position_tracker.update_current_prices(latest_prices)
            
            # Update position objects
            current_positions = self.position_tracker.get_all_positions()
        
        # 3. Show current positions
        if current_positions:
            self.position_tracker.print_positions_summary()
        
        # 4. Generate new signals
        new_signals = self.generate_signals()
        
        if not new_signals:
            print("⚠️  No signals generated, aborting rebalance")
            return
        
        # 5. Run smart rebalancing
        decision = self.rebalancer.rebalance(current_positions, new_signals)
        
        # 6. Execute trades
        if not self.dry_run:
            self._execute_rebalance(decision, new_signals)
        else:
            print("\n[DRY RUN MODE - No trades executed]")
        
        # 7. Show final state
        print("\n[Final Portfolio State]")
        self.position_tracker.print_positions_summary()
        
        # 8. Show performance
        perf = self.position_tracker.get_performance_summary()
        print(f"\n[Performance Summary]")
        print(f"  Total Trades:     {perf['total_trades']}")
        print(f"  Win Rate:         {perf['win_rate']:.1%}")
        print(f"  Avg P&L:          {perf['avg_pnl_pct']:+.2%}")
        print(f"  Total P&L:        ${perf['total_pnl_dollars']:+,.2f}")
        print(f"  Unrealized P&L:   ${perf['unrealized_pnl']:+,.2f}")
        
        print(f"\n{'='*100}")
        print(f"REBALANCE COMPLETE")
        print(f"{'='*100}\n")
    
    def _get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices from Alpaca"""
        if self.trading_client is None:
            # Simulated prices
            return {s: 100 + np.random.randn() * 10 for s in symbols}
        
        try:
            positions = self.trading_client.get_all_positions()
            prices = {}
            for pos in positions:
                if pos.symbol in symbols:
                    prices[pos.symbol] = float(pos.current_price)
            return prices
        except Exception as e:
            print(f"⚠️  Failed to get latest prices: {e}")
            return {}
    
    def _execute_rebalance(self, decision: RebalanceDecision, signals: Dict[str, float]):
        """Execute rebalancing trades"""
        if self.trading_client is None:
            print("⚠️  No trading client, cannot execute trades")
            return
        
        print(f"\n[Executing Trades]")
        
        # Get account info
        account = self.trading_client.get_account()
        buying_power = float(account.buying_power)
        
        # Execute exits first
        for action in decision.get_exits():
            try:
                position = self.position_tracker.get_position(action.symbol)
                if position:
                    # Close position
                    self._close_position(position, action.signal)
                    print(f"  ✓ Closed {action.symbol}")
            except Exception as e:
                print(f"  ⚠️  Failed to close {action.symbol}: {e}")
        
        # Execute new positions
        for action in decision.get_adds():
            try:
                # Calculate position size
                target_value = buying_power * action.target_weight
                latest_price = self._get_latest_prices([action.symbol]).get(action.symbol)
                
                if latest_price:
                    quantity = int(target_value / latest_price)
                    
                    if quantity > 0:
                        self._open_position(action.symbol, action.signal, quantity, latest_price)
                        print(f"  ✓ Opened {action.symbol}: {quantity} shares @ ${latest_price:.2f}")
            except Exception as e:
                print(f"  ⚠️  Failed to open {action.symbol}: {e}")
        
        print(f"✓ Trade execution complete\n")
    
    def _open_position(self, symbol: str, signal: float, quantity: int, price: float):
        """Open new position"""
        if self.trading_client is None:
            return
        
        # Determine direction
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        direction = 'LONG' if signal > 0 else 'SHORT'
        
        # Submit order
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.trading_client.submit_order(order_data)
        
        # Add to tracker
        position = Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now(),
            entry_signal=signal,
            portfolio_weight=0.1  # Will be updated
        )
        
        self.position_tracker.add_position(position)
    
    def _close_position(self, position: Position, exit_signal: float):
        """Close existing position"""
        if self.trading_client is None:
            return
        
        # Close position via Alpaca
        try:
            self.trading_client.close_position(position.symbol)
        except:
            pass  # Position might already be closed
        
        # Remove from tracker
        self.position_tracker.remove_position(
            position.symbol,
            position.current_price,
            exit_signal,
            "Smart rebalance"
        )
    
    def schedule_daily_rebalance(self, time_str: str = "09:35"):
        """
        Schedule daily rebalancing
        
        Args:
            time_str: Time to run (HH:MM format, EST)
        """
        print(f"\n{'='*100}")
        print(f"SCHEDULING DAILY REBALANCE AT {time_str} EST")
        print(f"{'='*100}\n")
        
        schedule.every().day.at(time_str).do(self.run_daily_rebalance)
        
        print(f"✓ Bot scheduled to run daily at {time_str}")
        print(f"✓ Press Ctrl+C to stop\n")
        
        while True:
            schedule.run_pending()
            time.sleep(60)


# Example usage
if __name__ == "__main__":
    # IMPORTANT: Set your Alpaca API credentials
    API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_API_KEY_HERE")
    API_SECRET = os.getenv("ALPACA_API_SECRET", "YOUR_API_SECRET_HERE")
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  WARNING: Please set your Alpaca API credentials!")
        print("⚠️  Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        print("⚠️  Or edit this file to hardcode them (not recommended for production)\n")
    
    # Create bot
    bot = AlpacaTradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        paper=True,  # Use paper trading
        dry_run=True,  # Don't execute actual trades (set to False when ready)
        max_positions=10,
        universe_symbols=None  # None = use default 100 symbols
    )
    
    # Option 1: Run once
    print("\n[Running single rebalance]")
    bot.run_daily_rebalance()
    
    # Option 2: Schedule daily (uncomment to use)
    # bot.schedule_daily_rebalance(time_str="09:35")