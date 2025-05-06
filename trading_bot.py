"""
Trading Bot for Bitcoin Trading
"""

import os
import time
import pandas as pd
import numpy as np
import ccxt
import logging
from datetime import datetime, timedelta
import json
import threading

import config
from data_loader import fetch_historical_data, create_exchange
from indicators.technical import add_all_indicators
from strategies import get_strategy

logger = logging.getLogger(__name__)

class TradingBot:
    """Trading Bot for automatic trading"""
    
    def __init__(self, strategy_name='hybrid', model_path=None, 
                 parameters=None, testnet=True, backtest_mode=False):
        """
        Initialize Trading Bot
        
        Args:
            strategy_name (str): Strategy name ('technical', 'ml', 'hybrid')
            model_path (str): Path to ML model
            parameters (dict): Strategy parameters
            testnet (bool): Whether to use testnet for trading
            backtest_mode (bool): Whether to run in backtest mode
        """
        self.strategy_name = strategy_name
        self.model_path = model_path
        self.parameters = parameters or {}
        self.testnet = testnet
        self.backtest_mode = backtest_mode
        
        # Initialize strategy
        self.strategy = get_strategy(strategy_name, model_path, parameters)
        
        # Initialize exchange connection if not in backtest mode
        self.exchange = None
        if not backtest_mode:
            self.exchange = create_exchange(testnet)
            
        # Initialize state variables
        self.current_data = None
        self.current_position = 0  # 0 = no position, 1 = long, -1 = short
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trade_history = []
        self.running = False
        self.last_update_time = None
        
        # Load trade history if exists
        self._load_trade_history()
        
    def _load_trade_history(self):
        """Load trade history from file"""
        history_file = os.path.join(config.DATA_DIR, 'trade_history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} trades from history")
            except Exception as e:
                logger.error(f"Error loading trade history: {e}")
                self.trade_history = []
    
    def _save_trade_history(self):
        """Save trade history to file"""
        history_file = os.path.join(config.DATA_DIR, 'trade_history.json')
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def update_data(self, timeframe=config.TIMEFRAME, limit=1000):
        """
        Update market data
        
        Args:
            timeframe (str): Timeframe for candles
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: Updated market data
        """
        if self.backtest_mode:
            logger.warning("Cannot update data in backtest mode")
            return self.current_data
        
        try:
            # Fetch latest data
            symbol = config.SYMBOL
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add indicators
            df = add_all_indicators(df)
            
            # Set current data
            self.current_data = df
            self.last_update_time = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return self.current_data
    
    def get_account_balance(self):
        """
        Get account balance
        
        Returns:
            dict: Account balance
        """
        if self.backtest_mode:
            logger.warning("Cannot get account balance in backtest mode")
            return None
        
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None
    
    def get_current_price(self):
        """
        Get current price
        
        Returns:
            float: Current price
        """
        if self.backtest_mode:
            logger.warning("Cannot get current price in backtest mode")
            if self.current_data is not None:
                return self.current_data['close'].iloc[-1]
            return None
        
        try:
            ticker = self.exchange.fetch_ticker(config.SYMBOL)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return None
    
    def place_order(self, order_type, side, amount, price=None, params={}):
        """
        Place an order
        
        Args:
            order_type (str): Order type ('limit', 'market')
            side (str): Order side ('buy', 'sell')
            amount (float): Order amount
            price (float): Order price (for limit orders)
            params (dict): Additional parameters
            
        Returns:
            dict: Order result
        """
        if self.backtest_mode:
            logger.warning("Cannot place orders in backtest mode")
            return None
        
        try:
            symbol = config.SYMBOL
            order = self.exchange.create_order(symbol, order_type, side, amount, price, params)
            logger.info(f"Placed {order_type} {side} order for {amount} {symbol} at {price}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def close_position(self):
        """
        Close current position
        
        Returns:
            dict: Order result
        """
        if self.backtest_mode:
            logger.warning("Cannot close position in backtest mode")
            return None
        
        if self.current_position == 0:
            logger.info("No position to close")
            return None
        
        try:
            symbol = config.SYMBOL
            side = 'sell' if self.current_position > 0 else 'buy'
            amount = abs(self.position_size)
            
            # Place market order to close position
            order = self.place_order('market', side, amount)
            
            if order:
                # Calculate profit/loss
                exit_price = self.get_current_price()
                entry_price = self.entry_price
                pnl = (exit_price - entry_price) * amount if self.current_position > 0 else (entry_price - exit_price) * amount
                
                # Record trade
                trade = {
                    'entry_time': self.entry_time.isoformat() if hasattr(self, 'entry_time') else None,
                    'exit_time': datetime.now().isoformat(),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': 'long' if self.current_position > 0 else 'short',
                    'amount': amount,
                    'pnl': pnl
                }
                self.trade_history.append(trade)
                self._save_trade_history()
                
                # Reset position
                self.current_position = 0
                self.position_size = 0
                self.entry_price = 0
                self.stop_loss = 0
                self.take_profit = 0
                
                logger.info(f"Position closed with P&L: {pnl}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def check_stop_loss_take_profit(self):
        """
        Check if stop loss or take profit has been hit
        
        Returns:
            bool: Whether position was closed
        """
        if self.backtest_mode:
            logger.warning("Cannot check SL/TP in backtest mode")
            return False
        
        if self.current_position == 0:
            return False
        
        current_price = self.get_current_price()
        if current_price is None:
            return False
        
        # Check stop loss
        if self.stop_loss > 0:
            if (self.current_position > 0 and current_price <= self.stop_loss) or \
               (self.current_position < 0 and current_price >= self.stop_loss):
                logger.info(f"Stop loss hit at {current_price}")
                self.close_position()
                return True
        
        # Check take profit
        if self.take_profit > 0:
            if (self.current_position > 0 and current_price >= self.take_profit) or \
               (self.current_position < 0 and current_price <= self.take_profit):
                logger.info(f"Take profit hit at {current_price}")
                self.close_position()
                return True
        
        return False
    
    def execute_signals(self, signals_df):
        """
        Execute trading signals
        
        Args:
            signals_df (pd.DataFrame): DataFrame with signals
            
        Returns:
            bool: Whether execution was successful
        """
        if self.backtest_mode:
            logger.info("Simulating signal execution in backtest mode")
            return True
        
        try:
            # Get latest signal
            latest_signal = signals_df['signal'].iloc[-1]
            
            # Check if we need to change position
            if latest_signal != self.current_position:
                # Close current position if any
                if self.current_position != 0:
                    self.close_position()
                
                # Open new position if signal is not neutral
                if latest_signal != 0:
                    # Calculate position size
                    balance = self.get_account_balance()
                    if balance is None:
                        return False
                    
                    usdt_balance = balance['total'].get('USDT', 0)
                    current_price = self.get_current_price()
                    
                    # Use fixed amount from config or % of balance
                    if hasattr(config, 'TRADE_AMOUNT') and config.TRADE_AMOUNT > 0:
                        amount = config.TRADE_AMOUNT
                    else:
                        # Use 10% of balance
                        amount = (usdt_balance * 0.1) / current_price
                    
                    # Place order
                    side = 'buy' if latest_signal > 0 else 'sell'
                    order = self.place_order('market', side, amount)
                    
                    if order:
                        # Set position details
                        self.current_position = latest_signal
                        self.position_size = amount if latest_signal > 0 else -amount
                        self.entry_price = current_price
                        self.entry_time = datetime.now()
                        
                        # Set stop loss and take profit
                        sl_percent = config.STOP_LOSS_PERCENT / 100
                        tp_percent = config.TAKE_PROFIT_PERCENT / 100
                        
                        if latest_signal > 0:  # Long position
                            self.stop_loss = current_price * (1 - sl_percent)
                            self.take_profit = current_price * (1 + tp_percent)
                        else:  # Short position
                            self.stop_loss = current_price * (1 + sl_percent)
                            self.take_profit = current_price * (1 - tp_percent)
                        
                        logger.info(f"Opened {side} position of {amount} BTC at {current_price}")
                        logger.info(f"Stop loss: {self.stop_loss}, Take profit: {self.take_profit}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing signals: {e}")
            return False
    
    def run_single_iteration(self):
        """
        Run a single iteration of the trading bot
        
        Returns:
            bool: Whether iteration was successful
        """
        try:
            # Update data
            df = self.update_data()
            if df is None:
                logger.error("Failed to update data")
                return False
            
            # Check stop loss / take profit
            if not self.backtest_mode:
                self.check_stop_loss_take_profit()
            
            # Generate signals
            signals_df = self.strategy.generate_signals(df)
            
            # Execute signals
            success = self.execute_signals(signals_df)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
            return False
    
    def run_backtest(self, start_date, end_date=None, deposit=1000, commission=0.001):
        """
        Run backtest
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            deposit (float): Initial deposit
            commission (float): Commission per trade (as percentage)
            
        Returns:
            tuple: (DataFrame with backtest results, performance metrics)
        """
        try:
            # Set backtest mode
            self.backtest_mode = True
            
            # Fetch historical data
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            
            df = fetch_historical_data(
                symbol=config.SYMBOL,
                timeframe=config.TIMEFRAME,
                start_date=start,
                end_date=end
            )
            
            if df.empty:
                logger.error("No data available for backtest")
                return None, None
            
            # Run strategy backtest
            results_df, metrics = self.strategy.backtest(df, deposit, commission)
            
            # Reset backtest mode
            self.backtest_mode = False
            
            return results_df, metrics
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            self.backtest_mode = False
            return None, None
    
    def run(self, interval=60):
        """
        Run trading bot in continuous mode
        
        Args:
            interval (int): Time interval between iterations in seconds
        """
        if self.backtest_mode:
            logger.warning("Cannot run continuous mode in backtest mode")
            return
        
        self.running = True
        
        def _run_loop():
            while self.running:
                try:
                    # Run iteration
                    self.run_single_iteration()
                    
                    # Wait for next iteration
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in run loop: {e}")
                    time.sleep(10)  # Wait before retrying
        
        # Start trading thread
        trading_thread = threading.Thread(target=_run_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        logger.info(f"Trading bot started with {interval}s interval")
        
        return trading_thread
    
    def stop(self):
        """Stop trading bot"""
        self.running = False
        logger.info("Trading bot stopped")
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics from trade history
        
        Returns:
            dict: Performance metrics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit_per_trade': 0,
                'max_profit': 0,
                'max_loss': 0
            }
        
        total_trades = len(self.trade_history)
        profitable_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(trade['pnl'] for trade in self.trade_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        profits = [trade['pnl'] for trade in self.trade_history]
        max_profit = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0
        
        metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_trade': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss
        }
        
        return metrics
    
    def save_state(self):
        """Save bot state to file"""
        state_file = os.path.join(config.DATA_DIR, 'bot_state.json')
        
        state = {
            'strategy_name': self.strategy_name,
            'model_path': self.model_path,
            'parameters': self.parameters,
            'current_position': self.current_position,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None
        }
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info("Bot state saved")
        except Exception as e:
            logger.error(f"Error saving bot state: {e}")
    
    def load_state(self):
        """Load bot state from file"""
        state_file = os.path.join(config.DATA_DIR, 'bot_state.json')
        
        if not os.path.exists(state_file):
            logger.info("No saved state found")
            return False
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.strategy_name = state.get('strategy_name', self.strategy_name)
            self.model_path = state.get('model_path', self.model_path)
            self.parameters = state.get('parameters', self.parameters)
            self.current_position = state.get('current_position', 0)
            self.position_size = state.get('position_size', 0)
            self.entry_price = state.get('entry_price', 0)
            self.stop_loss = state.get('stop_loss', 0)
            self.take_profit = state.get('take_profit', 0)
            
            if state.get('last_update_time'):
                self.last_update_time = datetime.fromisoformat(state['last_update_time'])
            
            # Reinitialize strategy with saved parameters
            self.strategy = get_strategy(self.strategy_name, self.model_path, self.parameters)
            
            logger.info("Bot state loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading bot state: {e}")
            return False