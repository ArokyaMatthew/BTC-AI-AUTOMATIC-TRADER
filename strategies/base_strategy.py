"""
Base Strategy Class for Bitcoin Trading Bot
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

import config
from indicators.technical import add_all_indicators

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base strategy class for trading strategies"""
    
    def __init__(self, name, parameters=None):
        """
        Initialize strategy
        
        Args:
            name (str): Strategy name
            parameters (dict): Strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        
    @abstractmethod
    def generate_signals(self, df):
        """
        Generate trading signals
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        pass
    
    def calculate_metrics(self, df, deposit=1000, commission=0.001):
        """
        Calculate strategy performance metrics
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            deposit (float): Initial deposit
            commission (float): Commission per trade (as percentage)
            
        Returns:
            dict: Performance metrics
        """
        # Make a copy of DataFrame
        df = df.copy()
        
        # Check if signals column exists
        if 'signal' not in df.columns:
            raise ValueError("DataFrame must contain 'signal' column")
        
        # Calculate returns based on signals
        df['next_return'] = df['close'].pct_change(1).shift(-1)
        df['strategy_return'] = df['signal'] * df['next_return']
        
        # Calculate commissions
        df['commission'] = np.where(df['signal'] != df['signal'].shift(1), commission, 0)
        df['net_return'] = df['strategy_return'] - df['commission']
        
        # Calculate cumulative returns
        df['cum_return'] = (1 + df['next_return']).cumprod() - 1
        df['cum_strategy_return'] = (1 + df['net_return']).cumprod() - 1
        
        # Calculate metrics
        total_trades = (df['signal'] != df['signal'].shift(1)).sum()
        profitable_trades = ((df['signal'] != 0) & (df['strategy_return'] > 0)).sum()
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate drawdowns
        df['peak'] = df['cum_strategy_return'].cummax()
        df['drawdown'] = df['peak'] - df['cum_strategy_return']
        max_drawdown = df['drawdown'].max()
        
        # Calculate Sharpe ratio (annualized)
        returns_mean = df['net_return'].mean() * 252  # 252 trading days in a year
        returns_std = df['net_return'].std() * np.sqrt(252)
        sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0
        
        # Calculate final metrics
        final_value = deposit * (1 + df['cum_strategy_return'].iloc[-1])
        total_profit = final_value - deposit
        total_return = total_profit / deposit
        
        metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_profit': total_profit,
            'total_return': total_return,
            'final_value': final_value
        }
        
        return metrics
    
    def backtest(self, df, deposit=1000, commission=0.001):
        """
        Backtest strategy
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            deposit (float): Initial deposit
            commission (float): Commission per trade (as percentage)
            
        Returns:
            tuple: (DataFrame with signals and metrics, performance metrics)
        """
        # Generate signals
        df_with_signals = self.generate_signals(df)
        
        # Calculate metrics
        metrics = self.calculate_metrics(df_with_signals, deposit, commission)
        
        # Log results
        logger.info(f"Backtest results for {self.name} strategy:")
        logger.info(f"Total trades: {metrics['total_trades']}")
        logger.info(f"Win rate: {metrics['win_rate']:.2%}")
        logger.info(f"Total return: {metrics['total_return']:.2%}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        
        return df_with_signals, metrics