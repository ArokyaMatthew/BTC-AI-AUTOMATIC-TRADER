"""
Position Sizer - Advanced position sizing to maximize profits while controlling risk
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PositionSizer:
    """Position sizing strategies for optimizing trade size"""
    
    def __init__(self, initial_capital=1000, max_risk_per_trade=0.02, max_position_size=0.5):
        """
        Initialize PositionSizer
        
        Args:
            initial_capital (float): Initial capital
            max_risk_per_trade (float): Maximum risk per trade as fraction of capital
            max_position_size (float): Maximum position size as fraction of capital
        """
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.current_capital = initial_capital
        
    def fixed_size(self, price, confidence=None, volatility=None):
        """
        Fixed position sizing
        
        Args:
            price (float): Current price
            confidence (float): Signal confidence (not used)
            volatility (float): Price volatility (not used)
            
        Returns:
            float: Position size in BTC
        """
        # Fixed percentage of capital
        position_value = self.current_capital * self.max_position_size
        position_size = position_value / price
        return position_size
    
    def kelly_criterion(self, price, win_rate, profit_factor, confidence=None):
        """
        Kelly criterion position sizing
        
        Args:
            price (float): Current price
            win_rate (float): Historical win rate
            profit_factor (float): Ratio of average win to average loss
            confidence (float): Signal confidence (optional)
            
        Returns:
            float: Position size in BTC
        """
        # Calculate Kelly fraction: f = (bp - q) / b
        # where b = profit_factor, p = win_rate, q = 1 - p
        p = win_rate
        q = 1 - p
        b = profit_factor
        
        kelly_fraction = (b * p - q) / b
        
        # Adjust by confidence if provided
        if confidence is not None:
            kelly_fraction *= confidence
        
        # Often safer to use half Kelly
        half_kelly = kelly_fraction * 0.5
        
        # Ensure within max position size
        position_fraction = min(half_kelly, self.max_position_size)
        
        # Calculate position size
        position_value = self.current_capital * position_fraction
        position_size = position_value / price
        
        return position_size
    
    def volatility_adjusted(self, price, volatility, confidence=None):
        """
        Volatility-adjusted position sizing
        
        Args:
            price (float): Current price
            volatility (float): Price volatility (e.g., ATR)
            confidence (float): Signal confidence (optional)
            
        Returns:
            float: Position size in BTC
        """
        if volatility <= 0:
            return self.fixed_size(price)
        
        # Calculate maximum risk amount
        max_risk_amount = self.current_capital * self.max_risk_per_trade
        
        # Calculate stop loss distance based on volatility
        stop_loss_factor = 1.5  # ATR multiplier for stop loss
        stop_loss_distance = volatility * stop_loss_factor
        
        # Calculate position size to risk the max risk amount
        position_value = max_risk_amount / (stop_loss_distance / price)
        
        # Adjust by confidence if provided
        if confidence is not None:
            position_value *= confidence
        
        # Ensure within max position size
        max_position_value = self.current_capital * self.max_position_size
        position_value = min(position_value, max_position_value)
        
        # Calculate position size in BTC
        position_size = position_value / price
        
        return position_size
    
    def dynamic_adaptive(self, price, confidence, win_rate, profit_factor, volatility, recent_performance=0):
        """
        Dynamic adaptive position sizing combining multiple methods
        
        Args:
            price (float): Current price
            confidence (float): Signal confidence
            win_rate (float): Historical win rate
            profit_factor (float): Ratio of average win to average loss
            volatility (float): Price volatility
            recent_performance (float): Recent performance metric (-1 to 1)
            
        Returns:
            float: Position size in BTC
        """
        # Calculate position sizes using different methods
        fixed = self.fixed_size(price)
        kelly = self.kelly_criterion(price, win_rate, profit_factor, confidence)
        vol_adjusted = self.volatility_adjusted(price, volatility, confidence)
        
        # Combine methods with weights that adapt to recent performance
        # If recent performance is good, increase Kelly and volatility weight
        # If recent performance is poor, favor fixed sizing
        if recent_performance > 0:
            fixed_weight = 0.2
            kelly_weight = 0.4 + (recent_performance * 0.2)
            vol_weight = 0.4 + (recent_performance * 0.2)
        else:
            fixed_weight = 0.6 + abs(recent_performance * 0.2)
            kelly_weight = 0.2 - (recent_performance * 0.1)
            vol_weight = 0.2 - (recent_performance * 0.1)
        
        # Ensure weights sum to 1
        total_weight = fixed_weight + kelly_weight + vol_weight
        fixed_weight /= total_weight
        kelly_weight /= total_weight
        vol_weight /= total_weight
        
        # Calculate weighted average position size
        position_size = (
            fixed_weight * fixed +
            kelly_weight * kelly +
            vol_weight * vol_adjusted
        )
        
        # Apply additional scaling based on confidence
        confidence_factor = 0.5 + (confidence * 0.5)  # Scale 0.5-1.0
        position_size *= confidence_factor
        
        # Apply risk controls
        # Maximum position size
        max_size = (self.current_capital * self.max_position_size) / price
        position_size = min(position_size, max_size)
        
        return position_size
    
    def update_capital(self, new_capital):
        """
        Update current capital
        
        Args:
            new_capital (float): New capital value
        """
        self.current_capital = new_capital