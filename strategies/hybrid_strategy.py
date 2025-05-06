"""
Hybrid Strategy for Bitcoin Trading Bot
"""

import pandas as pd
import numpy as np
import logging

from indicators.technical import add_all_indicators
from .technical_strategy import TechnicalStrategy
from .ml_strategy import MLStrategy
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class HybridStrategy(BaseStrategy):
    """Hybrid strategy combining technical and ML approaches"""
    
    def __init__(self, ml_model_path=None, parameters=None):
        """
        Initialize Hybrid strategy
        
        Args:
            ml_model_path (str): Path to trained ML model
            parameters (dict): Strategy parameters
        """
        default_params = {
            'technical_weight': 0.5,  # Weight for technical signals (0-1)
            'ml_weight': 0.5,        # Weight for ML signals (0-1)
            'signal_threshold': 0.3,  # Threshold for combined signal
            'confirmation_window': 3  # Number of periods for confirmation
        }
        
        # Update default parameters with provided ones
        if parameters:
            default_params.update(parameters)
            
        super().__init__("hybrid_strategy", default_params)
        
        # Initialize sub-strategies
        self.technical_strategy = TechnicalStrategy()
        self.ml_strategy = MLStrategy(model_path=ml_model_path)
        
    def generate_signals(self, df):
        """
        Generate trading signals using hybrid approach
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        # Make a copy of the DataFrame
        df = df.copy()
        
        # Add indicators
        df = add_all_indicators(df)
        
        # Get technical signals
        technical_df = self.technical_strategy.generate_signals(df)
        technical_signals = technical_df['signal']
        
        # Get ML signals if model is available
        if self.ml_strategy.model:
            ml_df = self.ml_strategy.generate_signals(df)
            ml_signals = ml_df['signal']
        else:
            logger.warning("No ML model available, using technical signals only")
            ml_signals = pd.Series(0, index=df.index)
            self.parameters['technical_weight'] = 1.0
            self.parameters['ml_weight'] = 0.0
        
        # Combine signals with weights
        technical_weight = self.parameters['technical_weight']
        ml_weight = self.parameters['ml_weight']
        
        # Normalize weights
        total_weight = technical_weight + ml_weight
        technical_weight = technical_weight / total_weight
        ml_weight = ml_weight / total_weight
        
        # Calculate combined signal
        combined_signal = (
            technical_weight * technical_signals + 
            ml_weight * ml_signals
        )
        
        # Apply threshold
        threshold = self.parameters['signal_threshold']
        df['raw_signal'] = combined_signal
        df['signal'] = np.where(
            combined_signal > threshold, 1,
            np.where(combined_signal < -threshold, -1, 0)
        )
        
        # Apply confirmation filter
        if self.parameters['confirmation_window'] > 1:
            window = self.parameters['confirmation_window']
            
            # Calculate moving average of signal
            df['signal_ma'] = df['signal'].rolling(window=window).mean()
            
            # Only take strong signals
            df['signal'] = np.where(
                df['signal_ma'] > threshold, 1,
                np.where(df['signal_ma'] < -threshold, -1, 0)
            )
        
        return df