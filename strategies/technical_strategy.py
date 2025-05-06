"""
Technical Strategy for Bitcoin Trading Bot
"""

from indicators.technical import add_all_indicators, generate_signals
from .base_strategy import BaseStrategy

class TechnicalStrategy(BaseStrategy):
    """Technical analysis based strategy"""
    
    def __init__(self, parameters=None):
        """Initialize Technical strategy"""
        default_params = {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'ma_fast': 9,
            'ma_slow': 21,
            'bb_std': 2,
            'macd_signal_threshold': 0.2
        }
        
        # Update default parameters with provided ones
        if parameters:
            default_params.update(parameters)
            
        super().__init__("technical_strategy", default_params)
        
    def generate_signals(self, df):
        """Generate trading signals based on technical indicators"""
        # Make a copy of the DataFrame
        df = df.copy()
        
        # Add indicators
        df = add_all_indicators(df)
        
        # Generate technical signals
        df = generate_signals(df)
        
        # Rename final signal column to match BaseStrategy requirements
        df['signal'] = df['signal_final']
        
        return df