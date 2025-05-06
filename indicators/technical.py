"""
Technical indicators for Bitcoin Trading Bot using pandas_ta instead of talib
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


def add_all_indicators(df):
    """
    Add all technical indicators to DataFrame using pandas_ta
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Make sure we have the required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in DataFrame")

    # Make sure data is properly sorted
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    
    # Moving Averages
    df['ma_short'] = df.ta.sma(length=9, close='close')
    df['ma_medium'] = df.ta.sma(length=21, close='close')
    df['ma_long'] = df.ta.sma(length=50, close='close')
    df['ema'] = df.ta.ema(length=20, close='close')
    
    # RSI
    df['rsi'] = df.ta.rsi(length=14, close='close')
    
    # MACD
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # Bollinger Bands
    bbands = df.ta.bbands(length=20, std=2)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_middle'] = bbands['BBM_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']
    
    # ATR (Average True Range) - volatility indicator
    df['atr'] = df.ta.atr(length=14)
    
    # OBV (On Balance Volume)
    df['obv'] = df.ta.obv()
    
    # CCI (Commodity Channel Index)
    df['cci'] = df.ta.cci(length=14)
    
    # ADX (Average Directional Index) - trend strength
    adx = df.ta.adx(length=14)
    df['adx'] = adx['ADX_14']
    
    # Stochastic
    stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Price rate of change
    df['price_change_pct'] = df['close'].pct_change()
    
    # Volume indicators
    df['volume_ma'] = df.ta.sma(length=20, close='volume')
    df['volume_change_pct'] = df['volume'].pct_change()
    
    # Historical returns
    df['previous_return_1'] = df['close'].pct_change(1)
    df['previous_return_2'] = df['close'].pct_change(2)
    df['previous_return_3'] = df['close'].pct_change(3)
    
    # Volatility (standard deviation of returns)
    df['volatility'] = df['previous_return_1'].rolling(window=14).std()
    
    # Advanced indicators
    # High-Low indicator
    df['high_low'] = (df['high'] - df['low']) / df['close']
    
    # Ichimoku Cloud components - calculate manually
    df['tenkan_sen'] = ichimoku_conversion_line(df)
    df['kijun_sen'] = ichimoku_base_line(df)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    
    return df


def ichimoku_conversion_line(df, period=9):
    """Calculate Ichimoku Conversion Line"""
    high_period = df['high'].rolling(window=period).max()
    low_period = df['low'].rolling(window=period).min()
    return (high_period + low_period) / 2


def ichimoku_base_line(df, period=26):
    """Calculate Ichimoku Base Line"""
    high_period = df['high'].rolling(window=period).max()
    low_period = df['low'].rolling(window=period).min()
    return (high_period + low_period) / 2


def generate_signals(df):
    """
    Generate trading signals based on technical indicators
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        pd.DataFrame: DataFrame with signal columns
    """
    df = df.copy()
    
    # Initialize signal columns
    df['signal_ma_cross'] = 0  # Moving average crossover
    df['signal_rsi'] = 0       # RSI overbought/oversold
    df['signal_macd'] = 0      # MACD crossover
    df['signal_bb'] = 0        # Bollinger Bands
    df['signal_final'] = 0     # Combined signal
    
    # MA Crossover (Short MA crosses above Medium MA)
    df['signal_ma_cross'] = np.where(
        (df['ma_short'] > df['ma_medium']) & 
        (df['ma_short'].shift(1) <= df['ma_medium'].shift(1)),
        1,  # Buy
        np.where(
            (df['ma_short'] < df['ma_medium']) & 
            (df['ma_short'].shift(1) >= df['ma_medium'].shift(1)),
            -1,  # Sell
            0    # Hold
        )
    )
    
    # RSI (Oversold/Overbought)
    df['signal_rsi'] = np.where(
        (df['rsi'] < 30) & (df['rsi'].shift(1) < 30) & (df['rsi'] > df['rsi'].shift(1)),
        1,  # Buy (RSI moving up from oversold)
        np.where(
            (df['rsi'] > 70) & (df['rsi'].shift(1) > 70) & (df['rsi'] < df['rsi'].shift(1)),
            -1,  # Sell (RSI moving down from overbought)
            0    # Hold
        )
    )
    
    # MACD Crossover
    df['signal_macd'] = np.where(
        (df['macd'] > df['macd_signal']) & 
        (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
        1,  # Buy
        np.where(
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
            -1,  # Sell
            0    # Hold
        )
    )
    
    # Bollinger Bands
    df['signal_bb'] = np.where(
        df['close'] < df['bb_lower'],
        1,  # Buy (price below lower band)
        np.where(
            df['close'] > df['bb_upper'],
            -1,  # Sell (price above upper band)
            0    # Hold
        )
    )
    
    # Combine signals (weighted approach)
    # You can adjust weights based on performance
    df['signal_final'] = (
        0.3 * df['signal_ma_cross'] +
        0.2 * df['signal_rsi'] +
        0.3 * df['signal_macd'] +
        0.2 * df['signal_bb']
    )
    
    # Threshold for final signal
    df['signal_final'] = np.where(
        df['signal_final'] > 0.2,
        1,  # Strong buy
        np.where(
            df['signal_final'] < -0.2,
            -1,  # Strong sell
            0    # Hold
        )
    )
    
    return df