"""
Data loading module for Bitcoin Trading Bot
"""

import os
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

def create_exchange(testnet=True):
    """
    Create exchange connection
    """
    if config.API_KEY and config.API_SECRET:
        exchange = ccxt.binance({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'testnet': testnet
            }
        })
    else:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'testnet': testnet
            }
        })
    
    return exchange

def fetch_historical_data(symbol=config.SYMBOL, timeframe=config.TIMEFRAME, days=None, start_date=None, end_date=None):
    """
    Fetch historical OHLCV data
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
        timeframe (str): Timeframe for candles (e.g., '1m', '1h', '1d')
        days (int): Number of days of data to fetch
        start_date (datetime): Start date for data fetching
        end_date (datetime): End date for data fetching
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    exchange = create_exchange(testnet=False)  # Use main network for historical data
    
    # Calculate timestamp limits
    if days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    elif not start_date:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Default to 1 year
    
    # Convert dates to millisecond timestamps
    since = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    
    # Initialize data list
    all_candles = []
    
    # Fetch data in chunks to avoid rate limiting
    while since < end_time:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1  # Start from the next timestamp
            
            # Add small delay to prevent rate limit issues
            exchange.sleep(exchange.rateLimit / 1000)
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            exchange.sleep(5000)  # Wait 5 seconds and retry
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    
    # Convert to DataFrame
    if all_candles:
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    return pd.DataFrame()

def load_file_data(filepath):
    """
    Load data from CSV file with the format:
    timeOpen;timeClose;timeHigh;timeLow;name;open;high;low;close;volume;marketCap;timestamp
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = pd.read_csv(filepath, sep=';')
    
    # Convert timestamps to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    elif 'timeOpen' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timeOpen'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
    # Rename columns to standard format if needed
    column_mapping = {
        'timeOpen': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'marketCap': 'marketcap'
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Ensure essential columns are present
    essential_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in essential_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in data file")
            
    # Convert price and volume columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Remove rows with NaN values
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    return df

def prepare_data_for_ml(df, feature_columns, target_column='target', lookahead=1, train_split=0.8):
    """
    Prepare data for machine learning
    
    Args:
        df (pd.DataFrame): DataFrame with features
        feature_columns (list): List of feature column names
        target_column (str): Target column name
        lookahead (int): Number of periods to look ahead for target
        train_split (float): Train/test split ratio
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Create target variable (future price movement)
    df[target_column] = df['close'].shift(-lookahead) / df['close'] - 1
    
    # Filter out rows with NaN in target or features
    df = df.dropna(subset=feature_columns + [target_column])
    
    # Split data
    train_size = int(len(df) * train_split)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Extract features and target
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    
    return X_train, X_test, y_train, y_test

def create_classification_target(df, threshold=0.0, lookahead=1):
    """
    Create classification target based on price movement
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        threshold (float): Threshold for price movement
        lookahead (int): Number of periods to look ahead
        
    Returns:
        pd.Series: Classification target (1 for up, 0 for down)
    """
    future_return = df['close'].shift(-lookahead) / df['close'] - 1
    target = (future_return > threshold).astype(int)
    return target