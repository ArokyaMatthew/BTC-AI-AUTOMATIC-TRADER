"""
Configuration file for Bitcoin Trading Bot
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')
USE_TESTNET = os.getenv('USE_TESTNET', 'True').lower() in ('true', '1', 't')

# Trading Parameters
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'  # 1 minute candles
ANALYSIS_TIMEFRAMES = ['1m', '5m', '15m', '1h']  # Multiple timeframes for analysis
MAX_POSITIONS = 3  # Maximum number of open positions
TRADE_AMOUNT = 0.001  # Amount of BTC to trade per position
STOP_LOSS_PERCENT = 0.5  # Default stop loss percentage
TAKE_PROFIT_PERCENT = 1.5  # Default take profit percentage

# Profit Maximizer Settings
TARGET_PROFIT_PCT = 2000  # Target profit percentage
MAX_RISK_PCT = 25  # Maximum acceptable risk percentage
DYNAMIC_POSITION_SIZING = True  # Use dynamic position sizing
MAX_POSITION_SIZE_PCT = 50  # Maximum position size as percentage of capital
MAX_RISK_PER_TRADE_PCT = 2  # Maximum risk per trade as percentage of capital

# Moving Average Parameters
SHORT_MA = 9
MEDIUM_MA = 21
LONG_MA = 50
EMA_PERIOD = 20

# RSI Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands Parameters
BB_PERIOD = 20
BB_STD = 2

# Trading Strategy Parameters
STRATEGY = 'hybrid'  # Options: 'ml_ensemble', 'technical', 'hybrid'
ML_FEATURES = [
    'ma_short', 'ma_medium', 'ma_long', 'ema', 
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
    'obv', 'cci', 'adx', 'stoch_k', 'stoch_d',
    'previous_return_1', 'previous_return_2', 'previous_return_3',
    'volume', 'volume_ma', 'price_change_pct', 'volatility'
]

# Paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
LOG_DIR = 'logs'

# Dashboard Configuration
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 8001
DEBUG = True

# Logging Configuration
LOG_LEVEL = 'INFO'