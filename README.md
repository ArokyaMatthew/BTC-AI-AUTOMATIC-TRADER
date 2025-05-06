# Bitcoin AI Trading Bot

A sophisticated, high-performance AI trading bot designed to trade Bitcoin 24/7 with maximum profitability while managing risk.

## Overview

This AI trading bot uses a combination of technical analysis, machine learning, and advanced optimization techniques to generate profitable trading signals for Bitcoin. The bot can operate continuously, analyzing market data, making trading decisions, and executing trades automatically.

### Key Features

- **Multiple Trading Strategies**: Technical, Machine Learning, and Hybrid approaches
- **Advanced ML Models**: Ensemble of Random Forest, LightGBM, XGBoost, and Neural Networks
- **Profit Maximizer**: Optimization targeting 2000% profit with controlled risk
- **Dynamic Position Sizing**: Adaptive trade sizing based on confidence and market conditions
- **Real-time Dashboard**: Visualization of performance and trades
- **Extensive Backtesting**: Validate strategies using historical data from 2010-2025
- **Risk Management**: Advanced stop-loss and take-profit mechanisms

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/btc-trading-bot.git
cd btc-trading-bot
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API credentials:

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
USE_TESTNET=True  # Set to False for real trading
```

## Usage

### Optimize Trading Strategies

To find the most profitable strategy and parameters:

```bash
python main.py optimize --days 180
```

### Run Backtest

To backtest a strategy over a specific period:

```bash
python main.py backtest --strategy hybrid --start 2023-01-01 --end 2023-12-31 --capital 10000
```

### Run Trading Bot

To start the trading bot using the optimized strategy:

```bash
python main.py run --optimize
```

Without running optimization first:

```bash
python main.py run --strategy hybrid
```

### Launch Dashboard

To start the web dashboard:

```bash
python main.py dashboard
```

Then open http://localhost:8001 in your web browser.

## Data Format

The bot works with OHLCV (Open, High, Low, Close, Volume) data in the format:

```
timeOpen;timeClose;timeHigh;timeLow;name;open;high;low;close;volume;marketCap;timestamp
```

Historical data is loaded automatically from Binance or can be provided as CSV files.

## Configuration

The main settings are in `config.py`. Key parameters include:

- `TARGET_PROFIT_PCT`: Target profit percentage (default: 2000%)
- `MAX_RISK_PCT`: Maximum acceptable risk percentage (default: 25%)
- `STRATEGY`: Default trading strategy (options: 'technical', 'ml', 'hybrid')
- `TIMEFRAME`: Trading timeframe (default: '1m')
- `STOP_LOSS_PERCENT`: Default stop loss percentage (default: 0.5%)
- `TAKE_PROFIT_PERCENT`: Default take profit percentage (default: 1.5%)

## Strategy Types

### Technical Strategy

Uses traditional technical indicators such as:
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

### ML Strategy

Uses machine learning models to predict price movements:
- Random Forest
- LightGBM
- XGBoost
- Neural Networks (Deep Learning)

### Hybrid Strategy

Combines technical and machine learning approaches with weighted signals.

## Performance Optimization

The `ProfitMaximizer` class optimizes strategies to achieve the target profit (2000%) while controlling risk. It uses:

1. **Differential Evolution**: Advanced global optimization algorithm
2. **Grid Search**: Exhaustive search over parameter space
3. **Backtesting**: Evaluation with historical data
4. **Monte Carlo Simulation**: Risk and return projections

## Disclaimer

Trading cryptocurrencies involves significant risk of loss and is not suitable for all investors. This software is for educational purposes only and is not financial advice. Always do your own research before trading.

## License

MIT License