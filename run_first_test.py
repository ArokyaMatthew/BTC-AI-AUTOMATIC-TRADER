"""
Quick start script for running a backtest and optimization
"""

import os
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import fetch_historical_data
from trading_bot import TradingBot
from optimizers import ProfitMaximizer
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    for directory in [config.DATA_DIR, config.MODELS_DIR, 'logs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def run_quick_backtest():
    """Run a quick backtest with default settings"""
    print("=== RUNNING QUICK BACKTEST ===")
    print("This will test the hybrid strategy with default parameters on 1 year of data.")
    
    # Create directories
    setup_directories()
    
    # Set dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Create bot
    bot = TradingBot(strategy_name='hybrid', backtest_mode=True)
    
    # Run backtest
    results_df, metrics = bot.run_backtest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        deposit=10000,
        commission=0.001
    )
    
    if metrics:
        print("\n=== BACKTEST RESULTS ===")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Final Value: ${metrics['final_value']:.2f}")
        
        # Plot results
        if results_df is not None:
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot close price
                ax1 = plt.subplot(2, 1, 1)
                ax1.plot(results_df.index, results_df['close'], 'gray', alpha=0.5, label='BTC Price')
                ax1.set_ylabel('Price (USD)')
                ax1.set_title('BTC Price and Signals')
                
                # Plot buy/sell signals
                buy_signals = results_df[results_df['signal'] > 0]
                sell_signals = results_df[results_df['signal'] < 0]
                
                ax1.scatter(buy_signals.index, buy_signals['close'], color='green', label='Buy Signal', marker='^', s=100)
                ax1.scatter(sell_signals.index, sell_signals['close'], color='red', label='Sell Signal', marker='v', s=100)
                ax1.legend()
                
                # Plot strategy returns vs buy and hold
                ax2 = plt.subplot(2, 1, 2)
                ax2.plot(results_df.index, results_df['cum_return'], 'b', label='Buy & Hold')
                ax2.plot(results_df.index, results_df['cum_strategy_return'], 'g', label='Strategy')
                ax2.set_ylabel('Cumulative Return')
                ax2.set_title('Strategy Performance')
                ax2.legend()
                
                plt.tight_layout()
                
                # Save figure
                plot_file = os.path.join(config.DATA_DIR, 'backtest_plot.png')
                plt.savefig(plot_file)
                plt.close()
                
                print(f"\nPerformance plot saved to {plot_file}")
                
            except Exception as e:
                logger.error(f"Error plotting results: {e}")
    
    return results_df, metrics

def run_profit_optimizer():
    """Run the profit maximizer to optimize for 2000% returns"""
    print("\n=== RUNNING PROFIT MAXIMIZER ===")
    print("This will optimize trading strategies to aim for 2000% profit with managed risk.")
    print("This may take several minutes to complete...")
    
    # Create profit maximizer
    optimizer = ProfitMaximizer(
        target_profit_pct=config.TARGET_PROFIT_PCT,
        max_risk_pct=config.MAX_RISK_PCT
    )
    
    # Load 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    data = optimizer.load_data(start_date, end_date)
    if data is None:
        print("Failed to load data for optimization.")
        return None
    
    # Run hyperparameter optimization (faster than full optimization)
    best_params = optimizer.run_hyperparameter_optimization('hybrid')
    
    if best_params:
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Best Parameters: {best_params}")
        
        # Calculate expected profit
        expectations = optimizer.calculate_profit_expectations()
        
        if expectations:
            print("\n=== PROFIT EXPECTATIONS ===")
            print(f"Expected Daily Return: {expectations['daily_return']:.4%}")
            print(f"Expected 1-Month Return: {expectations['expectations']['30_days']['expected_return']:.2%}")
            print(f"Expected 1-Year Return: {expectations['expectations']['365_days']['expected_return']:.2%}")
            print(f"Expected 5-Year Return: {expectations['expectations']['1825_days']['expected_return']:.2%}")
            print(f"Probability of Achieving 2000% in 5 Years: {expectations['expectations']['1825_days']['prob_achieving_target']:.2%}")
    
    return best_params

if __name__ == "__main__":
    print("====== BITCOIN AI TRADING BOT - QUICK START ======")
    print("This script will run a backtest and optimization to show the potential of the trading bot.")
    
    input("Press Enter to start...")
    
    # Run backtest
    run_quick_backtest()
    
    # Ask if user wants to run optimization
    run_opt = input("\nDo you want to run the profit optimizer? (y/n): ")
    if run_opt.lower() == 'y':
        run_profit_optimizer()
    
    print("\n====== QUICK START COMPLETED ======")
    print("To run the full trading bot, use: python main.py run --optimize")
    print("To start the dashboard, use: python main.py dashboard")
    print("For more options, see: python main.py --help")