"""
Main entry point for Bitcoin Trading Bot
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime, timedelta
import json

import config
from data_loader import fetch_historical_data
from trading_bot import TradingBot
from optimizers import ProfitMaximizer, PositionSizer
from strategies import get_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'bot.log'), mode='a'),
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

def optimize_strategy(days=180):
    """
    Optimize trading strategy
    
    Args:
        days (int): Number of days of historical data to use for optimization
    """
    logger.info(f"Starting strategy optimization with {days} days of data...")
    
    # Create profit maximizer
    optimizer = ProfitMaximizer(
        target_profit_pct=config.TARGET_PROFIT_PCT if hasattr(config, 'TARGET_PROFIT_PCT') else 2000,
        max_risk_pct=config.MAX_RISK_PCT if hasattr(config, 'MAX_RISK_PCT') else 25
    )
    
    # Load historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = optimizer.load_data(start_date, end_date)
    if data is None:
        logger.error("Failed to load data for optimization")
        return None
    
    # Run optimization
    result = optimizer.optimize_all_strategies()
    
    if result:
        logger.info(f"Optimization completed successfully")
        logger.info(f"Best strategy: {result['strategy']}")
        logger.info(f"Parameters: {result['parameters']}")
        
        # Calculate expected profit
        expectations = optimizer.calculate_profit_expectations()
        
        if expectations:
            logger.info(f"Expected daily return: {expectations['daily_return']:.4%}")
            logger.info(f"Expected 1-year return: {expectations['expectations']['365_days']['expected_return']:.2%}")
            logger.info(f"Probability of achieving target: {expectations['expectations']['365_days']['prob_achieving_target']:.2%}")
            
            # Save expectations to file
            expectations_file = os.path.join(config.DATA_DIR, 'profit_expectations.json')
            with open(expectations_file, 'w') as f:
                json.dump(expectations, f, indent=2, default=str)
    
    return result

def run_backtest(strategy_name, parameters, start_date_str, end_date_str=None, initial_capital=10000):
    """
    Run backtest
    
    Args:
        strategy_name (str): Strategy name
        parameters (dict): Strategy parameters
        start_date_str (str): Start date in YYYY-MM-DD format
        end_date_str (str): End date in YYYY-MM-DD format
        initial_capital (float): Initial capital
    """
    logger.info(f"Running backtest for {strategy_name} strategy...")
    
    # Create trading bot in backtest mode
    bot = TradingBot(
        strategy_name=strategy_name,
        parameters=parameters,
        backtest_mode=True
    )
    
    # Run backtest
    results_df, metrics = bot.run_backtest(
        start_date=start_date_str,
        end_date=end_date_str,
        deposit=initial_capital
    )
    
    if metrics:
        logger.info(f"Backtest completed successfully")
        logger.info(f"Total return: {metrics['total_return']:.2%}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Final value: ${metrics['final_value']:.2f}")
        
        # Save results
        if results_df is not None:
            results_file = os.path.join(config.DATA_DIR, f'backtest_results_{datetime.now().strftime("%Y%m%d")}.csv')
            results_df.to_csv(results_file)
            logger.info(f"Results saved to {results_file}")
    
    return results_df, metrics

def run_bot(strategy_name=None, parameters=None, model_path=None, run_optimization=False):
    """
    Run trading bot
    
    Args:
        strategy_name (str): Strategy name
        parameters (dict): Strategy parameters
        model_path (str): Path to ML model
        run_optimization (bool): Whether to run optimization first
    """
    # Run optimization if requested
    if run_optimization:
        logger.info("Running optimization before starting bot...")
        optimization_result = optimize_strategy()
        
        if optimization_result:
            strategy_name = optimization_result['strategy']
            parameters = optimization_result['parameters']
            model_path = optimization_result.get('model_path')
    
    # Load optimized parameters if not provided
    if not parameters:
        results_file = os.path.join(config.DATA_DIR, 'optimization_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                strategy_name = results.get('best_strategy', strategy_name or config.STRATEGY)
                parameters = results.get('best_parameters')
                logger.info(f"Loaded optimized parameters for {strategy_name} strategy")
    
    # Create trading bot
    bot = TradingBot(
        strategy_name=strategy_name or config.STRATEGY,
        model_path=model_path,
        parameters=parameters,
        testnet=config.USE_TESTNET
    )
    
    # Load previous state if available
    bot.load_state()
    
    # Run bot
    logger.info(f"Starting trading bot with {strategy_name} strategy...")
    trading_thread = bot.run()
    
    try:
        # Main loop
        while True:
            time.sleep(60)
            
            # Save state periodically
            bot.save_state()
            
            # Log current position
            if bot.current_position != 0:
                current_price = bot.get_current_price()
                if current_price is not None:
                    entry_price = bot.entry_price
                    pnl = (current_price - entry_price) * bot.position_size if bot.current_position > 0 else (entry_price - current_price) * bot.position_size
                    logger.info(f"Current position: {'Long' if bot.current_position > 0 else 'Short'}, Size: {bot.position_size}, Entry: {entry_price}, Current: {current_price}, P&L: {pnl:.2f}")
    
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
        bot.stop()
        bot.save_state()
        
        # Close position if requested
        close_position = input("Close current position? (y/n): ")
        if close_position.lower() == 'y':
            logger.info("Closing position...")
            bot.close_position()
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        bot.stop()
        bot.save_state()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Bitcoin Trading Bot')
    subparsers = parser.add_subparsers(dest='command')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize trading strategy')
    optimize_parser.add_argument('--days', type=int, default=180, help='Days of historical data to use')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--strategy', type=str, default='hybrid', help='Strategy name')
    backtest_parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run trading bot')
    run_parser.add_argument('--strategy', type=str, help='Strategy name')
    run_parser.add_argument('--optimize', action='store_true', help='Run optimization first')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run dashboard')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    # Execute command
    if args.command == 'optimize':
        optimize_strategy(args.days)
    
    elif args.command == 'backtest':
        # Load optimized parameters if available
        parameters = None
        results_file = os.path.join(config.DATA_DIR, 'optimization_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                if results.get('best_strategy') == args.strategy:
                    parameters = results.get('best_parameters')
        
        run_backtest(args.strategy, parameters, args.start, args.end, args.capital)
    
    elif args.command == 'run':
        run_bot(args.strategy, run_optimization=args.optimize)
    
    elif args.command == 'dashboard':
        # Import here to avoid circular imports
        from dashboard.dashboard import app
        app.run_server(debug=config.DEBUG, host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()