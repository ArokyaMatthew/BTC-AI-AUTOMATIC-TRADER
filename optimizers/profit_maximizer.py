"""
Profit Maximizer - Advanced optimization for maximizing BTC trading profits
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import itertools
import os
import json
from scipy.optimize import differential_evolution
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_loader import fetch_historical_data
from strategies import get_strategy
from indicators.technical import add_all_indicators
from models.ml_models import (
    EnsembleModel, RandomForestModel, LightGBMModel, XGBoostModel, NeuralNetworkModel, train_ensemble_model
)

logger = logging.getLogger(__name__)

class ProfitMaximizer:
    """
    Advanced optimization for maximizing BTC trading profits with risk management
    """
    
    def __init__(self, target_profit_pct=2000, max_risk_pct=25):
        """
        Initialize ProfitMaximizer
        
        Args:
            target_profit_pct (float): Target profit percentage (e.g., 2000%)
            max_risk_pct (float): Maximum acceptable risk percentage (e.g., 25%)
        """
        self.target_profit_pct = target_profit_pct
        self.max_risk_pct = max_risk_pct
        self.data = None
        self.best_strategy = None
        self.best_parameters = None
        self.best_performance = None
        self.optimization_results = []
        
    def load_data(self, start_date=None, end_date=None, days=365*5, timeframe=config.TIMEFRAME):
        """
        Load historical data for optimization
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            days (int): Number of days to load if start_date not provided
            timeframe (str): Timeframe for candles
            
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            if not start_date:
                end_date = end_date or datetime.now()
                start_date = end_date - timedelta(days=days)
                
            logger.info(f"Loading data from {start_date} to {end_date}...")
            
            # Fetch historical data
            df = fetch_historical_data(
                symbol=config.SYMBOL,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.error("No data available")
                return None
                
            # Add technical indicators
            df = add_all_indicators(df)
            
            # Drop NaN values
            df = df.dropna()
            
            self.data = df
            logger.info(f"Loaded {len(df)} data points")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
            
    def _calculate_objective(self, params, strategy_name, train_df, test_df, param_names=None):
        """
        Calculate objective function for optimization
        
        Args:
            params (list): Parameter values
            strategy_name (str): Strategy name
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            param_names (list): Parameter names
            
        Returns:
            float: Negative sharpe ratio (for minimization)
        """
        try:
            # Convert params to dictionary
            if param_names:
                parameters = dict(zip(param_names, params))
            else:
                parameters = {
                    'technical_weight': params[0],
                    'ml_weight': 1 - params[0],
                    'signal_threshold': params[1],
                    'confirmation_window': int(params[2]),
                    'rsi_overbought': 70 + params[3] * 10,  # 70-80
                    'rsi_oversold': 30 - params[3] * 10,    # 20-30
                }
            
            # Get strategy
            strategy = get_strategy(strategy_name, parameters=parameters)
            
            # Backtest on training data
            train_results_df, train_metrics = strategy.backtest(train_df)
            
            # Backtest on test data
            test_results_df, test_metrics = strategy.backtest(test_df)
            
            # Calculate risk-adjusted return
            sharpe_ratio = test_metrics['sharpe_ratio']
            max_drawdown = test_metrics['max_drawdown']
            total_return = test_metrics['total_return']
            
            # Penalize high drawdowns
            if max_drawdown > self.max_risk_pct/100:
                sharpe_ratio *= (self.max_risk_pct/100) / max_drawdown
            
            # Penalize low returns
            if total_return < self.target_profit_pct/100:
                sharpe_ratio *= total_return / (self.target_profit_pct/100)
            
            # Maximize Sharpe ratio (higher is better)
            # Return negative value for minimization
            return -sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 0
    
    def optimize_strategy_parameters(self, strategy_name='hybrid', train_size=0.7, iterations=50):
        """
        Optimize strategy parameters
        
        Args:
            strategy_name (str): Strategy name ('technical', 'ml', 'hybrid')
            train_size (float): Train/test split ratio
            iterations (int): Number of optimization iterations
            
        Returns:
            dict: Optimized parameters
        """
        if self.data is None:
            logger.error("No data available. Load data first.")
            return None
        
        try:
            # Split data into train and test sets
            train_size_idx = int(len(self.data) * train_size)
            train_df = self.data.iloc[:train_size_idx]
            test_df = self.data.iloc[train_size_idx:]
            
            logger.info(f"Optimizing parameters for {strategy_name} strategy...")
            
            # Define parameter bounds based on strategy
            if strategy_name == 'technical':
                param_names = [
                    'rsi_overbought', 'rsi_oversold', 
                    'ma_fast', 'ma_slow', 
                    'bb_std', 'macd_signal_threshold'
                ]
                bounds = [
                    (65, 85),   # rsi_overbought
                    (15, 35),   # rsi_oversold
                    (5, 20),    # ma_fast
                    (15, 100),  # ma_slow
                    (1.5, 3.0), # bb_std
                    (0.1, 0.3)  # macd_signal_threshold
                ]
            elif strategy_name == 'ml':
                param_names = [
                    'prediction_threshold', 
                    'confidence_threshold',
                    'position_sizing'
                ]
                bounds = [
                    (0.5, 0.8),    # prediction_threshold
                    (0.6, 0.9),    # confidence_threshold
                    (0.0, 1.0)     # position_sizing (continuous value for optimization)
                ]
            else:  # hybrid
                param_names = [
                    'technical_weight',  # ml_weight will be 1 - technical_weight
                    'signal_threshold',
                    'confirmation_window',
                    'rsi_range_factor'   # Used to calculate overbought/oversold
                ]
                bounds = [
                    (0.2, 0.8),    # technical_weight
                    (0.2, 0.6),    # signal_threshold
                    (1, 5),        # confirmation_window
                    (0.0, 1.0)     # rsi_range_factor
                ]
            
            # Run optimization
            result = differential_evolution(
                func=lambda x: self._calculate_objective(x, strategy_name, train_df, test_df, param_names),
                bounds=bounds,
                maxiter=iterations,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                workers=-1  # Use all available cores
            )
            
            # Convert result to parameters dictionary
            optimized_params = dict(zip(param_names, result.x))
            
            # Adjust parameters to proper types
            if 'confirmation_window' in optimized_params:
                optimized_params['confirmation_window'] = int(optimized_params['confirmation_window'])
            
            if strategy_name == 'hybrid':
                # Calculate derived parameters
                optimized_params['ml_weight'] = 1 - optimized_params['technical_weight']
                if 'rsi_range_factor' in optimized_params:
                    factor = optimized_params.pop('rsi_range_factor')
                    optimized_params['rsi_overbought'] = 70 + factor * 10
                    optimized_params['rsi_oversold'] = 30 - factor * 10
            
            # Evaluate final performance on test set
            strategy = get_strategy(strategy_name, parameters=optimized_params)
            _, test_metrics = strategy.backtest(test_df)
            
            # Store results
            optimization_result = {
                'strategy_name': strategy_name,
                'parameters': optimized_params,
                'metrics': test_metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_results.append(optimization_result)
            
            # Update best strategy if this one is better
            if self.best_performance is None or test_metrics['sharpe_ratio'] > self.best_performance.get('sharpe_ratio', 0):
                self.best_strategy = strategy_name
                self.best_parameters = optimized_params
                self.best_performance = test_metrics
            
            logger.info(f"Optimization completed. Best parameters: {optimized_params}")
            logger.info(f"Performance metrics: Total return: {test_metrics['total_return']:.2%}, Max drawdown: {test_metrics['max_drawdown']:.2%}, Sharpe ratio: {test_metrics['sharpe_ratio']:.2f}")
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return None
    
    def _train_ml_model_wrapper(self, model_type, X_train, y_train, X_val, y_val, params=None):
        """
        Train ML model wrapper for parallel training
        
        Args:
            model_type (str): Model type
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation targets
            params (dict): Model parameters
            
        Returns:
            tuple: (model_type, trained_model, metrics)
        """
        try:
            if model_type == 'rf':
                model = RandomForestModel(model_params=params)
            elif model_type == 'lgb':
                model = LightGBMModel(model_params=params)
            elif model_type == 'xgb':
                model = XGBoostModel(model_params=params)
            elif model_type == 'nn':
                model = NeuralNetworkModel(model_params=params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate model
            metrics = model.evaluate(X_val, y_val)
            
            return model_type, model, metrics
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            return model_type, None, None
    
    def optimize_ml_models(self, prediction_horizon=1, use_multiprocessing=True):
        """
        Optimize and train ML models
        
        Args:
            prediction_horizon (int): Number of periods to look ahead for prediction
            use_multiprocessing (bool): Whether to use multiprocessing
            
        Returns:
            EnsembleModel: Trained ensemble model
        """
        if self.data is None:
            logger.error("No data available. Load data first.")
            return None
        
        try:
            # Add indicators
            df = add_all_indicators(self.data)
            
            # Create target
            df['target'] = (df['close'].shift(-prediction_horizon) > df['close']).astype(int)
            
            # Extract features
            feature_cols = config.ML_FEATURES
            
            # Check if all features are available
            missing_features = [col for col in feature_cols if col not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Use available features only
                feature_cols = [col for col in feature_cols if col in df.columns]
            
            # Drop rows with NaN values
            df = df.dropna(subset=feature_cols + ['target'])
            
            # Split data
            train_size = int(len(df) * 0.7)
            val_size = int(len(df) * 0.15)
            
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:train_size+val_size]
            test_df = df.iloc[train_size+val_size:]
            
            X_train = train_df[feature_cols]
            y_train = train_df['target']
            X_val = val_df[feature_cols]
            y_val = val_df['target']
            X_test = test_df[feature_cols]
            y_test = test_df['target']
            
            logger.info(f"Training ML models with {len(feature_cols)} features...")
            
            # Define model types and parameters
            model_configs = [
                ('rf', {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }),
                ('lgb', {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }),
                ('xgb', {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }),
                ('nn', {
                    'hidden_layers': [64, 32],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'epochs': 50,
                    'batch_size': 32
                })
            ]
            
            # Train models
            trained_models = []
            
            if use_multiprocessing:
                # Parallel training
                with ProcessPoolExecutor() as executor:
                    futures = []
                    
                    for model_type, params in model_configs:
                        future = executor.submit(
                            self._train_ml_model_wrapper,
                            model_type, X_train, y_train, X_val, y_val, params
                        )
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        model_type, model, metrics = future.result()
                        if model is not None:
                            trained_models.append(model)
                            logger.info(f"Trained {model_type} model with accuracy: {metrics['accuracy']:.4f}")
            else:
                # Sequential training
                for model_type, params in model_configs:
                    _, model, metrics = self._train_ml_model_wrapper(
                        model_type, X_train, y_train, X_val, y_val, params
                    )
                    if model is not None:
                        trained_models.append(model)
                        logger.info(f"Trained {model_type} model with accuracy: {metrics['accuracy']:.4f}")
            
            # Create ensemble model
            ensemble = EnsembleModel(name="optimized_ensemble", models=trained_models)
            
            # Evaluate ensemble
            metrics = ensemble.evaluate(X_test, y_test)
            logger.info(f"Ensemble model accuracy: {metrics['accuracy']:.4f}")
            
            # Save models
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            model_path = os.path.join(config.MODELS_DIR, "optimized_ensemble.joblib")
            ensemble.save(model_path)
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error optimizing ML models: {e}")
            return None
    
    def optimize_all_strategies(self):
        """
        Optimize all strategies
        
        Returns:
            dict: Best strategy and parameters
        """
        if self.data is None:
            logger.error("No data available. Load data first.")
            return None
        
        try:
            # Optimize each strategy
            strategies = ['technical', 'ml', 'hybrid']
            
            for strategy in strategies:
                logger.info(f"Optimizing {strategy} strategy...")
                self.optimize_strategy_parameters(strategy)
            
            # Train ML models
            ensemble = self.optimize_ml_models()
            
            # Save best strategy results
            results_file = os.path.join(config.DATA_DIR, 'optimization_results.json')
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            results = {
                'best_strategy': self.best_strategy,
                'best_parameters': self.best_parameters,
                'best_performance': self.best_performance,
                'all_results': self.optimization_results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {results_file}")
            logger.info(f"Best strategy: {self.best_strategy}")
            logger.info(f"Best parameters: {self.best_parameters}")
            logger.info(f"Best performance: Total return: {self.best_performance.get('total_return', 0):.2%}, Max drawdown: {self.best_performance.get('max_drawdown', 0):.2%}")
            
            return {
                'strategy': self.best_strategy,
                'parameters': self.best_parameters,
                'performance': self.best_performance,
                'model_path': os.path.join(config.MODELS_DIR, "optimized_ensemble.joblib") if ensemble else None
            }
            
        except Exception as e:
            logger.error(f"Error optimizing all strategies: {e}")
            return None
    
    def run_hyperparameter_optimization(self, strategy_name='hybrid', param_grid=None):
        """
        Run hyperparameter optimization with grid search
        
        Args:
            strategy_name (str): Strategy name
            param_grid (dict): Parameter grid
            
        Returns:
            dict: Best parameters
        """
        if self.data is None:
            logger.error("No data available. Load data first.")
            return None
        
        try:
            # Split data into train and test sets
            train_size = int(len(self.data) * 0.7)
            train_df = self.data.iloc[:train_size]
            test_df = self.data.iloc[train_size:]
            
            # Default parameter grids
            if param_grid is None:
                if strategy_name == 'technical':
                    param_grid = {
                        'rsi_overbought': [65, 70, 75, 80],
                        'rsi_oversold': [20, 25, 30, 35],
                        'ma_fast': [5, 9, 12],
                        'ma_slow': [21, 50, 100],
                        'bb_std': [1.5, 2.0, 2.5, 3.0],
                        'macd_signal_threshold': [0.1, 0.2, 0.3]
                    }
                elif strategy_name == 'ml':
                    param_grid = {
                        'prediction_threshold': [0.5, 0.6, 0.7],
                        'confidence_threshold': [0.6, 0.7, 0.8, 0.9],
                        'position_sizing': ['fixed', 'kelly', 'prop']
                    }
                else:  # hybrid
                    param_grid = {
                        'technical_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
                        'ml_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
                        'signal_threshold': [0.2, 0.3, 0.4, 0.5],
                        'confirmation_window': [1, 2, 3, 4, 5]
                    }
            
            # Generate all combinations
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            combinations = list(itertools.product(*values))
            
            logger.info(f"Running grid search for {strategy_name} with {len(combinations)} combinations...")
            
            best_sharpe = -np.inf
            best_params = None
            best_metrics = None
            
            for i, combination in enumerate(combinations):
                params = dict(zip(keys, combination))
                
                # Skip invalid combinations
                if 'technical_weight' in params and 'ml_weight' in params:
                    if abs(params['technical_weight'] + params['ml_weight'] - 1.0) > 0.01:
                        continue
                
                # Create strategy with parameters
                strategy = get_strategy(strategy_name, parameters=params)
                
                # Backtest on train and test data
                _, train_metrics = strategy.backtest(train_df)
                test_results_df, test_metrics = strategy.backtest(test_df)
                
                # Calculate risk-adjusted return
                sharpe_ratio = test_metrics['sharpe_ratio']
                max_drawdown = test_metrics['max_drawdown']
                total_return = test_metrics['total_return']
                
                # Log progress every 10 combinations
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(combinations)} combinations")
                
                # Update best if better
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_params = params
                    best_metrics = test_metrics
                    logger.info(f"New best: Sharpe={sharpe_ratio:.2f}, Return={total_return:.2%}, Drawdown={max_drawdown:.2%}")
            
            logger.info(f"Grid search completed. Best parameters: {best_params}")
            logger.info(f"Best metrics: {best_metrics}")
            
            # Update best if better than current
            if self.best_performance is None or best_sharpe > self.best_performance.get('sharpe_ratio', 0):
                self.best_strategy = strategy_name
                self.best_parameters = best_params
                self.best_performance = best_metrics
            
            # Store results
            optimization_result = {
                'strategy_name': strategy_name,
                'parameters': best_params,
                'metrics': best_metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_results.append(optimization_result)
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return None
    
    def _calculate_profit_expectation(self, daily_return, days, drawdown_limit):
        """
        Calculate expected profit and risk metrics
        
        Args:
            daily_return (float): Expected daily return
            days (int): Trading period in days
            drawdown_limit (float): Maximum allowable drawdown
            
        Returns:
            dict: Profit expectation metrics
        """
        # Calculate compounded return
        compounded_return = (1 + daily_return) ** days - 1
        
        # Estimate maximum drawdown using Monte Carlo simulation
        n_simulations = 1000
        paths = np.zeros((n_simulations, days))
        
        for i in range(n_simulations):
            # Generate random daily returns
            daily_returns = np.random.normal(daily_return, daily_return * 2, days)
            
            # Calculate cumulative returns
            path = np.cumprod(1 + daily_returns) - 1
            paths[i] = path
        
        # Calculate drawdowns for each path
        drawdowns = np.zeros(n_simulations)
        for i in range(n_simulations):
            peaks = np.maximum.accumulate(paths[i])
            drawdowns[i] = np.max(1 - (1 + paths[i]) / (1 + peaks))
        
        # Calculate expected drawdown
        expected_drawdown = np.mean(drawdowns)
        worst_drawdown = np.max(drawdowns)
        
        # Calculate probability of exceeding drawdown limit
        prob_exceeding_limit = np.mean(drawdowns > drawdown_limit)
        
        # Calculate probability of achieving target return
        target_return = self.target_profit_pct / 100
        prob_achieving_target = np.mean(paths[:, -1] >= target_return)
        
        return {
            'expected_return': compounded_return,
            'expected_drawdown': expected_drawdown,
            'worst_drawdown': worst_drawdown,
            'prob_exceeding_drawdown_limit': prob_exceeding_limit,
            'prob_achieving_target': prob_achieving_target
        }
    
    def calculate_profit_expectations(self, strategy=None, parameters=None):
        """
        Calculate profit expectations for a strategy
        
        Args:
            strategy (str): Strategy name (if None, use best_strategy)
            parameters (dict): Strategy parameters (if None, use best_parameters)
            
        Returns:
            dict: Profit expectation metrics
        """
        strategy_name = strategy or self.best_strategy
        params = parameters or self.best_parameters
        
        if strategy_name is None or params is None:
            logger.error("No strategy or parameters available.")
            return None
        
        try:
            # Use the last 6 months of data for estimation
            recent_data = self.data.iloc[-180:]
            
            # Create strategy
            strategy_obj = get_strategy(strategy_name, parameters=params)
            
            # Backtest on recent data
            results_df, metrics = strategy_obj.backtest(recent_data)
            
            # Calculate daily return
            daily_return = (1 + metrics['total_return']) ** (1/len(results_df)) - 1
            
            # Calculate expectation for different time periods
            expectations = {}
            for days in [30, 90, 180, 365, 365*2, 365*5]:
                expectation = self._calculate_profit_expectation(
                    daily_return, days, self.max_risk_pct/100
                )
                expectations[f'{days}_days'] = expectation
            
            logger.info(f"Profit expectations calculated:")
            logger.info(f"Daily return: {daily_return:.4%}")
            logger.info(f"30-day expectation: {expectations['30_days']['expected_return']:.2%}")
            logger.info(f"1-year expectation: {expectations['365_days']['expected_return']:.2%}")
            logger.info(f"5-year expectation: {expectations['1825_days']['expected_return']:.2%}")
            
            return {
                'daily_return': daily_return,
                'recent_metrics': metrics,
                'expectations': expectations
            }
            
        except Exception as e:
            logger.error(f"Error calculating profit expectations: {e}")
            return None