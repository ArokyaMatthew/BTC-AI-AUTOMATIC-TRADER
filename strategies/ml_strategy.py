"""
Machine Learning Strategy for Bitcoin Trading Bot
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging

from indicators.technical import add_all_indicators
from models.ml_models import (
    EnsembleModel, RandomForestModel, LightGBMModel, XGBoostModel
)
from .base_strategy import BaseStrategy
import config

logger = logging.getLogger(__name__)

class MLStrategy(BaseStrategy):
    """Machine Learning based strategy"""
    
    def __init__(self, model_path=None, parameters=None):
        """
        Initialize ML strategy
        
        Args:
            model_path (str): Path to trained model
            parameters (dict): Strategy parameters
        """
        default_params = {
            'prediction_threshold': 0.6,  # Threshold for positive prediction
            'features': config.ML_FEATURES,
            'position_sizing': 'fixed',  # 'fixed', 'kelly', 'prop'
            'confidence_threshold': 0.7,  # Minimum confidence for trading
        }
        
        # Update default parameters with provided ones
        if parameters:
            default_params.update(parameters)
            
        super().__init__("ml_strategy", default_params)
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
    def load_model(self, model_path):
        """
        Load ML model
        
        Args:
            model_path (str): Path to model file
        """
        try:
            # Check file extension to determine model type
            if model_path.endswith('.joblib'):
                # Check if it's an ensemble model
                model_data = joblib.load(model_path)
                if 'name' in model_data and 'model_paths' in model_data:
                    # Ensemble model
                    self.model = EnsembleModel()
                    self.model.load(model_path)
                else:
                    # Individual model
                    if 'model_type' in model_data:
                        model_type = model_data['model_type']
                        if model_type == 'rf':
                            self.model = RandomForestModel()
                        elif model_type == 'lgb':
                            self.model = LightGBMModel()
                        elif model_type == 'xgb':
                            self.model = XGBoostModel()
                        else:
                            # Default to Random Forest
                            self.model = RandomForestModel()
                    else:
                        # Default to Random Forest
                        self.model = RandomForestModel()
                    
                    self.model.load(model_path)
            else:
                raise ValueError(f"Unsupported model file format: {model_path}")
                
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            
    def generate_signals(self, df):
        """
        Generate trading signals based on ML predictions
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        # Make a copy of the DataFrame
        df = df.copy()
        
        # Check if model is loaded
        if not self.model:
            logger.error("No model loaded, cannot generate signals")
            df['signal'] = 0
            return df
        
        # Add technical indicators
        df = add_all_indicators(df)
        
        # Extract features
        feature_cols = self.parameters['features']
        
        # Check if all features are available
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Use available features only
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Drop rows with NaN values in features
        df_clean = df.dropna(subset=feature_cols)
        
        # Get predictions
        try:
            # For classification models
            predictions = self.model.predict(df_clean[feature_cols])
            
            # Try to get probabilities if available
            try:
                probabilities = self.model.predict_proba(df_clean[feature_cols])
                df_clean['prediction_prob'] = probabilities
            except:
                # If probabilities not available, use binary predictions
                df_clean['prediction_prob'] = predictions
            
            # Set predictions in original DataFrame
            df['prediction'] = np.nan
            df['prediction_prob'] = np.nan
            df.loc[df_clean.index, 'prediction'] = predictions
            df.loc[df_clean.index, 'prediction_prob'] = df_clean['prediction_prob']
            
            # Generate signals based on predictions and confidence
            threshold = self.parameters['prediction_threshold']
            confidence_threshold = self.parameters['confidence_threshold']
            
            # Apply thresholds
            df['signal'] = 0  # Default is no position
            
            # Buy signal (long position)
            buy_condition = (
                (df['prediction'] == 1) & 
                (df['prediction_prob'] >= confidence_threshold)
            )
            df.loc[buy_condition, 'signal'] = 1
            
            # Forward fill signals (maintain position until opposite signal)
            df['signal'] = df['signal'].fillna(method='ffill').fillna(0)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            df['signal'] = 0
        
        return df