"""
Machine Learning Models for Bitcoin Trading Bot
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import logging

import config
from data_loader import prepare_data_for_ml

logger = logging.getLogger(__name__)

class MLModel:
    """Base class for machine learning models"""
    
    def __init__(self, name, model_type, model_params=None):
        """
        Initialize ML model
        
        Args:
            name (str): Model name
            model_type (str): Model type (e.g., 'rf', 'lgb', 'nn')
            model_params (dict): Model parameters
        """
        self.name = name
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation targets
            
        Returns:
            self: Trained model
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            raise NotImplementedError("Model does not support probability predictions")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Scale features
        X_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def save(self, directory=config.MODELS_DIR):
        """
        Save model to file
        
        Args:
            directory (str): Directory to save model
            
        Returns:
            str: Path to saved model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        model_path = os.path.join(directory, f"{self.name}.joblib")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_params': self.model_params
        }, model_path)
        
        return model_path
    
    def load(self, model_path):
        """
        Load model from file
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            self: Loaded model
        """
        data = joblib.load(model_path)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        self.model_params = data['model_params']
        self.is_trained = True
        
        return self


class RandomForestModel(MLModel):
    """Random Forest Classifier model"""
    
    def __init__(self, name="rf_model", model_params=None):
        """Initialize Random Forest model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Override defaults with provided params
        if model_params:
            default_params.update(model_params)
            
        super().__init__(name, 'rf', default_params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_scaled, y_train)
        
        # Set trained flag
        self.is_trained = True
        
        return self


class LightGBMModel(MLModel):
    """LightGBM Classifier model"""
    
    def __init__(self, name="lgb_model", model_params=None):
        """Initialize LightGBM model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Override defaults with provided params
        if model_params:
            default_params.update(model_params)
            
        super().__init__(name, 'lgb', default_params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = lgb.LGBMClassifier(**self.model_params)
        
        if X_val is not None and y_val is not None:
            # Scale validation data
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train with validation
            self.model.fit(
                X_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric='auc',
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            # Train without validation
            self.model.fit(X_scaled, y_train)
        
        # Set trained flag
        self.is_trained = True
        
        return self


class XGBoostModel(MLModel):
    """XGBoost Classifier model"""
    
    def __init__(self, name="xgb_model", model_params=None):
        """Initialize XGBoost model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Override defaults with provided params
        if model_params:
            default_params.update(model_params)
            
        super().__init__(name, 'xgb', default_params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = xgb.XGBClassifier(**self.model_params)
        
        if X_val is not None and y_val is not None:
            # Scale validation data
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train with validation
            self.model.fit(
                X_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric='auc',
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            # Train without validation
            self.model.fit(X_scaled, y_train)
        
        # Set trained flag
        self.is_trained = True
        
        return self


class NeuralNetworkModel(MLModel):
    """Neural Network model"""
    
    def __init__(self, name="nn_model", model_params=None):
        """Initialize Neural Network model"""
        default_params = {
            'input_dim': None,  # Set during training
            'hidden_layers': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32
        }
        
        # Override defaults with provided params
        if model_params:
            default_params.update(model_params)
            
        super().__init__(name, 'nn', default_params)
    
    def _build_model(self, input_dim):
        """Build neural network model"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.model_params['hidden_layers'][0], input_dim=input_dim, activation='relu'))
        model.add(Dropout(self.model_params['dropout_rate']))
        
        # Hidden layers
        for units in self.model_params['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.model_params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.model_params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Neural Network model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Set input dimension
        self.model_params['input_dim'] = X_train.shape[1]
        
        # Build model
        self.model = self._build_model(self.model_params['input_dim'])
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        
        # Train model
        self.model.fit(
            X_scaled, y_train,
            epochs=self.model_params['epochs'],
            batch_size=self.model_params['batch_size'],
            validation_data=validation_data,
            verbose=0
        )
        
        # Set trained flag
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """Make predictions with Neural Network"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probs = self.model.predict(X_scaled)
        
        # Convert to binary predictions
        return (probs > 0.5).astype(int).flatten()
    
    def save(self, directory=config.MODELS_DIR):
        """Save Neural Network model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save keras model
        model_path = os.path.join(directory, f"{self.name}.h5")
        self.model.save(model_path)
        
        # Save scaler and params
        meta_path = os.path.join(directory, f"{self.name}_meta.joblib")
        joblib.dump({
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_params': self.model_params
        }, meta_path)
        
        return model_path
    
    def load(self, model_path):
        """Load Neural Network model"""
        # Extract base path and filenames
        directory = os.path.dirname(model_path)
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Load keras model
        if model_path.endswith('.h5'):
            self.model = load_model(model_path)
            meta_path = os.path.join(directory, f"{base_name}_meta.joblib")
        else:
            meta_path = model_path
            self.model = load_model(os.path.join(directory, f"{base_name}.h5"))
        
        # Load metadata
        meta = joblib.load(meta_path)
        self.scaler = meta['scaler']
        self.model_type = meta['model_type']
        self.model_params = meta['model_params']
        
        # Set trained flag
        self.is_trained = True
        
        return self


class LSTMModel(MLModel):
    """LSTM model for time series prediction"""
    
    def __init__(self, name="lstm_model", model_params=None):
        """Initialize LSTM model"""
        default_params = {
            'input_dim': None,  # Set during training
            'sequence_length': 10,
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32
        }
        
        # Override defaults with provided params
        if model_params:
            default_params.update(model_params)
            
        super().__init__(name, 'lstm', default_params)
    
    def _build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential()
        
        # Input LSTM layer
        model.add(LSTM(
            self.model_params['lstm_units'][0],
            input_shape=input_shape,
            return_sequences=len(self.model_params['lstm_units']) > 1
        ))
        model.add(Dropout(self.model_params['dropout_rate']))
        
        # Additional LSTM layers
        for i, units in enumerate(self.model_params['lstm_units'][1:]):
            return_sequences = i < len(self.model_params['lstm_units']) - 2
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(self.model_params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.model_params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_sequences(self, X):
        """Prepare sequences for LSTM"""
        seq_length = self.model_params['sequence_length']
        n_features = X.shape[1]
        
        # Create empty sequences
        sequences = []
        for i in range(len(X) - seq_length + 1):
            # Extract sequence
            seq = X[i:i+seq_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train LSTM model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare sequences
        seq_length = self.model_params['sequence_length']
        X_seq = self._prepare_sequences(X_scaled)
        y_seq = y_train[seq_length-1:].values
        
        # Set input shape
        input_shape = (seq_length, X_train.shape[1])
        
        # Build model
        self.model = self._build_model(input_shape)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq = self._prepare_sequences(X_val_scaled)
            y_val_seq = y_val[seq_length-1:].values
            validation_data = (X_val_seq, y_val_seq)
        
        # Train model
        self.model.fit(
            X_seq, y_seq,
            epochs=self.model_params['epochs'],
            batch_size=self.model_params['batch_size'],
            validation_data=validation_data,
            verbose=0
        )
        
        # Set trained flag
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """Make predictions with LSTM"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Prepare sequences
        X_seq = self._prepare_sequences(X_scaled)
        
        # Get probabilities
        probs = self.model.predict(X_seq)
        
        # Convert to binary predictions
        preds = (probs > 0.5).astype(int).flatten()
        
        # Pad predictions to match original length
        padded_preds = np.full(len(X), np.nan)
        padded_preds[self.model_params['sequence_length']-1:] = preds
        
        return padded_preds
    
    def save(self, directory=config.MODELS_DIR):
        """Save LSTM model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save keras model
        model_path = os.path.join(directory, f"{self.name}.h5")
        self.model.save(model_path)
        
        # Save scaler and params
        meta_path = os.path.join(directory, f"{self.name}_meta.joblib")
        joblib.dump({
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_params': self.model_params
        }, meta_path)
        
        return model_path
    
    def load(self, model_path):
        """Load LSTM model"""
        # Extract base path and filenames
        directory = os.path.dirname(model_path)
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Load keras model
        if model_path.endswith('.h5'):
            self.model = load_model(model_path)
            meta_path = os.path.join(directory, f"{base_name}_meta.joblib")
        else:
            meta_path = model_path
            self.model = load_model(os.path.join(directory, f"{base_name}.h5"))
        
        # Load metadata
        meta = joblib.load(meta_path)
        self.scaler = meta['scaler']
        self.model_type = meta['model_type']
        self.model_params = meta['model_params']
        
        # Set trained flag
        self.is_trained = True
        
        return self


class EnsembleModel:
    """Ensemble of multiple models"""
    
    def __init__(self, name="ensemble_model", models=None, weights=None):
        """
        Initialize Ensemble model
        
        Args:
            name (str): Model name
            models (list): List of MLModel instances
            weights (list): List of weights for each model
        """
        self.name = name
        self.models = models or []
        
        # If weights not provided, use equal weighting
        if weights is None and models:
            self.weights = [1/len(models)] * len(models)
        else:
            self.weights = weights or []
    
    def add_model(self, model, weight=None):
        """
        Add model to ensemble
        
        Args:
            model (MLModel): Model to add
            weight (float): Weight for the model
        """
        self.models.append(model)
        
        # Adjust weights
        if weight is None:
            # Equal weighting
            n_models = len(self.models)
            self.weights = [1/n_models] * n_models
        else:
            # Add weight and normalize
            self.weights.append(weight)
            self.weights = [w/sum(self.weights) for w in self.weights]
    
    def predict(self, X):
        """
        Make predictions with ensemble
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.models:
            raise RuntimeError("Ensemble has no models")
        
        # Get predictions from each model
        all_preds = []
        for model in self.models:
            preds = model.predict(X)
            all_preds.append(preds)
        
        # Weighted average of predictions
        weighted_preds = np.zeros(len(X))
        for i, (preds, weight) in enumerate(zip(all_preds, self.weights)):
            weighted_preds += preds * weight
        
        # Convert to binary predictions
        binary_preds = (weighted_preds > 0.5).astype(int)
        
        return binary_preds
    
    def predict_proba(self, X):
        """
        Predict probabilities with ensemble
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.models:
            raise RuntimeError("Ensemble has no models")
        
        # Get prediction probabilities from each model
        all_probs = []
        for model in self.models:
            try:
                probs = model.predict_proba(X)[:, 1]  # Probability of positive class
            except:
                # If model doesn't support predict_proba, use predict
                probs = model.predict(X)
            
            all_probs.append(probs)
        
        # Weighted average of probabilities
        weighted_probs = np.zeros(len(X))
        for i, (probs, weight) in enumerate(zip(all_probs, self.weights)):
            weighted_probs += probs * weight
        
        return weighted_probs
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble performance
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.models:
            raise RuntimeError("Ensemble has no models")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def save(self, directory=config.MODELS_DIR):
        """
        Save ensemble to file
        
        Args:
            directory (str): Directory to save model
            
        Returns:
            str: Path to saved model
        """
        if not self.models:
            raise RuntimeError("Ensemble has no models")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each model
        model_paths = []
        for i, model in enumerate(self.models):
            model_name = f"{self.name}_model{i}_{model.name}"
            model.name = model_name
            path = model.save(directory)
            model_paths.append(path)
        
        # Save ensemble metadata
        ensemble_path = os.path.join(directory, f"{self.name}.joblib")
        joblib.dump({
            'name': self.name,
            'model_paths': model_paths,
            'weights': self.weights
        }, ensemble_path)
        
        return ensemble_path
    
    def load(self, ensemble_path):
        """
        Load ensemble from file
        
        Args:
            ensemble_path (str): Path to ensemble file
            
        Returns:
            self: Loaded ensemble
        """
        # Load ensemble metadata
        data = joblib.load(ensemble_path)
        
        self.name = data['name']
        self.weights = data['weights']
        self.models = []
        
        # Load each model
        for path in data['model_paths']:
            # Determine model type from filename
            filename = os.path.basename(path)
            
            if 'rf_model' in filename:
                model = RandomForestModel()
            elif 'lgb_model' in filename:
                model = LightGBMModel()
            elif 'xgb_model' in filename:
                model = XGBoostModel()
            elif 'lstm_model' in filename:
                model = LSTMModel()
            elif 'nn_model' in filename:
                model = NeuralNetworkModel()
            else:
                # Default to Random Forest
                model = RandomForestModel()
            
            # Load model
            model.load(path)
            self.models.append(model)
        
        return self


def train_ensemble_model(X, y, feature_columns, test_size=0.2, random_state=42):
    """
    Train ensemble model with different types of models
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        feature_columns (list): Feature column names
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed
        
    Returns:
        tuple: (ensemble_model, evaluation_metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_columns], y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Train individual models
    models = []
    
    # Random Forest
    rf_model = RandomForestModel(name="rf_model")
    rf_model.train(X_train, y_train)
    models.append(rf_model)
    
    # LightGBM
    lgb_model = LightGBMModel(name="lgb_model")
    lgb_model.train(X_train, y_train)
    models.append(lgb_model)
    
    # XGBoost
    xgb_model = XGBoostModel(name="xgb_model")
    xgb_model.train(X_train, y_train)
    models.append(xgb_model)
    
    # Create and train ensemble
    ensemble = EnsembleModel(name="btc_ensemble", models=models)
    
    # Evaluate ensemble
    metrics = ensemble.evaluate(X_test, y_test)
    
    # Save models and ensemble
    ensemble.save()
    
    return ensemble, metrics