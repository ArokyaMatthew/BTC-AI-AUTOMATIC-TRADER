from .base_strategy import BaseStrategy
from .technical_strategy import TechnicalStrategy
from .ml_strategy import MLStrategy
from .hybrid_strategy import HybridStrategy

def get_strategy(name, model_path=None, parameters=None):
    """
    Factory function to get strategy instance
    
    Args:
        name (str): Strategy name ('technical', 'ml', 'hybrid')
        model_path (str): Path to ML model
        parameters (dict): Strategy parameters
        
    Returns:
        BaseStrategy: Strategy instance
    """
    if name.lower() == 'technical':
        return TechnicalStrategy(parameters)
    elif name.lower() == 'ml':
        return MLStrategy(model_path, parameters)
    elif name.lower() == 'hybrid':
        return HybridStrategy(model_path, parameters)
    else:
        raise ValueError(f"Unknown strategy: {name}")