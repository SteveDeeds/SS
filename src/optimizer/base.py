"""
Base Optimizer Interface

Provides abstract base class for all optimization algorithms.
Defines standard interface for strategy optimization and testing.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Optional
from .strategy_interface import OptimizableStrategy
from .strategy_loader import StrategyLoader


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    
    Provides common functionality and defines the interface that
    all optimizers must implement.
    """
    
    def __init__(self, strategy_path: str):
        """
        Initialize optimizer with a strategy.
        
        Args:
            strategy_path: Path to strategy Python file
        """
        self.strategy_path = strategy_path
        self.strategy_class = StrategyLoader.load_strategy_from_file(strategy_path)
        self.strategy_info = StrategyLoader.get_strategy_info(self.strategy_class)
        self.parameter_ranges = StrategyLoader.get_parameter_ranges(self.strategy_class)
        self.parameter_defaults = StrategyLoader.get_parameter_defaults(self.strategy_class)
        
        print(f"✅ Loaded strategy: {self.strategy_info['name']}")
        print(f"   Type: {self.strategy_info['strategy_type']}")
        print(f"   Parameters: {list(self.parameter_ranges.keys())}")
    
    @abstractmethod
    def optimize(self, historical_data: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Run optimization algorithm on the strategy.
        
        Args:
            historical_data: Historical price data for backtesting
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dictionary containing optimization results
        """
        pass
    
    def create_strategy_instance(self, **params) -> OptimizableStrategy:
        """
        Create strategy instance with given parameters.
        
        Args:
            **params: Parameter values for strategy
            
        Returns:
            Strategy instance configured with given parameters
        """
        # Use defaults for any missing parameters
        full_params = self.parameter_defaults.copy()
        full_params.update(params)
        
        # Validate parameters
        instance = self.strategy_class()
        if not instance.validate_parameters(**full_params):
            raise ValueError(f"Invalid parameters: {params}")
        
        return self.strategy_class(**full_params)
    
    def validate_parameter_set(self, params: Dict[str, Any]) -> bool:
        """
        Validate a set of parameters against strategy requirements.
        
        Args:
            params: Parameter dictionary to validate
            
        Returns:
            True if parameters are valid
        """
        try:
            instance = self.create_strategy_instance(**params)
            return True
        except (ValueError, TypeError):
            return False
    
    def get_parameter_combinations(self, parameter_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from a grid.
        
        Args:
            parameter_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter dictionaries
        """
        import itertools
        
        # Get parameter names and value lists
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        # Generate all combinations
        combinations = []
        for value_combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, value_combo))
            combinations.append(param_dict)
        
        return combinations
    
    def get_strategy_summary(self) -> str:
        """
        Get a formatted summary of the loaded strategy.
        
        Returns:
            Human-readable strategy summary
        """
        info = self.strategy_info
        
        summary = f"Strategy: {info['name']}\n"
        summary += f"Type: {info['strategy_type']}\n"
        summary += f"Class: {info['class_name']}\n\n"
        
        summary += "Parameters:\n"
        for param_name, param_config in info['parameters'].items():
            param_type = param_config['type'].__name__
            default = param_config['default']
            param_range = param_config.get('range', 'No range specified')
            description = param_config.get('description', 'No description')
            
            summary += f"  • {param_name} ({param_type}): {description}\n"
            summary += f"    Default: {default}, Range: {param_range}\n"
        
        if info['docstring'] != "No description available":
            summary += f"\nDescription: {info['docstring']}\n"
        
        return summary
