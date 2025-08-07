"""
Strategy Loader for Dynamic Strategy Import

Loads and validates user-defined strategy modules from Python files.
Ensures strategies implement the required interface and extracts metadata.
"""
import importlib.util
import inspect
import os
from typing import Type, Dict, Any, Tuple, List
from .strategy_interface import OptimizableStrategy


class StrategyLoader:
    """
    Loads and validates user-defined strategy modules.
    
    Provides utilities for dynamically importing strategy classes,
    validating their implementation, and extracting optimization parameters.
    """
    
    @staticmethod
    def load_strategy_from_file(strategy_path: str) -> Type[OptimizableStrategy]:
        """
        Load strategy class from Python file.
        
        Args:
            strategy_path: Path to strategy .py file
            
        Returns:
            Strategy class that implements OptimizableStrategy
            
        Raises:
            FileNotFoundError: If strategy file doesn't exist
            ValueError: If no valid strategy class found
            ImportError: If strategy file has import errors
        """
        if not os.path.exists(strategy_path):
            raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
        
        # Load module from file
        spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {strategy_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Error executing strategy module: {e}")
        
        # Find strategy class in module
        strategy_class = StrategyLoader._find_strategy_class(module, strategy_path)
        
        # Validate the strategy class
        StrategyLoader.validate_strategy_class(strategy_class)
        
        return strategy_class
    
    @staticmethod
    def _find_strategy_class(module, strategy_path: str) -> Type[OptimizableStrategy]:
        """Find the strategy class in the loaded module"""
        strategy_classes = []
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, OptimizableStrategy) and 
                obj != OptimizableStrategy):
                strategy_classes.append(obj)
        
        if len(strategy_classes) == 0:
            raise ValueError(f"No strategy class implementing OptimizableStrategy found in {strategy_path}")
        elif len(strategy_classes) > 1:
            class_names = [cls.__name__ for cls in strategy_classes]
            raise ValueError(f"Multiple strategy classes found in {strategy_path}: {class_names}. "
                           f"Please ensure only one strategy class per file.")
        
        return strategy_classes[0]
    
    @staticmethod
    def validate_strategy_class(strategy_class: Type[OptimizableStrategy]) -> bool:
        """
        Validate that strategy class implements required interface correctly.
        
        Args:
            strategy_class: Strategy class to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If strategy class is invalid
        """
        # Check that it's actually a subclass
        if not issubclass(strategy_class, OptimizableStrategy):
            raise ValueError(f"{strategy_class.__name__} must inherit from OptimizableStrategy")
        
        # Check required methods exist
        required_methods = ['should_buy', 'should_sell', 'get_position_size']
        for method_name in required_methods:
            if not hasattr(strategy_class, method_name):
                raise ValueError(f"{strategy_class.__name__} missing required method: {method_name}")
            
            method = getattr(strategy_class, method_name)
            if not callable(method):
                raise ValueError(f"{strategy_class.__name__}.{method_name} is not callable")
        
        # Check required properties exist
        required_properties = ['parameters', 'name', 'strategy_type']
        for prop_name in required_properties:
            if not hasattr(strategy_class, prop_name):
                raise ValueError(f"{strategy_class.__name__} missing required property: {prop_name}")
        
        # Try to create an instance to validate parameters
        try:
            instance = strategy_class()
            
            # Validate parameters property
            parameters = instance.parameters
            if not isinstance(parameters, dict):
                raise ValueError(f"{strategy_class.__name__}.parameters must return a dictionary")
            
            # Validate each parameter definition
            for param_name, param_config in parameters.items():
                if not isinstance(param_config, dict):
                    raise ValueError(f"Parameter '{param_name}' configuration must be a dictionary")
                
                if 'type' not in param_config:
                    raise ValueError(f"Parameter '{param_name}' missing 'type' specification")
                
                if 'default' not in param_config:
                    raise ValueError(f"Parameter '{param_name}' missing 'default' value")
            
            # Validate name and strategy_type
            if not isinstance(instance.name, str) or not instance.name.strip():
                raise ValueError(f"{strategy_class.__name__}.name must be a non-empty string")
            
            if not isinstance(instance.strategy_type, str) or not instance.strategy_type.strip():
                raise ValueError(f"{strategy_class.__name__}.strategy_type must be a non-empty string")
                
        except Exception as e:
            raise ValueError(f"Error validating {strategy_class.__name__}: {e}")
        
        return True
    
    @staticmethod
    def get_parameter_ranges(strategy_class: Type[OptimizableStrategy]) -> Dict[str, Tuple]:
        """
        Extract parameter ranges for optimization.
        
        Args:
            strategy_class: Strategy class to analyze
            
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        instance = strategy_class()  # Create with defaults
        param_ranges = {}
        
        for param_name, param_config in instance.parameters.items():
            if 'range' in param_config:
                param_ranges[param_name] = param_config['range']
        
        return param_ranges
    
    @staticmethod
    def get_parameter_defaults(strategy_class: Type[OptimizableStrategy]) -> Dict[str, Any]:
        """
        Extract default parameter values.
        
        Args:
            strategy_class: Strategy class to analyze
            
        Returns:
            Dictionary mapping parameter names to default values
        """
        instance = strategy_class()  # Create with defaults
        defaults = {}
        
        for param_name, param_config in instance.parameters.items():
            defaults[param_name] = param_config['default']
        
        return defaults
    
    @staticmethod
    def get_strategy_info(strategy_class: Type[OptimizableStrategy]) -> Dict[str, Any]:
        """
        Get comprehensive information about a strategy.
        
        Args:
            strategy_class: Strategy class to analyze
            
        Returns:
            Dictionary with strategy metadata
        """
        instance = strategy_class()
        
        return {
            'class_name': strategy_class.__name__,
            'name': instance.name,
            'strategy_type': instance.strategy_type,
            'parameters': instance.parameters,
            'parameter_ranges': StrategyLoader.get_parameter_ranges(strategy_class),
            'parameter_defaults': StrategyLoader.get_parameter_defaults(strategy_class),
            'docstring': strategy_class.__doc__ or "No description available"
        }
