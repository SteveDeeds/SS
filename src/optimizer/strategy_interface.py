"""
Strategy Interface for New Optimizer Architecture

Defines the contract that all user-defined strategies must implement.
This ensures compatibility with all optimizers and provides type safety.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class OptimizableStrategy(ABC):
    """
    Abstract base class that all trading strategies must implement for optimization.
    
    This contract ensures that strategies can work with any optimizer
    and provides a consistent interface for strategy evaluation.
    """
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Return parameter definitions with types, ranges, and metadata.
        
        Returns:
            Dictionary mapping parameter names to their configuration:
            {
                "param_name": {
                    "type": int|float|str,
                    "default": default_value,
                    "range": (min_val, max_val),  # For optimization
                    "description": "Human readable description"
                }
            }
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for identification and logging"""
        pass
    
    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Strategy type for grouping and analysis (e.g., 'MA_Crossover', 'RSI_Momentum')"""
        pass
    
    @abstractmethod
    def should_buy(self, price_history: List[Dict], current_data: Dict) -> bool:
        """
        Determine if strategy should generate a buy signal.
        
        Args:
            price_history: List of historical price data dictionaries
            current_data: Current market data dictionary
            
        Returns:
            True if strategy recommends buying, False otherwise
        """
        pass
    
    @abstractmethod
    def should_sell(self, price_history: List[Dict], current_data: Dict) -> bool:
        """
        Determine if strategy should generate a sell signal.
        
        Args:
            price_history: List of historical price data dictionaries
            current_data: Current market data dictionary
            
        Returns:
            True if strategy recommends selling, False otherwise
        """
        pass
    
    @abstractmethod
    def get_position_size(self, available_cash: float, current_price: float) -> float:
        """
        Calculate position size for a trade.
        
        Args:
            available_cash: Available cash for trading
            current_price: Current price per share
            
        Returns:
            Dollar amount to invest in this trade
        """
        pass
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values for this strategy instance.
        
        Returns:
            Dictionary of current parameter values
        """
        current_params = {}
        for param_name in self.parameters.keys():
            if hasattr(self, param_name):
                current_params[param_name] = getattr(self, param_name)
        return current_params
    
    def validate_parameters(self, **params) -> bool:
        """
        Validate that given parameters are within acceptable ranges.
        
        Args:
            **params: Parameter values to validate
            
        Returns:
            True if all parameters are valid
        """
        for param_name, param_value in params.items():
            if param_name not in self.parameters:
                return False
            
            param_config = self.parameters[param_name]
            
            # Type check
            expected_type = param_config.get('type')
            if expected_type and not isinstance(param_value, expected_type):
                return False
            
            # Range check
            param_range = param_config.get('range')
            if param_range:
                min_val, max_val = param_range
                if param_value < min_val or param_value > max_val:
                    return False
        
        return True
    
    def _calculate_moving_average(self, price_history: List[Dict], period: int, 
                                price_field: str = 'close') -> Optional[float]:
        """
        Helper method to calculate simple moving average.
        
        Args:
            price_history: Historical price data
            period: Number of periods for the average
            price_field: Price field to use ('close', 'high', 'low', etc.)
            
        Returns:
            Moving average value or None if insufficient data
        """
        if len(price_history) < period:
            return None
        
        recent_prices = [day[price_field] for day in price_history[-period:]]
        return sum(recent_prices) / len(recent_prices)
    
    def __str__(self) -> str:
        """String representation of strategy with current parameters"""
        params = self.get_current_parameters()
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.name}({param_str})"
