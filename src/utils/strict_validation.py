#!/usr/bin/env python3
"""
Strict Validation Utilities

This module provides strict validation utilities to ensure data integrity
and fail fast when assumptions are violated. No more silent failures!
"""

from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import traceback


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class DataIntegrityError(Exception):
    """Raised when data structure integrity is violated"""
    pass


class RequiredFieldError(Exception):
    """Raised when a required field is missing"""
    pass


def require_not_none(value: Any, field_name: str) -> Any:
    """
    Strict validation that a value is not None
    
    Args:
        value: The value to check
        field_name: Name of the field for error messages
        
    Returns:
        The value if not None
        
    Raises:
        ValidationError: If value is None
    """
    if value is None:
        raise ValidationError(f"Required field '{field_name}' cannot be None")
    return value


def require_type(value: Any, expected_type: type, field_name: str) -> Any:
    """
    Strict type validation
    
    Args:
        value: The value to check
        expected_type: The expected type
        field_name: Name of the field for error messages
        
    Returns:
        The value if type is correct
        
    Raises:
        ValidationError: If type is incorrect
    """
    if not isinstance(value, expected_type):
        actual_type = type(value).__name__
        expected_name = expected_type.__name__
        raise ValidationError(
            f"Field '{field_name}' must be {expected_name}, got {actual_type}"
        )
    return value


def require_dict_keys(data: Dict[str, Any], required_keys: List[str], 
                     context: str = "data") -> Dict[str, Any]:
    """
    Strict validation that dictionary has all required keys
    
    Args:
        data: Dictionary to validate
        required_keys: List of required key names
        context: Context description for error messages
        
    Returns:
        The dictionary if valid
        
    Raises:
        DataIntegrityError: If required keys are missing
    """
    if not isinstance(data, dict):
        raise DataIntegrityError(f"{context} must be a dictionary, got {type(data).__name__}")
    
    missing_keys = []
    for key in required_keys:
        if key not in data:
            missing_keys.append(key)
    
    if missing_keys:
        raise DataIntegrityError(
            f"{context} missing required keys: {missing_keys}. "
            f"Available keys: {list(data.keys())}"
        )
    
    return data


def require_non_empty(value: Union[List, Dict, str], field_name: str) -> Union[List, Dict, str]:
    """
    Strict validation that collection is not empty
    
    Args:
        value: The collection to check
        field_name: Name of the field for error messages
        
    Returns:
        The value if not empty
        
    Raises:
        ValidationError: If collection is empty
    """
    if not value:
        raise ValidationError(f"Field '{field_name}' cannot be empty")
    return value


def require_positive(value: Union[int, float], field_name: str) -> Union[int, float]:
    """
    Strict validation that numeric value is positive
    
    Args:
        value: The numeric value to check
        field_name: Name of the field for error messages
        
    Returns:
        The value if positive
        
    Raises:
        ValidationError: If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Field '{field_name}' must be numeric, got {type(value).__name__}")
    
    if value <= 0:
        raise ValidationError(f"Field '{field_name}' must be positive, got {value}")
    
    return value


def validate_portfolio_state(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict validation of portfolio state structure
    
    Args:
        portfolio: Portfolio state dictionary
        
    Returns:
        Validated portfolio state
        
    Raises:
        DataIntegrityError: If portfolio structure is invalid
    """
    require_dict_keys(portfolio, 
                     ['portfolio_id', 'cash_balance', 'holdings', 'total_value'],
                     "Portfolio state")
    
    # Validate cash balance
    cash_balance = require_not_none(portfolio['cash_balance'], 'portfolio.cash_balance')
    require_type(cash_balance, (int, float), 'portfolio.cash_balance')
    
    # Validate holdings structure
    holdings = require_not_none(portfolio['holdings'], 'portfolio.holdings')
    require_type(holdings, dict, 'portfolio.holdings')
    
    # Validate each holding
    for symbol, holding in holdings.items():
        require_type(symbol, str, f'holding symbol')
        require_non_empty(symbol, f'holding symbol')
        
        require_dict_keys(holding, ['quantity', 'average_cost'], 
                         f"Holding for {symbol}")
        
        quantity = require_not_none(holding['quantity'], f'holding[{symbol}].quantity')
        require_type(quantity, (int, float), f'holding[{symbol}].quantity')
        
        avg_cost = require_not_none(holding['average_cost'], f'holding[{symbol}].average_cost')
        require_positive(avg_cost, f'holding[{symbol}].average_cost')
    
    return portfolio


def validate_market_data(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict validation of market data structure
    
    Args:
        market_data: Market data dictionary
        
    Returns:
        Validated market data
        
    Raises:
        DataIntegrityError: If market data structure is invalid
    """
    require_dict_keys(market_data, 
                     ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'],
                     "Market data")
    
    # Validate symbol
    symbol = require_not_none(market_data['symbol'], 'market_data.symbol')
    require_type(symbol, str, 'market_data.symbol')
    require_non_empty(symbol, 'market_data.symbol')
    
    # Validate OHLC prices
    for price_field in ['open', 'high', 'low', 'close']:
        price = require_not_none(market_data[price_field], f'market_data.{price_field}')
        require_positive(price, f'market_data.{price_field}')
    
    # Validate OHLC relationships
    open_price = market_data['open']
    high_price = market_data['high']
    low_price = market_data['low']
    close_price = market_data['close']
    
    if high_price < max(open_price, close_price):
        raise DataIntegrityError(
            f"High price {high_price} cannot be less than max(open={open_price}, close={close_price})"
        )
    
    if low_price > min(open_price, close_price):
        raise DataIntegrityError(
            f"Low price {low_price} cannot be greater than min(open={open_price}, close={close_price})"
        )
    
    # Validate volume
    volume = require_not_none(market_data['volume'], 'market_data.volume')
    require_type(volume, (int, float), 'market_data.volume')
    
    return market_data


def validate_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict validation of order structure
    
    Args:
        order: Order dictionary
        
    Returns:
        Validated order
        
    Raises:
        DataIntegrityError: If order structure is invalid
    """
    require_dict_keys(order, ['action', 'symbol'], "Order")
    
    # Validate action
    action = require_not_none(order['action'], 'order.action')
    require_type(action, str, 'order.action')
    valid_actions = ['BUY', 'SELL', 'buy', 'sell']
    if action not in valid_actions:
        raise ValidationError(f"Order action must be one of {valid_actions}, got '{action}'")
    
    # Validate symbol
    symbol = require_not_none(order['symbol'], 'order.symbol')
    require_type(symbol, str, 'order.symbol')
    require_non_empty(symbol, 'order.symbol')
    
    # Validate quantity/amount based on action
    if action.upper() == 'BUY':
        if 'amount' not in order:
            raise RequiredFieldError("BUY orders must have 'amount' field")
        amount = require_not_none(order['amount'], 'order.amount')
        require_positive(amount, 'order.amount')
    
    elif action.upper() == 'SELL':
        if 'shares' not in order:
            raise RequiredFieldError("SELL orders must have 'shares' field")
        shares = require_not_none(order['shares'], 'order.shares')
        require_positive(shares, 'order.shares')
    
    return order


def strict_error_handler(func: Callable) -> Callable:
    """
    Decorator that provides strict error handling with detailed stack traces
    
    Args:
        func: Function to wrap with strict error handling
        
    Returns:
        Wrapped function that fails fast with detailed errors
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValidationError, DataIntegrityError, RequiredFieldError) as e:
            # Re-raise validation errors with full context
            print(f"\n🚨 VALIDATION ERROR in {func.__name__}:")
            print(f"   Error: {str(e)}")
            print(f"   Function: {func.__module__}.{func.__name__}")
            print(f"   Args: {args}")
            print(f"   Kwargs: {kwargs}")
            raise e
        except Exception as e:
            # Wrap unexpected errors with context
            print(f"\n💥 UNEXPECTED ERROR in {func.__name__}:")
            print(f"   Error: {str(e)}")
            print(f"   Type: {type(e).__name__}")
            print(f"   Function: {func.__module__}.{func.__name__}")
            print(f"   Args: {args}")
            print(f"   Kwargs: {kwargs}")
            print(f"\n📍 Full Stack Trace:")
            traceback.print_exc()
            raise RuntimeError(f"Unexpected error in {func.__name__}: {str(e)}") from e
    
    return wrapper


def enable_strict_mode():
    """
    Enable strict mode globally by setting environment variable
    This can be checked by other modules to enable/disable strict validation
    """
    import os
    os.environ['TRADING_SYSTEM_STRICT_MODE'] = '1'
    print("🔒 STRICT MODE ENABLED - System will fail fast on any validation errors")


def is_strict_mode() -> bool:
    """
    Check if strict mode is enabled
    
    Returns:
        True if strict mode is enabled
    """
    import os
    return os.environ.get('TRADING_SYSTEM_STRICT_MODE', '0') == '1'


# Example usage and testing
if __name__ == "__main__":
    # Enable strict mode
    enable_strict_mode()
    
    print("🧪 Testing strict validation utilities...")
    
    # Test portfolio validation
    try:
        invalid_portfolio = {'cash_balance': None, 'holdings': {}}
        validate_portfolio_state(invalid_portfolio)
    except ValidationError as e:
        print(f"✅ Caught invalid portfolio: {e}")
    
    # Test market data validation
    try:
        invalid_market_data = {
            'symbol': 'SPXL',
            'date': '2024-01-01',
            'open': 100,
            'high': 95,  # Invalid: high < open
            'low': 90,
            'close': 98,
            'volume': 1000
        }
        validate_market_data(invalid_market_data)
    except DataIntegrityError as e:
        print(f"✅ Caught invalid market data: {e}")
    
    print("🔒 Strict validation utilities are working correctly!")
