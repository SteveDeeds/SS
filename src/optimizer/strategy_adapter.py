"""
Strategy Adapter to bridge new OptimizableStrategy interface with existing simulator

This adapter allows new-style strategies (with should_buy/should_sell) to work
with the existing simulator that expects evaluate_and_place_orders method.
"""
from typing import List, Dict, Optional, Any
from datetime import timedelta
import sys
import os

# Add project root to path for strict validation and Order import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys_src_path = os.path.join(project_root, 'src')
if sys_src_path not in sys.path:
    sys.path.append(sys_src_path)

# Import Order class for object creation
from orders.order import Order

try:
    from src.utils.strict_validation import (
        strict_error_handler, 
        validate_portfolio_state, 
        validate_market_data, 
        validate_order,
        require_not_none,
        require_positive,
        is_strict_mode
    )
    STRICT_VALIDATION_AVAILABLE = True
except ImportError:
    print("⚠️ Strict validation not available - running in legacy mode")
    STRICT_VALIDATION_AVAILABLE = False
    
    # Provide fallback functions
    def strict_error_handler(func):
        return func
    def validate_portfolio_state(portfolio):
        return portfolio
    def validate_market_data(market_data):
        return market_data
    def validate_order(order):
        return order
    def require_not_none(value, field_name):
        if value is None:
            raise ValueError(f"Required field '{field_name}' cannot be None")
        return value
    def require_positive(value, field_name):
        if value <= 0:
            raise ValueError(f"Field '{field_name}' must be positive, got {value}")
        return value
    def is_strict_mode():
        return False


class StrategyAdapter:
    """
    Adapter that wraps an OptimizableStrategy to work with the existing simulator.
    
    The simulator expects evaluate_and_place_orders() method, but new strategies
    use should_buy()/should_sell() methods. This adapter bridges the gap.
    """
    
    def __init__(self, optimizable_strategy):
        """
        Initialize adapter with an OptimizableStrategy instance.
        
        Args:
            optimizable_strategy: Instance of a class implementing OptimizableStrategy
        """
        self.strategy = require_not_none(optimizable_strategy, "optimizable_strategy")
        
        # Validate strategy has required methods
        if not hasattr(self.strategy, 'should_buy'):
            raise AttributeError(f"Strategy {type(self.strategy).__name__} must implement should_buy() method")
        if not hasattr(self.strategy, 'should_sell'):
            raise AttributeError(f"Strategy {type(self.strategy).__name__} must implement should_sell() method")
        if not hasattr(self.strategy, 'get_position_size'):
            raise AttributeError(f"Strategy {type(self.strategy).__name__} must implement get_position_size() method")
        
        # Copy strategy attributes for compatibility
        self.name = getattr(optimizable_strategy, 'name', 'Unknown Strategy')
        self.strategy_type = getattr(optimizable_strategy, 'strategy_type', 'Unknown')
    
    @strict_error_handler
    def evaluate_and_place_orders(self, price_history: List[Dict], 
                                portfolio: Dict, current_market_data: Dict) -> List[Order]:
        """
        Adapter method that converts new strategy interface to simulator interface.
        
        If the strategy implements its own evaluate_and_place_orders method, use that.
        Otherwise, use the default should_buy/should_sell logic with market orders.
        
        Args:
            price_history: Historical price data
            portfolio: Current portfolio state
            current_market_data: Current market data
            
        Returns:
            List of Order objects for the simulator
        """
        # Check if strategy has its own evaluate_and_place_orders implementation
        if hasattr(self.strategy, 'evaluate_and_place_orders') and callable(getattr(self.strategy, 'evaluate_and_place_orders')):
            # Use strategy's custom order placement logic
            try:
                return self.strategy.evaluate_and_place_orders(price_history, portfolio, current_market_data)
            except Exception as e:
                if is_strict_mode():
                    print(f"💥 STRATEGY CUSTOM ORDER PLACEMENT ERROR: {e}")
                    raise e
                else:
                    print(f"⚠️ Strategy custom order placement error, falling back to default: {e}")
                    # Fall through to default logic
        
        # Default logic using should_buy/should_sell with market orders
        return self._default_evaluate_and_place_orders(price_history, portfolio, current_market_data)
    
    def _default_evaluate_and_place_orders(self, price_history: List[Dict], 
                                         portfolio: Dict, current_market_data: Dict) -> List[Order]:
        """
        Default order placement logic using should_buy/should_sell with market orders.
        
        Args:
            price_history: Historical price data
            portfolio: Current portfolio state
            current_market_data: Current market data
            
        Returns:
            List of Order objects for the simulator
        """
        # Strict validation of inputs
        if is_strict_mode():
            require_not_none(price_history, "price_history")
            require_not_none(portfolio, "portfolio")
            require_not_none(current_market_data, "current_market_data")
            
            validate_portfolio_state(portfolio)
            validate_market_data(current_market_data)
            
            if not isinstance(price_history, list) or len(price_history) == 0:
                raise ValueError("price_history must be a non-empty list")
        
        orders = []
        
        # Get current position info - fix portfolio structure access
        holdings = portfolio.get('holdings', {})
        symbol = require_not_none(current_market_data.get('symbol'), "current_market_data.symbol")
        current_shares = holdings.get(symbol, {}).get('quantity', 0)
        available_cash = require_not_none(portfolio.get('cash_balance'), "portfolio.cash_balance")
        current_price = require_positive(current_market_data.get('close', 0), "current_market_data.close")
        
        try:
            # Check for buy signal
            if current_shares == 0:  # Only buy if we don't have shares
                should_buy = self.strategy.should_buy(price_history, current_market_data)
                
                if should_buy:
                    # Calculate position size
                    position_size = self.strategy.get_position_size(available_cash, current_price)
                    
                    if position_size > 0 and position_size <= available_cash:
                        # Calculate quantity from position size (amount in dollars)
                        quantity = int(position_size / current_price)
                        
                        if quantity > 0:
                            # Get placement time from market data
                            placement_time = current_market_data.get('date') if 'date' in current_market_data else None
                            
                            # Create Order object with correct placement time
                            order = Order(
                                symbol=symbol,
                                order_type='MARKET_BUY',
                                quantity=quantity,
                                limit_price=None,  # Market order
                                stop_price=None,   # No stop loss for now
                                placement_time=placement_time
                            )

                            orders.append(order)            # Check for sell signal
            elif current_shares > 0:  # Only sell if we have shares
                should_sell = self.strategy.should_sell(price_history, current_market_data)
                
                if should_sell:
                    # Get placement time from market data
                    placement_time = current_market_data.get('date') if 'date' in current_market_data else None
                    
                    # Set expiration to next day at market close (4:00 PM)
                    expiration_time = None
                    if placement_time:
                        next_day = placement_time + timedelta(days=1)
                        expiration_time = next_day.replace(hour=16, minute=0, second=0, microsecond=0)
                    
                    # Create Order object with correct placement time
                    order = Order(
                        symbol=symbol,
                        order_type='MARKET_SELL',
                        quantity=current_shares,
                        limit_price=None,  # Market order
                        stop_price=None,   # No stop loss for now
                        placement_time=placement_time,
                        expiration_date=expiration_time  # Next day at 4:00 PM market close
                    )

                    orders.append(order)

        except Exception as e:
            # In strict mode, re-raise all exceptions
            if is_strict_mode():
                print(f"💥 STRATEGY ADAPTER CRITICAL ERROR: {e}")
                print(f"   Portfolio: {portfolio}")
                print(f"   Market Data: {current_market_data}")
                print(f"   Holdings: {holdings}")
                raise e
            else:
                # Log error but don't crash simulation (legacy behavior)
                print(f"⚠️ Strategy adapter error: {e}")
        
        return orders
    
    def __str__(self):
        """String representation"""
        return f"StrategyAdapter({self.strategy})"
    
    def __repr__(self):
        """Detailed string representation"""
        return f"StrategyAdapter(strategy={repr(self.strategy)})"
