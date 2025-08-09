"""
Order class for trading simulation

Represents individual orders with lifecycle tracking.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import uuid


class Order:
    """
    Trading order with simplified structure and lifecycle tracking
    """
    
    def __init__(self, 
                 symbol: str,
                 order_type: str,
                 quantity: int,
                 limit_price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 expiration_date: Optional[datetime] = None,
                 placement_time: Optional[datetime] = None):
        """
        Create a new order
        
        Args:
            symbol: Stock symbol (e.g., 'SPXL')
            order_type: 'MARKET_BUY', 'MARKET_SELL', 'LIMIT_BUY', 'LIMIT_SELL'
            quantity: Number of shares
            limit_price: Limit price for limit orders (None for market orders)
            stop_price: Stop price for stop orders (None for regular orders)
            expiration_date: When order expires (defaults to 1 day for market, 30 days for limit)
            placement_time: When order was placed (required for simulation)
        """
        self.order_id = self._generate_order_id(symbol, order_type, quantity)
        self.symbol = symbol
        self.type = order_type
        self.quantity = quantity
        self.limit_price = limit_price
        self.stop_price = stop_price
        
        # datetimes - placement_time must be explicitly provided in simulation
        if placement_time is None:
            raise ValueError("placement_time is required for simulation orders")
        self.datetime_placed = placement_time
        self.datetime_expired = expiration_date or self._default_expiration(order_type)
        self.datetime_executed: Optional[datetime] = None
        
        # Status and execution
        self.status = 'PENDING'  # PENDING, EXECUTED, CANCELED, EXPIRED
        self.execution_price: Optional[float] = None
    
    def _generate_order_id(self, symbol: str, order_type: str, quantity: int) -> str:
        """Generate unique order ID"""
        prefix = order_type.split('_')[0]  # MARKET or LIMIT
        unique_suffix = str(uuid.uuid4())[:8]
        return f"{prefix}_{symbol}_{quantity}_{unique_suffix}"
    
    def _default_expiration(self, order_type: str) -> datetime:
        """Set default expiration based on order type"""
        if order_type.startswith('MARKET'):
            # Market orders expire end of following day
            return (self.datetime_placed + timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # Limit orders expire in 30 days
            return self.datetime_placed + timedelta(days=30)
    
    def is_expired(self, current_time: datetime) -> bool:
        """Check if order has expired"""
        return current_time >= self.datetime_expired
    
    def is_market_order(self) -> bool:
        """Check if this is a market order"""
        return self.type in ['MARKET_BUY', 'MARKET_SELL']
    
    def is_limit_order(self) -> bool:
        """Check if this is a limit order"""
        return self.type in ['LIMIT_BUY', 'LIMIT_SELL']
    
    def is_buy_order(self) -> bool:
        """Check if this is a buy order"""
        return self.type in ['MARKET_BUY', 'LIMIT_BUY']
    
    def is_sell_order(self) -> bool:
        """Check if this is a sell order"""
        return self.type in ['MARKET_SELL', 'LIMIT_SELL']
    
    def fill(self, execution_price: float, execution_time: datetime) -> None:
        """Mark order as executed"""
        self.status = 'EXECUTED'
        self.execution_price = execution_price
        self.datetime_executed = execution_time
    
    def cancel(self) -> None:
        """Cancel the order"""
        self.status = 'CANCELED'
    
    def expire(self, expiration_time: datetime) -> None:
        """Mark order as expired"""
        self.status = 'EXPIRED'
        self.datetime_executed = expiration_time
    
    def __str__(self) -> str:
        """String representation"""
        status_info = f"{self.status}"
        if self.execution_price:
            status_info += f" on ${self.datetime_executed} @ ${self.execution_price:.2f}"
        
        return f"{self.type}: {self.quantity} {self.symbol} ({status_info})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"Order({self.order_id}, {self.type}, {self.quantity} {self.symbol}, {self.status})"
