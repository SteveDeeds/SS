"""
Order Manager for trading simulation

Manages order lifecycle, execution, and tracking.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from .order import Order


def is_limit_triggered(order_type: str, limit_price: float, market_data: Dict[str, Any]) -> bool:
    """Check if limit order should be triggered"""
    current_price = market_data.get('close', market_data.get('price', 0))
    
    if order_type == 'LIMIT_BUY':
        return current_price <= limit_price
    elif order_type == 'LIMIT_SELL':
        return current_price >= limit_price
    
    return False


def get_execution_price(order_type: str, limit_price: float, market_data: Dict[str, Any], slippage_pct: float = 0.0) -> float:
    """Get execution price for order with slippage"""
    # For market orders, use current market price
    if order_type.startswith('MARKET'):
        base_price = market_data.get('close', market_data.get('price', 0))
        
        # Apply slippage: buy orders pay more, sell orders receive less
        if order_type == 'MARKET_BUY':
            return base_price * (1 + slippage_pct)  # Pay higher price
        elif order_type == 'MARKET_SELL':
            return base_price * (1 - slippage_pct)  # Receive lower price
        else:
            return base_price
    
    # For limit orders, use limit price (no slippage applied to limit orders)
    return limit_price


def calculate_commission(quantity: int, price: float, commission_per_trade: float) -> float:
    """Calculate commission for trade"""
    return commission_per_trade


class OrderManager:
    """
    Manages the lifecycle of trading orders
    """
    
    def __init__(self, commission_per_trade: float = 1.0, slippage_pct: float = 0.0):
        """
        Initialize order manager
        
        Args:
            commission_per_trade: Commission charged per trade
            slippage_pct: Slippage percentage (e.g., 0.001 for 0.1%)
        """
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct
        self.pending_orders: List[Order] = []
        self.executed_orders: List[Order] = []
        self.expired_orders: List[Order] = []
        self.canceled_orders: List[Order] = []
    
    def place_order(self, order: Order) -> None:
        """
        Place an order for execution
        
        Args:
            order: Order to place
        """
        self.pending_orders.append(order)
    
    def place_orders(self, orders: List[Order]) -> None:
        """
        Place multiple orders
        
        Args:
            orders: List of orders to place
        """
        for order in orders:
            self.place_order(order)
    
    def process_pending_orders(self, market_data: Dict[str, Any], portfolio: Dict[str, Any], 
                             current_time: datetime) -> List[Dict[str, Any]]:
        """
        Process pending orders against current market data
        
        Args:
            market_data: Current market data
            portfolio: Current portfolio state
            current_time: Current simulation time (required)
            
        Returns:
            List of executed trades
        """
        
        executed_trades = []
        remaining_orders = []
        
        for order in self.pending_orders:
            # Check if order has expired
            if order.is_expired(current_time):
                order.expire(current_time)
                self.expired_orders.append(order)
                continue
            
            # Only process orders for matching symbol
            if order.symbol != market_data.get('symbol'):
                remaining_orders.append(order)
                continue
            
            # Try to execute the order
            trade = self._try_execute_order(order, market_data, portfolio, current_time)
            
            if trade:
                # Order executed successfully
                self.executed_orders.append(order)
                executed_trades.append(trade)
            else:
                # Order remains pending
                remaining_orders.append(order)
        
        # Update pending orders list
        self.pending_orders = remaining_orders
        
        return executed_trades
    
    def _try_execute_order(self, order: Order, market_data: Dict[str, Any], 
                          portfolio: Dict[str, Any], current_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Attempt to execute a single order
        
        Args:
            order: Order to execute
            market_data: Current market data
            portfolio: Current portfolio state
            current_time: Current time
            
        Returns:
            Trade dictionary if executed, None if not executed
        """
        # Check execution conditions based on order type
        can_execute = False
        
        if order.is_market_order():
            # Market orders always execute (if sufficient funds/shares)
            can_execute = True
        elif order.is_limit_order():
            # Limit orders execute only if price conditions are met
            can_execute = is_limit_triggered(order.type, order.limit_price, market_data)
        
        if not can_execute:
            return None
        
        # Determine execution price with slippage
        execution_price = get_execution_price(order.type, order.limit_price, market_data, self.slippage_pct)
        
        # Check if sufficient funds/shares available
        if order.is_buy_order():
            commission = calculate_commission(order.quantity, execution_price, self.commission_per_trade)
            total_cost = (execution_price * order.quantity) + commission
            
            if portfolio.get('cash_balance', 0) < total_cost:
                print(f"🚫 BUY order rejected: Need ${total_cost:.2f}, only have ${portfolio.get('cash_balance', 0):.2f}")
                return None  # Insufficient funds
        
        elif order.is_sell_order():
            holdings = portfolio.get('holdings', {})
            symbol_holding = holdings.get(order.symbol, {})
            available_quantity = symbol_holding.get('quantity', 0)
            
            if available_quantity < order.quantity:
                print(f"🚫 SELL order rejected: Need {order.quantity} shares of {order.symbol}, only have {available_quantity}")
                return None  # Insufficient shares
        
        # Execute the order
        commission = calculate_commission(order.quantity, execution_price, self.commission_per_trade)
        order.fill(execution_price, current_time)
        
        # Create trade result
        trade = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'execution_price': execution_price,
            'commission': commission,
            'direction': 'BUY' if order.is_buy_order() else 'SELL',
            'executed_at': current_time,
            'order_type': order.type
        }
        
        # Add cost/proceeds based on direction
        if order.is_buy_order():
            trade['total_cost'] = (execution_price * order.quantity) + commission
        else:
            trade['total_proceeds'] = (execution_price * order.quantity) - commission
        
        return trade
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was canceled, False if not found
        """
        for i, order in enumerate(self.pending_orders):
            if order.order_id == order_id:
                order.cancel()
                self.canceled_orders.append(order)
                del self.pending_orders[i]
                return True
        
        return False
    
    def get_pending_orders(self) -> List[Order]:
        """Get list of pending orders"""
        return self.pending_orders.copy()
    
    def get_executed_orders(self) -> List[Order]:
        """Get list of executed orders"""
        return self.executed_orders.copy()
    
    def get_pending_orders_count(self) -> int:
        """Get count of pending orders"""
        return len(self.pending_orders)
    
    def get_order_summary(self) -> Dict[str, int]:
        """Get summary of order counts by status"""
        return {
            'pending': len(self.pending_orders),
            'executed': len(self.executed_orders),
            'expired': len(self.expired_orders),
            'canceled': len(self.canceled_orders)
        }
