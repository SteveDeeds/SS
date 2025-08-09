"""
Bollinger Bands Strategy

A mean reversion strategy that uses Bollinger Bands to identify overbought 
and oversold conditions. Buys when price touches or goes below the lower band
and sells when price touches or goes above the upper band.
"""
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import timedelta

# Add src directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)  # Insert at beginning to ensure priority

from src.optimizer.strategy_interface import OptimizableStrategy

# Import Order class for custom order placement
try:
    from src.orders.order import Order
except ImportError:
    # Fallback import path
    import sys
    sys.path.append(os.path.join(project_root, 'src'))
    from orders.order import Order


class BollingerBandsStrategy(OptimizableStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    
    Uses Bollinger Bands to identify mean reversion opportunities.
    Generates buy signals when price touches the lower band and
    sell signals when price touches the upper band.
    """
    
    def __init__(self, period: int = 20, std_positive: float = 2.0, 
                 std_negative: float = 2.0, cash_percentage: float = 0.15):
        """
        Initialize strategy with parameters.
        
        Args:
            period: Period for moving average calculation (typically 20)
            std_positive: Standard deviations for upper band (sell signal)
            std_negative: Standard deviations for lower band (buy signal)
            cash_percentage: Percentage of available cash to use per trade
        """
        self.period = period
        self.std_positive = std_positive
        self.std_negative = std_negative
        self.cash_percentage = cash_percentage
        
        # Ensure valid parameters
        if self.period < 2:
            self.period = 20
        if self.std_positive <= 0:
            self.std_positive = 2.0
        if self.std_negative <= 0:
            self.std_negative = 2.0
        
        # Fixed parameters
        self.stop_loss_pct = 0.05  # 5% stop loss
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define optimizable parameters with metadata"""
        return {
            "period": {
                "type": int,
                "default": 20,
                "range": (5, 100),
                "description": "Moving average period for Bollinger Bands"
            },
            "std_positive": {
                "type": float,
                "default": 2.0,
                "range": (0.5, 4.0),
                "description": "Standard deviations for upper band (sell signal)"
            },
            "std_negative": {
                "type": float,
                "default": 2.0,
                "range": (0.5, 4.0),
                "description": "Standard deviations for lower band (buy signal)"
            },
            "cash_percentage": {
                "type": float,
                "default": 0.15,
                "range": (0.05, 1.0),
                "description": "Percentage of available cash to use per trade"
            }
        }
    
    @property
    def name(self) -> str:
        """Strategy display name"""
        return f"BollingerBands({self.period}) - Upper:{self.std_positive}/Lower:{self.std_negative} - Cash({self.cash_percentage:.2f})"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type for grouping"""
        return "Bollinger_Bands_MeanReversion"
    
    def should_buy(self, price_history: List[Dict], current_data: Dict) -> bool:
        """
        Generate buy signal when price touches or goes below lower Bollinger Band.
        
        Args:
            price_history: Historical price data
            current_data: Current market data
            
        Returns:
            True if should buy (price at or below lower band)
        """
        # Need enough data for Bollinger Bands calculation
        if len(price_history) < self.period:
            return False
        
        # Calculate Bollinger Bands
        bands = self._calculate_bollinger_bands(price_history)
        if bands is None:
            return False
        
        middle, upper, lower = bands
        current_price = current_data['close']
        
        # Buy signal: price touches or goes below lower band
        return current_price <= lower
    
    def should_sell(self, price_history: List[Dict], current_data: Dict) -> bool:
        """
        Generate sell signal when price touches or goes above upper Bollinger Band.
        
        Args:
            price_history: Historical price data
            current_data: Current market data
            
        Returns:
            True if should sell (price at or above upper band)
        """
        # Need enough data for Bollinger Bands calculation
        if len(price_history) < self.period:
            return False
        
        # Calculate Bollinger Bands
        bands = self._calculate_bollinger_bands(price_history)
        if bands is None:
            return False
        
        middle, upper, lower = bands
        current_price = current_data['close']
        
        # Sell signal: price touches or goes above upper band
        return current_price >= upper
    
    def get_position_size(self, available_cash: float, current_price: float) -> float:
        """
        Calculate position size based on cash percentage.
        
        Args:
            available_cash: Available cash for trading
            current_price: Current price per share
            
        Returns:
            Dollar amount to invest
        """
        return available_cash * self.cash_percentage
    
    def get_stop_loss_price(self, entry_price: float) -> float:
        """Calculate stop loss price"""
        return entry_price * (1 - self.stop_loss_pct)
    
    def _calculate_bollinger_bands(self, price_history: List[Dict]) -> Optional[Tuple[float, float, float]]:
        """
        Calculate Bollinger Bands for the given price history.
        
        Args:
            price_history: List of price data dictionaries with 'close' prices
            
        Returns:
            Tuple of (middle_band, upper_band, lower_band) or None if insufficient data
        """
        if len(price_history) < self.period:
            return None
        
        # Get closing prices for the period
        closes = [day['close'] for day in price_history[-self.period:]]
        
        # Calculate moving average (middle band)
        middle_band = sum(closes) / len(closes)
        
        # Calculate standard deviation
        variance = sum((price - middle_band) ** 2 for price in closes) / len(closes)
        std_dev = variance ** 0.5
        
        # Calculate upper and lower bands
        upper_band = middle_band + (self.std_positive * std_dev)
        lower_band = middle_band - (self.std_negative * std_dev)
        
        return middle_band, upper_band, lower_band

    def evaluate_and_place_orders(self, price_history: List[Dict], 
                                 portfolio: Dict, current_market_data: Dict) -> List[Order]:
        """
        Custom order placement logic using limit orders for Bollinger Bands strategy.
        
        This method overrides the default StrategyAdapter behavior to implement
        sophisticated limit order placement based on Bollinger Bands levels.
        
        Args:
            price_history: Historical price data
            portfolio: Current portfolio state
            current_market_data: Current market data
            
        Returns:
            List of Order objects for limit buy/sell orders
        """
        orders = []
        
        # Get current position info
        holdings = portfolio.get('holdings', {})
        symbol = current_market_data.get('symbol')
        if not symbol:
            return orders
            
        current_shares = holdings.get(symbol, {}).get('quantity', 0)
        available_cash = portfolio.get('cash_balance', 0)
        current_price = current_market_data.get('close', 0)
        
        # Get pending orders to calculate committed shares
        pending_orders = portfolio.get('pending_orders', [])
        
        # Calculate shares committed to pending sell orders for this symbol
        committed_shares = 0
        for order in pending_orders:
            if (order.symbol == symbol and 
                order.type in ['LIMIT_SELL', 'MARKET_SELL'] and
                order.status == 'PENDING'):
                committed_shares += order.quantity
        
        # Calculate available shares for new sell orders
        available_shares = current_shares - committed_shares
        
        if current_price <= 0:
            return orders
        
        # Need enough data for Bollinger Bands calculation
        if len(price_history) < self.period:
            return orders
        
        # Calculate Bollinger Bands
        bands = self._calculate_bollinger_bands(price_history)
        if bands is None:
            return orders
        
        middle, upper, lower = bands
        placement_time = current_market_data.get('date')
        
        # Strategy: Use limit orders at band levels for better fills
        try:
            # BUY LOGIC: Place limit buy order at lower band if we don't have shares
            if current_shares == 0 and available_cash > 0:
                # Check if price is near or below lower band (buy signal)
                if current_price <= lower * 1.01:  # Allow 1% tolerance
                    # Calculate position size
                    position_size = self.get_position_size(available_cash, current_price)
                    
                    if position_size > 0 and position_size <= available_cash:
                        quantity = int(position_size / lower)  # Use lower band price for quantity calc
                        
                        if quantity > 0:
                            # Set expiration to next day at market close
                            expiration_time = None
                            if placement_time:
                                next_day = placement_time + timedelta(days=1)
                                expiration_time = next_day.replace(hour=16, minute=0, second=0, microsecond=0)
                            
                            # Create LIMIT BUY order at lower band
                            buy_order = Order(
                                symbol=symbol,
                                order_type='LIMIT_BUY',
                                quantity=quantity,
                                limit_price=lower,  # Buy at lower band
                                stop_price=None,
                                placement_time=placement_time,
                                expiration_date=expiration_time
                            )
                            orders.append(buy_order)
            
            # SELL LOGIC: Place limit sell order at upper band if we have available shares AND price is near upper band
            elif available_shares > 0:
                # Check if price is near or above upper band (sell signal)
                if current_price >= upper * 0.99:  # Allow 1% tolerance
                    # Set expiration to next day at market close
                    expiration_time = None
                    if placement_time:
                        next_day = placement_time + timedelta(days=1)
                        expiration_time = next_day.replace(hour=16, minute=0, second=0, microsecond=0)
                    
                    # Create LIMIT SELL order at upper band for available shares only
                    sell_order = Order(
                        symbol=symbol,
                        order_type='LIMIT_SELL',
                        quantity=available_shares,  # Only sell available shares, not all shares
                        limit_price=upper,  # Sell at upper band
                        stop_price=None,
                        placement_time=placement_time,
                        expiration_date=expiration_time
                    )
                    orders.append(sell_order)
                
        except Exception as e:
            print(f"⚠️ Bollinger Bands order placement error: {e}")
        
        return orders


# Strategy information for external tools
STRATEGY_INFO = {
    "name": "Bollinger Bands",
    "description": "Bollinger Bands mean reversion strategy using upper/lower band signals",
    "version": "1.0",
    "author": "Trading System",
    "strategy_class": BollingerBandsStrategy
}
