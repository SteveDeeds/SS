"""
Adaptive Moving Average Crossover Strategy

A trend-following strategy that uses two moving averages with configurable periods.
Both fast and slow periods are optimizable parameters, with validation to ensure
the fast period is always less than the slow period for proper signal generation.
"""
import sys
import os
from typing import Dict, Any, List, Optional, Tuple

# Add src directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)  # Insert at beginning to ensure priority

from src.optimizer.strategy_interface import OptimizableStrategy


class AdaptiveMAStrategy(OptimizableStrategy):
    """
    Adaptive Moving Average Crossover Strategy.
    
    Uses two moving averages with configurable fast and slow periods.
    Includes validation to ensure fast period < slow period for proper signals.
    """
    
    def __init__(self, buy_slow_period: int = 30, buy_fast_period: int = 15, 
                 sell_slow_period: int = 30, sell_fast_period: int = 15, cash_percentage: float = 0.15):
        """
        Initialize strategy with parameters.
        
        Args:
            buy_slow_period: Slow moving average period for buy signals
            buy_fast_period: Fast moving average period for buy signals
            sell_slow_period: Slow moving average period for sell signals
            sell_fast_period: Fast moving average period for sell signals
            cash_percentage: Percentage of available cash to use per trade
        """
        self.buy_slow_period = buy_slow_period
        self.buy_fast_period = buy_fast_period
        self.sell_slow_period = sell_slow_period
        self.sell_fast_period = sell_fast_period
        self.cash_percentage = cash_percentage
        
        # Ensure fast periods are always less than slow periods
        if self.buy_fast_period >= self.buy_slow_period:
            self.buy_fast_period = max(5, self.buy_slow_period // 2)
        if self.sell_fast_period >= self.sell_slow_period:
            self.sell_fast_period = max(5, self.sell_slow_period // 2)
        
        # Fixed parameters
        self.stop_loss_pct = 0.05  # 5% stop loss
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define optimizable parameters with metadata"""
        return {
            "buy_slow_period": {
                "type": int,
                "default": 30,
                "range": (5, 120),
                "description": "Slow moving average period for buy signals"
            },
            "buy_fast_period": {
                "type": int,
                "default": 15,
                "range": (1, 90),
                "description": "Fast moving average period for buy signals"
            },
            "sell_slow_period": {
                "type": int,
                "default": 30,
                "range": (5, 120),
                "description": "Slow moving average period for sell signals"
            },
            "sell_fast_period": {
                "type": int,
                "default": 15,
                "range": (1, 90),
                "description": "Fast moving average period for sell signals"
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
        return f"Adaptive MA Buy({self.buy_fast_period},{self.buy_slow_period}) Sell({self.sell_fast_period},{self.sell_slow_period}) - Cash({self.cash_percentage:.2f})"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type for grouping"""
        return "MA_Crossover"
    
    def should_buy(self, price_history: List[Dict], current_data: Dict) -> bool:
        """
        Generate buy signal when buy_fast_ma crosses above buy_slow_ma (Golden Cross).
        
        Args:
            price_history: Historical price data
            current_data: Current market data
            
        Returns:
            True if should buy (golden cross detected using buy MA periods)
        """
        # Need enough data for both moving averages
        if len(price_history) < self.buy_slow_period:
            return False
        
        # Calculate current moving averages using buy periods
        current_fast_ma = self._calculate_moving_average(price_history, self.buy_fast_period)
        current_slow_ma = self._calculate_moving_average(price_history, self.buy_slow_period)
        
        if current_fast_ma is None or current_slow_ma is None:
            return False
        
        # Calculate previous moving averages for crossover detection
        if len(price_history) < self.buy_slow_period + 1:
            return False  # Not enough data for crossover detection
        
        prev_fast_ma = self._calculate_moving_average(price_history[:-1], self.buy_fast_period)
        prev_slow_ma = self._calculate_moving_average(price_history[:-1], self.buy_slow_period)
        
        if prev_fast_ma is None or prev_slow_ma is None:
            return False
        
        # Check for golden cross: buy_fast_ma crosses above buy_slow_ma
        golden_cross = (
            prev_fast_ma <= prev_slow_ma and  # Was below or equal
            current_fast_ma > current_slow_ma  # Now above
        )
        
        return golden_cross
    
    def should_sell(self, price_history: List[Dict], current_data: Dict) -> bool:
        """
        Generate sell signal when sell_fast_ma crosses below sell_slow_ma (Death Cross).
        
        Args:
            price_history: Historical price data
            current_data: Current market data
            
        Returns:
            True if should sell (death cross detected using sell MA periods)
        """
        # Need enough data for both moving averages
        if len(price_history) < self.sell_slow_period:
            return False
        
        # Calculate current moving averages using sell periods
        current_fast_ma = self._calculate_moving_average(price_history, self.sell_fast_period)
        current_slow_ma = self._calculate_moving_average(price_history, self.sell_slow_period)
        
        if current_fast_ma is None or current_slow_ma is None:
            return False
        
        # Calculate previous moving averages for crossover detection
        if len(price_history) < self.sell_slow_period + 1:
            return False  # Not enough data for crossover detection
        
        prev_fast_ma = self._calculate_moving_average(price_history[:-1], self.sell_fast_period)
        prev_slow_ma = self._calculate_moving_average(price_history[:-1], self.sell_slow_period)
        
        if prev_fast_ma is None or prev_slow_ma is None:
            return False
        
        # Check for death cross: sell_fast_ma crosses below sell_slow_ma
        death_cross = (
            prev_fast_ma >= prev_slow_ma and  # Was above or equal
            current_fast_ma < current_slow_ma  # Now below
        )
        
        return death_cross
    
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
    
    def get_fast_period(self) -> int:
        """Get the computed buy fast period (for backward compatibility)"""
        return self.buy_fast_period
    
    def get_buy_fast_period(self) -> int:
        """Get the buy fast period"""
        return self.buy_fast_period
    
    def get_sell_fast_period(self) -> int:
        """Get the sell fast period"""
        return self.sell_fast_period
    
    def get_stop_loss_price(self, entry_price: float) -> float:
        """Calculate stop loss price"""
        return entry_price * (1 - self.stop_loss_pct)
    
    def _calculate_moving_average(self, price_history: List[Dict], period: int) -> Optional[float]:
        """
        Calculate simple moving average for the given period.
        
        Args:
            price_history: List of price data dictionaries with 'close' prices
            period: Number of periods for moving average
            
        Returns:
            Moving average value or None if insufficient data
        """
        if len(price_history) < period:
            return None
        
        # Get closing prices for the period
        closes = [day['close'] for day in price_history[-period:]]
        
        # Calculate simple moving average
        return sum(closes) / len(closes)


# Strategy information for external tools
STRATEGY_INFO = {
    "name": "Adaptive MA Crossover",
    "description": "Moving average crossover with adaptive fast/slow ratio",
    "version": "1.0",
    "author": "Trading System",
    "strategy_class": AdaptiveMAStrategy
}
