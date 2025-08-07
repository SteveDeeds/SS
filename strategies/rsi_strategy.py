"""
RSI Momentum Strategy

A momentum strategy that uses the Relative Strength Index (RSI) indicator
to identify overbought and oversold conditions for entry and exit signals.
Buys when RSI crosses below the oversold threshold and sells when RSI
crosses above the overbought threshold.
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


class RSIStrategy(OptimizableStrategy):
    """
    RSI Momentum Strategy.
    
    Uses the Relative Strength Index to identify overbought/oversold conditions.
    Generates buy signals when RSI crosses below oversold threshold and
    sell signals when RSI crosses above overbought threshold.
    """
    
    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30.0, 
                 overbought_threshold: float = 70.0, cash_percentage: float = 0.15):
        """
        Initialize strategy with parameters.
        
        Args:
            rsi_period: Period for RSI calculation (typically 14)
            oversold_threshold: RSI level below which stock is considered oversold (buy signal)
            overbought_threshold: RSI level above which stock is considered overbought (sell signal)
            cash_percentage: Percentage of available cash to use per trade
        """
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.cash_percentage = cash_percentage
        
        # Ensure valid threshold relationship
        if self.oversold_threshold >= self.overbought_threshold:
            self.oversold_threshold = 30.0
            self.overbought_threshold = 70.0
        
        # Fixed parameters
        self.stop_loss_pct = 0.05  # 5% stop loss
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define optimizable parameters with metadata"""
        return {
            "rsi_period": {
                "type": int,
                "default": 14,
                "range": (5, 100),
                "description": "RSI calculation period"
            },
            "oversold_threshold": {
                "type": float,
                "default": 30.0,
                "range": (5.0, 45.0),
                "description": "RSI oversold threshold (buy signal)"
            },
            "overbought_threshold": {
                "type": float,
                "default": 70.0,
                "range": (55.0, 95.0),
                "description": "RSI overbought threshold (sell signal)"
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
        return f"RSI({self.rsi_period}) - OS:{self.oversold_threshold}/OB:{self.overbought_threshold} - Cash({self.cash_percentage:.2f})"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type for grouping"""
        return "RSI_Momentum"
    
    def should_buy(self, price_history: List[Dict], current_data: Dict) -> bool:
        """
        Generate buy signal when RSI crosses below oversold threshold.
        
        Args:
            price_history: Historical price data
            current_data: Current market data
            
        Returns:
            True if should buy (RSI oversold condition detected)
        """
        # Need enough data for RSI calculation
        if len(price_history) < self.rsi_period + 1:
            return False
        
        # Calculate current and previous RSI
        current_rsi = self._calculate_rsi(price_history)
        if current_rsi is None:
            return False
        
        # Get previous RSI for crossover detection
        if len(price_history) < self.rsi_period + 2:
            return False  # Not enough data for crossover detection
        
        prev_rsi = self._calculate_rsi(price_history[:-1])
        if prev_rsi is None:
            return False
        
        # Check for oversold crossover: RSI crosses below oversold threshold
        oversold_crossover = (
            prev_rsi >= self.oversold_threshold and  # Was above or equal to threshold
            current_rsi < self.oversold_threshold    # Now below threshold (oversold)
        )
        
        return oversold_crossover
    
    def should_sell(self, price_history: List[Dict], current_data: Dict) -> bool:
        """
        Generate sell signal when RSI crosses above overbought threshold.
        
        Args:
            price_history: Historical price data
            current_data: Current market data
            
        Returns:
            True if should sell (RSI overbought condition detected)
        """
        # Need enough data for RSI calculation
        if len(price_history) < self.rsi_period + 1:
            return False
        
        # Calculate current and previous RSI
        current_rsi = self._calculate_rsi(price_history)
        if current_rsi is None:
            return False
        
        # Get previous RSI for crossover detection
        if len(price_history) < self.rsi_period + 2:
            return False  # Not enough data for crossover detection
        
        prev_rsi = self._calculate_rsi(price_history[:-1])
        if prev_rsi is None:
            return False
        
        # Check for overbought crossover: RSI crosses above overbought threshold
        overbought_crossover = (
            prev_rsi <= self.overbought_threshold and  # Was below or equal to threshold
            current_rsi > self.overbought_threshold    # Now above threshold (overbought)
        )
        
        return overbought_crossover
    
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
    
    def _calculate_rsi(self, price_history: List[Dict]) -> Optional[float]:
        """
        Calculate Relative Strength Index (RSI) for the given price history.
        
        Args:
            price_history: List of price data dictionaries with 'close' prices
            
        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(price_history) < self.rsi_period + 1:
            return None
        
        # Get closing prices
        closes = [day['close'] for day in price_history]
        
        # Calculate price changes
        deltas = []
        for i in range(1, len(closes)):
            deltas.append(closes[i] - closes[i-1])
        
        # Need at least rsi_period deltas for calculation
        if len(deltas) < self.rsi_period:
            return None
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate initial average gain and loss (first rsi_period values)
        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period
        
        # Avoid division by zero
        if avg_loss == 0:
            return 100.0  # RSI = 100 when there are no losses
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


# Strategy information for external tools
STRATEGY_INFO = {
    "name": "RSI Momentum",
    "description": "RSI-based momentum strategy using overbought/oversold signals",
    "version": "1.0",
    "author": "Trading System",
    "strategy_class": RSIStrategy
}
