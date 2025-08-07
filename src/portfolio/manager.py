"""
Portfolio Manager Implementation
"""
from typing import Dict, List, Optional, Any
import uuid


class Portfolio:
    """
    Portfolio class with standardized interface for compatibility
    """
    
    def __init__(self, portfolio_id: str, initial_capital: float, symbols: List[str]):
        self.portfolio_id = portfolio_id
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.symbols = symbols
        self.holdings = {}
        self.transaction_history = []
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Return portfolio state as dict for strategy compatibility
        """
        return {
            'portfolio_id': self.portfolio_id,
            'cash_balance': self.cash_balance,
            'holdings': self.holdings,
            'total_value': self.get_total_value(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'transaction_history': self.transaction_history
        }
    
    def get_total_value(self) -> float:
        """Return total portfolio value (cash + holdings market value)"""
        holdings_value = sum(
            holding.get('market_value', 0) for holding in self.holdings.values()
        )
        return self.cash_balance + holdings_value
    
    def get_unrealized_pnl(self) -> float:
        """Return total unrealized P&L across all holdings"""
        return sum(
            holding.get('unrealized_pnl', 0) for holding in self.holdings.values()
        )
    
    def update_market_values(self, market_data: Dict[str, float]):
        """
        Update current market values for all holdings
        """
        for symbol, holding in self.holdings.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                quantity = holding['quantity']
                
                # Update current market values
                holding['current_price'] = current_price
                holding['market_value'] = quantity * current_price
                
                # Calculate unrealized P&L
                cost_basis = quantity * holding['average_cost']
                holding['unrealized_pnl'] = holding['market_value'] - cost_basis
                holding['unrealized_pnl_percent'] = holding['unrealized_pnl'] / cost_basis if cost_basis > 0 else 0.0


class PortfolioManager:
    """
    Manages portfolio state, trade execution, and market value updates
    """
    
    def __init__(self):
        self.portfolios = {}  # Track multiple portfolios by ID
    
    def initialize_portfolio(self, initial_capital: float, symbols: List[str], 
                           portfolio_id: str = None) -> Portfolio:
        """
        Initialize a new portfolio with starting capital
        """
        if portfolio_id is None:
            portfolio_id = f"PORTFOLIO_{uuid.uuid4().hex[:8]}"
        
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            initial_capital=initial_capital,
            symbols=symbols
        )
        
        self.portfolios[portfolio_id] = portfolio
        return portfolio
    
    def execute_trade(self, portfolio: Portfolio, trade: Dict[str, Any]) -> bool:
        """
        Execute a trade against the portfolio
        """
        try:
            symbol = trade['symbol']
            quantity = trade['quantity']
            price = trade['execution_price']
            direction = trade['direction']  # 'BUY' or 'SELL'
            
            if direction == 'BUY':
                return self._execute_buy_trade(portfolio, symbol, quantity, price)
            elif direction == 'SELL':
                return self._execute_sell_trade(portfolio, symbol, quantity, price)
            else:
                raise ValueError(f"Invalid trade direction: {direction}")
                
        except Exception as e:
            print(f"Trade execution failed: {e}")
            return False
    
    def _execute_buy_trade(self, portfolio: Portfolio, symbol: str, 
                          quantity: int, price: float) -> bool:
        """Execute a buy trade"""
        total_cost = quantity * price
        
        # Check if sufficient cash available
        if portfolio.cash_balance < total_cost:
            print(f"🚫 BUY trade failed: Need ${total_cost:.2f}, only have ${portfolio.cash_balance:.2f}")
            return False
        
        # Update cash balance
        portfolio.cash_balance -= total_cost
        
        # Update holdings
        if symbol in portfolio.holdings:
            # Calculate new average cost
            existing_qty = portfolio.holdings[symbol]['quantity']
            existing_cost = portfolio.holdings[symbol]['average_cost']
            
            total_quantity = existing_qty + quantity
            total_cost_basis = (existing_qty * existing_cost) + (quantity * price)
            new_average_cost = total_cost_basis / total_quantity
            
            portfolio.holdings[symbol].update({
                'quantity': total_quantity,
                'average_cost': new_average_cost,
                'current_price': price,  # Update current price to trade price
                'market_value': total_quantity * price,  # Update market value
                'unrealized_pnl': (total_quantity * price) - (total_quantity * new_average_cost),
                'unrealized_pnl_percent': ((total_quantity * price) - (total_quantity * new_average_cost)) / (total_quantity * new_average_cost) if new_average_cost > 0 else 0.0
            })
        else:
            # New position
            portfolio.holdings[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'average_cost': price,
                'current_price': price,
                'market_value': quantity * price,
                'unrealized_pnl': 0.0,
                'unrealized_pnl_percent': 0.0
            }
        
        # Add to transaction history
        portfolio.transaction_history.append({
            'transaction_id': f"TXN_{uuid.uuid4().hex[:8]}",
            'symbol': symbol,
            'type': 'BUY',
            'quantity': quantity,
            'price': price,
            'total_cost': total_cost
        })
        
        return True
    
    def _execute_sell_trade(self, portfolio: Portfolio, symbol: str, 
                           quantity: int, price: float) -> bool:
        """Execute a sell trade"""
        # Check if sufficient shares available
        if symbol not in portfolio.holdings:
            print(f"🚫 SELL trade failed: No position in {symbol}")
            return False
        
        available_shares = portfolio.holdings[symbol]['quantity']
        if available_shares < quantity:
            print(f"🚫 SELL trade failed: Need {quantity} shares of {symbol}, only have {available_shares}")
            return False
        
        # Calculate proceeds
        proceeds = quantity * price
        
        # Update cash balance
        portfolio.cash_balance += proceeds
        
        # Update holdings
        portfolio.holdings[symbol]['quantity'] -= quantity
        
        # Remove position if fully sold
        if portfolio.holdings[symbol]['quantity'] == 0:
            del portfolio.holdings[symbol]
        
        # Add to transaction history
        portfolio.transaction_history.append({
            'transaction_id': f"TXN_{uuid.uuid4().hex[:8]}",
            'symbol': symbol,
            'type': 'SELL',
            'quantity': quantity,
            'price': price,
            'total_proceeds': proceeds
        })
        
        return True
    
    def update_market_values(self, portfolio: Portfolio, market_data: Dict[str, float]):
        """
        Update current market values for all holdings
        """
        portfolio.update_market_values(market_data)
