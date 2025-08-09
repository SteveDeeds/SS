"""
Trading Simulator Engine
"""
from datetime import datetime
from typing import Dict, List, Any
import uuid
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from orders import OrderManager, Order

DEBUG = False

class ScenarioResult:
    """Results from a single scenario execution"""
    
    def __init__(self, scenario_id, strategy_name, final_portfolio_value, 
                 total_return, trades_executed, orders_placed, daily_values):
        self.scenario_id = scenario_id
        self.strategy_name = strategy_name
        self.final_portfolio_value = final_portfolio_value
        self.total_return = total_return
        self.trades_executed = trades_executed
        self.orders_placed = orders_placed
        self.daily_values = daily_values


class TradingSimulator:
    """
    Main trading simulator with realistic order management
    """
    
    def __init__(self, commission_per_trade: float = 1.0, slippage_pct: float = 0.001):
        # Import here to avoid circular imports
        from portfolio.manager import PortfolioManager
        
        self.portfolio_manager = PortfolioManager()
        self.order_manager = OrderManager(commission_per_trade=commission_per_trade, slippage_pct=slippage_pct)
        self.slippage_pct = slippage_pct  # Default 0.1% slippage
    
    def run_single_scenario(self, composite_strategy, price_history: List[Dict], 
                           initial_capital: float = 100000, symbol: str = 'SPY',
                           warmup_days: int = 252) -> ScenarioResult:
        """
        Run strategy on historical price data
        
        Args:
            composite_strategy: The trading strategy to execute
            price_history: Historical OHLCV data (full 2 years)
            initial_capital: Starting capital amount
            symbol: Stock symbol being traded
            warmup_days: Number of days to use for indicator warmup (default: 252 = ~1 year)
        """
        # Initialize portfolio
        portfolio = self.portfolio_manager.initialize_portfolio(
            initial_capital=initial_capital,
            symbols=[symbol]
        )
        
        # Track orders and trades
        all_orders = []
        all_trades = []
        daily_portfolio_values = []
        
        # print(f"🔄 Starting simulation with {warmup_days} warmup days...")
        # print(f"📊 Total data: {len(price_history)} days")
        # print(f"🎯 Active trading period: {len(price_history) - warmup_days} days")
        
        # Simulate trading day by day
        for day_index, current_market_data in enumerate(price_history):
            # Calculate simulation day index (relative to start of active trading)
            simulation_day = day_index - warmup_days
            is_warmup_period = day_index < warmup_days
            # Get historical context up to current day
            historical_context = price_history[:day_index + 1]
            
            # Add symbol to market data if not present
            if 'symbol' not in current_market_data:
                current_market_data['symbol'] = symbol
            
            # Create simulation time from market data (should be datetime object from loader)
            market_date = current_market_data.get('date')
            if not isinstance(market_date, datetime):
                # Fallback if date is string or missing
                if isinstance(market_date, str):
                    simulation_time = datetime.strptime(market_date, '%Y-%m-%d').replace(hour=16, minute=0, second=0, microsecond=0)
                else:
                    raise ValueError(f"Invalid or missing date in market data for day {day_index}: {current_market_data}")
            else:
                simulation_time = market_date  # Already a datetime object with market close time
            
            # Execute any pending orders from previous day at market open (9:30 AM)
            if day_index > 0:  # Can't execute orders on first day
                # Create market open time for order execution (9:30 AM on current day)
                execution_time = simulation_time.replace(hour=9, minute=30, second=0, microsecond=0)
                
                executed_trades = self.order_manager.process_pending_orders(
                    market_data=current_market_data,
                    portfolio=portfolio.get_current_state(),
                    current_time=execution_time,
                    debug=DEBUG
                )
                
                # Update portfolio with executed trades
                for trade in executed_trades:
                    success = self.portfolio_manager.execute_trade(portfolio, trade)
                    if success:
                        all_trades.append(trade)
                        if DEBUG:
                            direction = trade.get('direction', 'UNKNOWN')
                            symbol = trade.get('symbol', 'UNKNOWN')
                            quantity = trade.get('quantity', 0)
                            price = trade.get('execution_price', 0)
                            print(f"🎯 ORDER EXECUTED: {direction} {quantity} shares of {symbol} at ${price:.2f}")
                            
                            # Show portfolio state after execution
                            current_state = portfolio.get_current_state()
                            cash_balance = current_state.get('cash_balance', 0)
                            holdings = current_state.get('holdings', {})
                            print(f"   📅 Simulation Time: {execution_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"   💰 Cash Balance: ${cash_balance:.2f}")
                            
                            if holdings:
                                for holding_symbol, holding_data in holdings.items():
                                    qty = holding_data.get('quantity', 0)
                                    avg_cost = holding_data.get('average_cost', 0)
                                    print(f"   📊 Holdings {holding_symbol}: {qty} shares @ ${avg_cost:.2f} avg cost")
                            else:
                                print(f"   📊 Holdings: None")
                            print(f"   " + "="*60)
            
            # Only generate new orders after warmup period
            if not is_warmup_period:
                # Get current portfolio state and add pending orders information
                portfolio_state = portfolio.get_current_state()
                
                # Add pending orders as actual Order objects for strategy decision making
                portfolio_state['pending_orders'] = self.order_manager.get_pending_orders()
                
                # Generate trading signals and orders based on current day's data
                orders = composite_strategy.evaluate_and_place_orders(
                    price_history=historical_context,
                    portfolio=portfolio_state,
                    current_market_data=current_market_data
                )
                
                # Place Order objects directly (placement time already set in strategy adapter)
                if orders:
                    for order in orders:
                        self.order_manager.place_order(order)
                        all_orders.append(order)  # Keep Order objects for results
                        if DEBUG:
                            order_type = order.type or 'UNKNOWN'
                            symbol = order.symbol or 'UNKNOWN'
                            quantity = order.quantity or 0
                            limit_price = order.limit_price
                            price_info = f" at ${limit_price:.2f}" if limit_price else " (market price)"
                            print(f"📋 ORDER PLACED: {order_type} {quantity} shares of {symbol}{price_info}")
            
            # Update portfolio value with current market prices (always)
            market_prices = {symbol: current_market_data['close']}
            self.portfolio_manager.update_market_values(portfolio, market_prices)
            
            # Calculate portfolio value directly in engine using simulation market data
            current_cash_balance = portfolio.cash_balance
            current_market_price = current_market_data['close']
            
            # Calculate holdings value using current simulation price
            holdings_value = 0.0
            holdings_quantity = 0
            
            # Calculate total value of all holdings at current market price
            for holding_symbol, holding in portfolio.holdings.items():
                if holding_symbol == symbol:  # For this simulation's symbol
                    quantity = holding['quantity']
                    holdings_value += quantity * current_market_price
                    holdings_quantity += quantity
                # Note: In multi-symbol portfolios, we'd need market prices for other symbols too
            
            # Total portfolio value = cash + holdings at current market price
            current_portfolio_value = current_cash_balance + holdings_value
            
            # Record daily snapshot with detailed tracking
            daily_snapshot = {
                'date': simulation_time,  # Use consistent datetime object
                'portfolio_value': current_portfolio_value,
                'cash_balance': current_cash_balance,
                'market_price': current_market_price,
                'day_index': day_index,  # Original day index in price_history
                'simulation_day': simulation_day,  # Day index relative to start of trading (negative during warmup)
                'is_warmup': is_warmup_period,
                'holdings_count': len(portfolio.holdings),
                'holdings_value': holdings_value,
                'holdings_quantity': holdings_quantity
            }
            daily_portfolio_values.append(daily_snapshot)
        
        # END OF SIMULATION CLEANUP
        # Liquidate remaining holdings and resolve pending orders
        if DEBUG:
            print(f"\n🧹 SIMULATION CLEANUP:")
        
        # 1. Sell any remaining holdings at final market price
        final_market_data = price_history[-1]  # Last day's data
        final_market_price = final_market_data.get('close', 0)
        
        remaining_holdings = portfolio.get_current_state().get('holdings', {})
        # Create a copy of holdings to avoid "dictionary changed size during iteration" error
        holdings_to_liquidate = dict(remaining_holdings)
        
        for symbol_held, holding_data in holdings_to_liquidate.items():
            quantity = holding_data.get('quantity', 0)
            if quantity > 0:
                # Create a liquidation trade at final market price
                liquidation_trade = {
                    'order_id': f'LIQUIDATION_{symbol_held}_{quantity}',
                    'symbol': symbol_held,
                    'quantity': quantity,
                    'execution_price': final_market_price,
                    'commission': 0.0,  # No commission for liquidation
                    'direction': 'SELL',
                    'executed_at': final_market_data.get('date'),
                    'order_type': 'LIQUIDATION'
                }
                
                success = self.portfolio_manager.execute_trade(portfolio, liquidation_trade)
                if success and DEBUG:
                    proceeds = quantity * final_market_price
                    print(f"   💰 LIQUIDATED: {quantity} shares of {symbol_held} at ${final_market_price:.2f} = ${proceeds:.2f}")
        
        # 2. Cancel all pending orders
        pending_orders = self.order_manager.get_pending_orders()
        canceled_count = 0
        for order in pending_orders:
            if order.status == 'PENDING':
                order.cancel()
                self.order_manager.canceled_orders.append(order)
                canceled_count += 1
        
        # Clear pending orders list
        self.order_manager.pending_orders = []
        
        if DEBUG and canceled_count > 0:
            print(f"   ❌ CANCELED: {canceled_count} pending orders")
        
        if DEBUG:
            final_cash = portfolio.get_current_state().get('cash_balance', 0)
            print(f"   💰 FINAL CASH BALANCE: ${final_cash:.2f}")
            print(f"   📊 FINAL HOLDINGS: None (all liquidated)")
        
        # Calculate final results
        final_value = portfolio.get_total_value()
        total_return = (final_value / initial_capital) - 1
        
        # Debug: Print complete order summary
        if DEBUG:
            print(f"\n" + "="*80)
            print(f"📋 SCENARIO COMPLETE - ORDER SUMMARY")
            print(f"="*80)
            print(f"Total Orders Placed: {len(all_orders)}")
            print(f"Total Trades Executed: {len(all_trades)}")
            print(f"Final Portfolio Value: ${final_value:.2f}")
            print(f"Total Return: {total_return:.2%}")
            print(f"\n📋 COMPLETE ORDER LIST:")
            
            if all_orders:
                for i, order in enumerate(all_orders, 1):
                    order_type = order.type or 'UNKNOWN'
                    symbol = order.symbol or 'UNKNOWN'
                    quantity = order.quantity or 0
                    limit_price = order.limit_price
                    placement_time = order.datetime_placed
                    execution_price = order.execution_price
                    
                    # Format placement time
                    placement_str = placement_time.strftime('%Y-%m-%d %H:%M:%S') if placement_time else 'Unknown'
                    
                    # Format price info
                    if limit_price:
                        price_info = f" at ${limit_price:.2f}"
                    else:
                        price_info = " (market price)"
                    
                    # Show execution status using actual order status
                    if order.status == 'EXECUTED':
                        status = f"✅ EXECUTED at ${order.execution_price:.2f}"
                    elif order.status == 'PENDING':
                        status = "⏳ PENDING"
                    elif order.status == 'EXPIRED':
                        status = "⏰ EXPIRED"
                    elif order.status == 'CANCELED':
                        status = "❌ CANCELED"
                    else:
                        status = f"❓ {order.status}"
                    
                    print(f"   {i:2d}. {order_type} {quantity} shares of {symbol}{price_info} - {status}")
                    print(f"       Placed: {placement_str}")
            else:
                print(f"   No orders were placed during this scenario.")
            
            print(f"="*80)
        
        return ScenarioResult(
            scenario_id=0,
            strategy_name=composite_strategy.name,
            final_portfolio_value=final_value,
            total_return=total_return,
            trades_executed=all_trades,
            orders_placed=all_orders,
            daily_values=daily_portfolio_values
        )
