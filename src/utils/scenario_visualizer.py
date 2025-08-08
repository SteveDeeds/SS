"""
Scenario Visualization Utility

This utility creates visualizations using the exact scenario data from simulations,
ensuring that candlesticks and order markers are perfectly aligned.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Add src to path and import Order class for type hints and auto-completion
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from orders.order import Order

DEBUG = False

class ScenarioVisualizer:
    """
    Creates visualizations using the exact scenario data from simulations
    """
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
        # Professional styling
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.titlesize': 16
        })
        
        # Color scheme
        self.colors = {
            'up_candle': '#00C851',      # Green for up days
            'down_candle': '#FF4444',    # Red for down days
            'buy_marker': '#00C851',     # Green for buy orders
            'sell_marker': '#FF4444',    # Red for sell orders
            'portfolio': '#2196F3',      # Blue for portfolio line
            'cash': '#FF9800',           # Orange for cash line
            'background': 'white'
        }
    
    def visualize_scenario_result(self, results: Dict, scenario_index: int = 0, 
                                strategy_name: str = None, save_path: str = None) -> plt.Figure:
        """
        Create a comprehensive visualization using actual scenario data
        
        Args:
            results: Results dictionary from grid search optimization
            scenario_index: Index of scenario to visualize (default: 0 = first scenario)
            strategy_name: Name of strategy to visualize (if None, uses best strategy)
            save_path: Optional path to save the chart
            
        Returns:
            matplotlib Figure object
        """
        print(f"🎨 Creating scenario visualization...")
        
        # Extract data from results
        scenarios = results.get('scenarios', [])
        if not scenarios:
            raise ValueError("No scenario data found in results. Make sure optimization was run with scenario storage.")
        
        if scenario_index >= len(scenarios):
            raise ValueError(f"Scenario index {scenario_index} out of range. Available scenarios: 0-{len(scenarios)-1}")
        
        # Get the scenario data (actual OHLCV data used in simulation)
        scenario_data = scenarios[scenario_index]
        print(f"   📊 Using scenario data with {len(scenario_data)} days")
        
        # Get strategy results
        if strategy_name is None:
            strategy_name = results['best_strategy']['name']
        
        scenario_results_map = results['scenario_results_map']
        if strategy_name not in scenario_results_map:
            available_strategies = list(scenario_results_map.keys())
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available_strategies}")
        
        strategy_results = scenario_results_map[strategy_name]
        if scenario_index >= len(strategy_results):
            raise ValueError(f"Scenario {scenario_index} not found for strategy '{strategy_name}'")
        
        scenario_result = strategy_results[scenario_index]
        print(f"   🎯 Strategy: {scenario_result.strategy_name}")
        print(f"   📈 Return: {scenario_result.total_return:.2%}")
        print(f"   📋 Orders: {len(scenario_result.orders_placed)}")
        
        # Create the visualization
        fig = self._create_chart(scenario_data, scenario_result, results['symbol'])
        
        # Save if requested
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Chart saved to: {save_path}")
        
        return fig
    
    def _create_chart(self, scenario_data: List[Dict], scenario_result, symbol: str) -> plt.Figure:
        """Create the actual chart with scenario data and orders"""
        
        # Prepare data - ensure all dates are datetime objects
        dates = []
        for day in scenario_data:
            date_obj = day['date']
            # Convert to datetime if needed, preserving existing datetime objects
            if isinstance(date_obj, datetime):
                dates.append(date_obj)
            else:
                # Handle legacy date objects by adding market close time
                dates.append(datetime.combine(date_obj, datetime.min.time().replace(hour=16, minute=0)))
        
        opens = [day['open'] for day in scenario_data]
        highs = [day['high'] for day in scenario_data]
        lows = [day['low'] for day in scenario_data]
        closes = [day['close'] for day in scenario_data]
        
        # Portfolio data - ensure datetime objects
        daily_values = scenario_result.daily_values
        portfolio_dates = []
        for day in daily_values:
            date_obj = day['date']
            # Convert to datetime if needed, preserving existing datetime objects
            if isinstance(date_obj, datetime):
                portfolio_dates.append(date_obj)
            else:
                # Handle legacy date objects by adding market close time
                portfolio_dates.append(datetime.combine(date_obj, datetime.min.time().replace(hour=16, minute=0)))
        portfolio_values = [day['portfolio_value'] for day in daily_values]
        cash_balances = [day['cash_balance'] for day in daily_values]
        
        # Create figure
        fig, (ax_price, ax_portfolio) = plt.subplots(2, 1, figsize=self.figsize, 
                                                    gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart with candlesticks
        self._draw_candlesticks(ax_price, dates, opens, highs, lows, closes)
        
        # Add order markers
        self._add_order_markers(ax_price, scenario_data, scenario_result.orders_placed, dates)
        
        # Portfolio chart
        ax_portfolio.plot(portfolio_dates, portfolio_values, 
                         color=self.colors['portfolio'], linewidth=2, label='Total Portfolio Value')
        ax_portfolio.plot(portfolio_dates, cash_balances, 
                         color=self.colors['cash'], linewidth=2, label='Cash Balance')
        
        # Synchronize x-axis for both charts
        self._synchronize_x_axis(ax_price, ax_portfolio, dates, portfolio_dates)
        
        # Styling
        self._style_price_chart(ax_price, symbol, scenario_result)
        self._style_portfolio_chart(ax_portfolio)
        
        # Main title
        total_return = scenario_result.total_return
        fig.suptitle(f"{scenario_result.strategy_name} - {symbol} Trading Strategy (Return: {total_return:.2%})",
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
        
        return fig
    
    def _draw_candlesticks(self, ax, dates, opens, highs, lows, closes):
        """Draw candlestick chart"""
        for i, (date, open_price, high, low, close) in enumerate(zip(dates, opens, highs, lows, closes)):
            # Determine color
            color = self.colors['up_candle'] if close >= open_price else self.colors['down_candle']
            
            # Draw high-low line
            ax.plot([date, date], [low, high], color='black', linewidth=1)
            
            # Draw body rectangle
            body_height = abs(close - open_price)
            body_bottom = min(open_price, close)
            
            rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
    
    def _add_order_markers(self, ax, scenario_data, orders: List[Order], dates):
        """Add buy/sell order markers to the chart with proper limit order visualization"""
        if not orders:
            return
        if DEBUG:
            print(f"🔍 DEBUG: Processing {len(orders)} orders for markers")
        
        # Separate orders by type and fill status
        filled_buy_orders = []      # Solid green diamonds
        filled_sell_orders = []     # Solid red diamonds
        unfilled_buy_orders = []    # Hollow green diamonds (limit orders not filled)
        unfilled_sell_orders = []   # Hollow red diamonds (limit orders not filled)
        
        for order in orders:
            if DEBUG:
                print(f"📋 Processing order: Type={order.type}, Symbol={order.symbol}, Qty={order.quantity}, Status={order.status}")
            
            # Get order placement time - this should align with scenario dates
            placement_time = order.datetime_placed
            
            if not placement_time:
                print(f"⚠️ Order has no placement_time, skipping")
                continue
                
            # Ensure placement_time is datetime object
            if not isinstance(placement_time, datetime):
                print(f"⚠️ Order placement_time is not datetime: {type(placement_time)}")
                continue
            
            # Find matching date in scenario data using placement time
            best_match_date = None
            best_match_price = None
            
            for day in scenario_data:
                day_date = day['date']
                # Ensure day_date is datetime object
                if not isinstance(day_date, datetime):
                    day_date = datetime.combine(day_date, datetime.min.time().replace(hour=16, minute=0))
                
                # Match by date (ignore time for daily data)
                if day_date.date() == placement_time.date():
                    best_match_date = day_date
                    
                    # Determine price based on order type and fill status
                    if order.status == 'FILLED' and order.execution_price:
                        # Use actual execution price for filled orders
                        best_match_price = order.execution_price
                    elif order.limit_price and ('LIMIT' in order.type):
                        # Use limit price for limit orders (filled or unfilled)
                        best_match_price = order.limit_price
                    else:
                        # Fall back to market price for market orders
                        best_match_price = day['close']
                    break
            
            if best_match_date and best_match_price:
                if DEBUG:
                    print(f"✅ Matched order to {best_match_date.strftime('%Y-%m-%d')} at ${best_match_price:.2f}, Status: {order.status}")
                
                # Categorize orders by type and fill status
                is_filled = (order.status == 'FILLED')
                
                if 'BUY' in order.type:
                    if is_filled:
                        filled_buy_orders.append((best_match_date, best_match_price))
                    else:
                        unfilled_buy_orders.append((best_match_date, best_match_price))
                elif 'SELL' in order.type:
                    if is_filled:
                        filled_sell_orders.append((best_match_date, best_match_price))
                    else:
                        unfilled_sell_orders.append((best_match_date, best_match_price))
            else:
                print(f"❌ Could not match order placed on {placement_time.strftime('%Y-%m-%d')} to scenario data")
        
        # Count totals for logging
        total_buy = len(filled_buy_orders) + len(unfilled_buy_orders)
        total_sell = len(filled_sell_orders) + len(unfilled_sell_orders)
        print(f"📊 Final counts: {total_buy} buy orders ({len(filled_buy_orders)} filled, {len(unfilled_buy_orders)} unfilled), {total_sell} sell orders ({len(filled_sell_orders)} filled, {len(unfilled_sell_orders)} unfilled)")
        
        # Plot filled order markers (solid diamonds)
        if filled_buy_orders:
            buy_dates, buy_prices = zip(*filled_buy_orders)
            ax.scatter(buy_dates, buy_prices, color=self.colors['buy_marker'], 
                      marker='D', s=50, label=f'BUY Orders Filled ({len(filled_buy_orders)})', 
                      edgecolor='black', linewidth=1, zorder=5)
        
        if filled_sell_orders:
            sell_dates, sell_prices = zip(*filled_sell_orders)
            ax.scatter(sell_dates, sell_prices, color=self.colors['sell_marker'], 
                      marker='D', s=50, label=f'SELL Orders Filled ({len(filled_sell_orders)})', 
                      edgecolor='black', linewidth=1, zorder=5)
        
        # Plot unfilled order markers (hollow diamonds)
        if unfilled_buy_orders:
            buy_dates, buy_prices = zip(*unfilled_buy_orders)
            ax.scatter(buy_dates, buy_prices, facecolors='none', 
                      edgecolors=self.colors['buy_marker'], linewidth=2,
                      marker='D', s=50, label=f'BUY Orders Unfilled ({len(unfilled_buy_orders)})', 
                      zorder=5)
        
        if unfilled_sell_orders:
            sell_dates, sell_prices = zip(*unfilled_sell_orders)
            ax.scatter(sell_dates, sell_prices, facecolors='none', 
                      edgecolors=self.colors['sell_marker'], linewidth=2,
                      marker='D', s=50, label=f'SELL Orders Unfilled ({len(unfilled_sell_orders)})', 
                      zorder=5)
    
    def _synchronize_x_axis(self, ax_price, ax_portfolio, price_dates, portfolio_dates):
        """Synchronize x-axis limits for both charts using common date range"""
        # Find the common date range
        all_dates = price_dates + portfolio_dates
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Set the same x-axis limits for both charts
        ax_price.set_xlim(min_date, max_date)
        ax_portfolio.set_xlim(min_date, max_date)
        
        print(f"   📅 Synchronized x-axis: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    def _style_price_chart(self, ax, symbol, scenario_result):
        """Apply styling to price chart"""
        ax.set_title("Price Chart with Order Execution", fontweight='bold', pad=20)
        ax.set_ylabel("Price ($)", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _style_portfolio_chart(self, ax):
        """Apply styling to portfolio chart"""
        ax.set_title("Portfolio Performance", fontweight='bold', pad=20)
        ax.set_ylabel("Value ($)", fontweight='bold')
        ax.set_xlabel("Date", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
