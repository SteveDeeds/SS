#!/usr/bin/env python3
"""
Mobile-Friendly Real-Time Trading Signals

This script loads strategy configurations from the best_strategy_per_symbol.csv file
and generates real-time trading signals displayed in a mobile-optimized HTML format.

Features:
- Loads optimal strategies and parameters from CSV results
- Single-column layout optimized for mobile viewing
- Real-time signal generation with detailed analysis
- Color-coded signals (Green=BUY, Red=SELL, Gray=HOLD)
- Performance metrics from optimization results
"""

import sys
import os
import csv
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import webbrowser
import tempfile

# Add project paths for imports
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
strategies_path = os.path.join(project_root, 'strategies')

if src_path not in sys.path:
    sys.path.insert(0, src_path)
if strategies_path not in sys.path:
    sys.path.insert(0, strategies_path)

# Import from src infrastructure
from src.data.loader import get_symbol_data, download_symbol_data

# Import strategies
from strategies.rsi_strategy import RSIStrategy
from strategies.adaptive_ma_crossover import AdaptiveMAStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy


class StrategyConfigLoader:
    """Loads strategy configurations from the best_strategy_per_symbol.csv file"""
    
    def __init__(self, csv_file_path: str):
        """Initialize with path to the CSV results file"""
        self.csv_file_path = csv_file_path
        self.configurations = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load strategy configurations from CSV file"""
        try:
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    symbol = row['symbol']
                    strategy_file = row['strategy_file']
                    
                    # Parse strategy parameters from the parameter_combination column
                    param_combination = json.loads(row['parameter_combination'].replace("'", '"'))
                    
                    # Determine strategy class based on file name
                    if 'rsi_strategy' in strategy_file:
                        strategy_class = RSIStrategy
                    elif 'adaptive_ma_crossover' in strategy_file:
                        strategy_class = AdaptiveMAStrategy
                    elif 'bollinger_bands_strategy' in strategy_file:
                        strategy_class = BollingerBandsStrategy
                    else:
                        print(f"Unknown strategy file: {strategy_file}, skipping {symbol}")
                        continue
                    
                    # Clean up strategy name to remove redundant parameter info
                    strategy_name = row['strategy_name']
                    # Remove the part in parentheses at the end (e.g., "(period=30, std_positive=2.0, ...)")
                    if ')(' in strategy_name:
                        strategy_name = strategy_name.split(')(')[0] + ')'
                    
                    # Create configuration
                    config = {
                        'strategy_class': strategy_class,
                        'strategy_params': param_combination,
                        'strategy_name': strategy_name,
                        'performance_metrics': {
                            'average_return': float(row['average_return']),
                            'win_rate': float(row['win_rate']),
                            'sharpe_ratio': float(row['sharpe_ratio']),
                            'total_trades': int(row['total_trades']),
                            'return_on_avg_invested_capital': float(row['return_on_avg_invested_capital'])
                        }
                    }
                    
                    self.configurations[symbol] = config
                    
            print(f"Loaded {len(self.configurations)} strategy configurations")
            
        except Exception as e:
            print(f"Error loading configurations: {e}")
            self.configurations = {}
    
    def get_config(self, symbol: str) -> Optional[Dict]:
        """Get configuration for a specific symbol"""
        return self.configurations.get(symbol.upper())
    
    def get_all_symbols(self) -> List[str]:
        """Get all configured symbols"""
        return list(self.configurations.keys())


class MobileSignalGenerator:
    """Generates trading signals optimized for mobile display"""
    
    def __init__(self, config_loader: StrategyConfigLoader):
        """Initialize with strategy configuration loader"""
        self.config_loader = config_loader
        self.signals_data = []
    
    def _load_historical_data(self, symbol: str) -> List[Dict]:
        """Load historical data for a symbol"""
        try:
            data = get_symbol_data(symbol, period="1y", force_download=False)
            if not data:
                print(f"No cached data found for {symbol}, downloading...")
                data = download_symbol_data(symbol, period="1y", save_to_file=True)
            return data
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return []
    
    def get_signal_for_symbol(self, symbol: str) -> Dict:
        """Get current trading signal for a symbol"""
        config = self.config_loader.get_config(symbol)
        if not config:
            return {
                'symbol': symbol,
                'signal': 'ERROR',
                'error': f'No configuration found for {symbol}'
            }
        
        # Load historical data
        historical_data = self._load_historical_data(symbol)
        if not historical_data:
            return {
                'symbol': symbol,
                'signal': 'ERROR',
                'error': f'No historical data available for {symbol}'
            }
        
        # Initialize strategy
        strategy_class = config['strategy_class']
        strategy_params = config['strategy_params']
        strategy = strategy_class(**strategy_params)
        
        # Get latest data point
        latest_data = historical_data[-1].copy()
        latest_data['date'] = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Determine minimum required data points
        min_required = self._get_min_required_days(strategy)
        if len(historical_data) < min_required:
            return {
                'symbol': symbol,
                'signal': 'ERROR',
                'error': f'Insufficient data (need {min_required}, have {len(historical_data)})'
            }
        
        # Generate signals
        should_buy = strategy.should_buy(historical_data, latest_data)
        should_sell = strategy.should_sell(historical_data, latest_data)
        
        # Determine signal
        if should_buy:
            signal = 'BUY'
            signal_color = '#28a745'  # Green
        elif should_sell:
            signal = 'SELL'
            signal_color = '#dc3545'  # Red
        else:
            signal = 'HOLD'
            signal_color = '#6c757d'  # Gray
        
        # Get strategy-specific details
        details = self._get_strategy_details(strategy, historical_data, latest_data)
        
        # Calculate price transitions
        transitions = self._calculate_price_transitions(strategy, historical_data, latest_data['close'])
        
        return {
            'symbol': symbol,
            'signal': signal,
            'signal_color': signal_color,
            'current_price': latest_data['close'],
            'strategy_name': config['strategy_name'],
            'performance': config['performance_metrics'],
            'details': details,
            'transitions': transitions,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _get_min_required_days(self, strategy) -> int:
        """Get minimum required days for strategy calculations"""
        if hasattr(strategy, 'rsi_period'):
            return strategy.rsi_period + 5
        elif hasattr(strategy, 'buy_slow_period'):
            return max(strategy.buy_slow_period, strategy.sell_slow_period) + 5
        elif hasattr(strategy, 'period'):
            return strategy.period + 5
        else:
            return 50  # Conservative default
    
    def _get_strategy_details(self, strategy, price_history: List[Dict], latest_data: Dict) -> Dict:
        """Get strategy-specific indicator values"""
        details = {}
        
        try:
            if hasattr(strategy, 'rsi_period'):
                # RSI Strategy
                rsi_value = strategy._calculate_rsi(price_history)
                details.update({
                    'rsi_value': round(rsi_value, 2) if rsi_value else None,
                    'oversold_threshold': strategy.oversold_threshold,
                    'overbought_threshold': strategy.overbought_threshold,
                    'rsi_period': strategy.rsi_period
                })
            
            elif hasattr(strategy, 'buy_slow_period'):
                # Adaptive MA Strategy
                buy_fast_ma = strategy._calculate_moving_average(price_history, strategy.buy_fast_period)
                buy_slow_ma = strategy._calculate_moving_average(price_history, strategy.buy_slow_period)
                details.update({
                    'buy_fast_ma': round(buy_fast_ma, 2) if buy_fast_ma else None,
                    'buy_slow_ma': round(buy_slow_ma, 2) if buy_slow_ma else None,
                    'buy_fast_period': strategy.buy_fast_period,
                    'buy_slow_period': strategy.buy_slow_period
                })
            
            elif hasattr(strategy, 'period'):
                # Bollinger Bands Strategy
                bands = strategy._calculate_bollinger_bands(price_history)
                if bands:
                    upper_band, middle_band, lower_band = bands
                    details.update({
                        'upper_band': round(upper_band, 2),
                        'middle_band': round(middle_band, 2),
                        'lower_band': round(lower_band, 2),
                        'period': strategy.period,
                        'std_positive': strategy.std_positive,
                        'std_negative': strategy.std_negative
                    })
        
        except Exception as e:
            details['calculation_error'] = str(e)
        
        return details
    
    def _calculate_price_transitions(self, strategy, price_history: List[Dict], current_price: float) -> Dict:
        """Calculate buy/sell transition prices using hypothetical scenarios"""
        transitions = {
            'buy_threshold': None,
            'sell_threshold': None,
            'buy_change_pct': None,
            'sell_change_pct': None
        }
        
        try:
            # Test scenarios from -20% to +20% in 0.5% steps for performance
            scenario_results = []
            
            for change_pct_halves in range(-40, 41, 1):  # -20% to +20% in 0.5% steps
                change_pct = change_pct_halves / 2.0
                test_price = current_price * (1.0 + change_pct / 100.0)
                
                # Create test data point with new price
                test_data = price_history[-1].copy()
                test_data['close'] = test_price
                test_data['high'] = max(test_data['high'], test_price)
                test_data['low'] = min(test_data['low'], test_price)
                
                # Simple approach: add one day with the modified price
                test_history = price_history + [test_data]
                
                should_buy = strategy.should_buy(test_history, test_data)
                should_sell = strategy.should_sell(test_history, test_data)
                
                if should_buy:
                    signal = 'BUY'
                elif should_sell:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                scenario_results.append({
                    'price': test_price,
                    'change_pct': change_pct,
                    'signal': signal
                })
            
            # Find transition points
            signal_transitions = []
            for i in range(1, len(scenario_results)):
                prev_signal = scenario_results[i-1]['signal']
                curr_signal = scenario_results[i]['signal']
                
                if prev_signal != curr_signal:
                    signal_transitions.append({
                        'from': prev_signal,
                        'to': curr_signal,
                        'price': scenario_results[i]['price'],
                        'change_pct': scenario_results[i]['change_pct']
                    })
            
            # Find BUY and SELL thresholds
            buy_transitions = [t for t in signal_transitions if t['to'] == 'BUY' or t['from'] == 'BUY']
            if buy_transitions:
                # Use the first BUY transition (typically the most relevant)
                buy_trans = buy_transitions[0]
                transitions['buy_threshold'] = buy_trans['price']
                transitions['buy_change_pct'] = buy_trans['change_pct']
            
            sell_transitions = [t for t in signal_transitions if t['to'] == 'SELL' or t['from'] == 'SELL']
            if sell_transitions:
                # Use the first SELL transition (typically the most relevant)
                sell_trans = sell_transitions[0]
                transitions['sell_threshold'] = sell_trans['price']
                transitions['sell_change_pct'] = sell_trans['change_pct']
                
        except Exception as e:
            # Silently handle errors to avoid breaking the main functionality
            pass
        
        return transitions
    
    def generate_all_signals(self) -> List[Dict]:
        """Generate signals for all configured symbols"""
        symbols = self.config_loader.get_all_symbols()
        self.signals_data = []
        
        print(f"Generating signals for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"  [{i}/{len(symbols)}] Processing {symbol}...")
            signal_data = self.get_signal_for_symbol(symbol)
            self.signals_data.append(signal_data)
        
        # Sort by signal priority (BUY first, then SELL, then HOLD)
        signal_priority = {'BUY': 1, 'SELL': 2, 'HOLD': 3, 'ERROR': 4}
        self.signals_data.sort(key=lambda x: (signal_priority.get(x['signal'], 5), x['symbol']))
        
        return self.signals_data


class MobileHTMLGenerator:
    """Generates mobile-optimized HTML display for trading signals"""
    
    @staticmethod
    def generate_html(signals_data: List[Dict]) -> str:
        """Generate mobile-optimized HTML page"""
        
        # Count signals by type
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'ERROR': 0}
        for signal in signals_data:
            signal_counts[signal['signal']] += 1
        
        # Generate timestamp
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Signals - Mobile</title>
    <link rel="stylesheet" href="mobile_signals.css">
</head>
<body>
    <div class="header">
        <h1>Trading Signals</h1>
        <div class="timestamp">Last Updated: {current_time}</div>
    </div>
    
    <div class="summary">
        <div class="summary-card buy">
            <div class="count">{signal_counts['BUY']}</div>
            <div class="label">Buy</div>
        </div>
        <div class="summary-card sell">
            <div class="count">{signal_counts['SELL']}</div>
            <div class="label">Sell</div>
        </div>
        <div class="summary-card hold">
            <div class="count">{signal_counts['HOLD']}</div>
            <div class="label">Hold</div>
        </div>
        <div class="summary-card error">
            <div class="count">{signal_counts['ERROR']}</div>
            <div class="label">Error</div>
        </div>
    </div>
"""
        
        # Add signal cards
        for signal_data in signals_data:
            html_content += MobileHTMLGenerator._generate_signal_card(signal_data)
        
        html_content += """
    <div class="refresh-info">
        Refresh this page to get updated signals<br>
        Based on optimized strategies from historical backtesting
    </div>

</body>
</html>
"""
        return html_content
    
    @staticmethod
    def _generate_signal_card(signal_data: Dict) -> str:
        """Generate HTML for a single signal card"""
        symbol = signal_data['symbol']
        signal = signal_data['signal']
        
        if signal == 'ERROR':
            return f"""
    <div class="signal-card">
        <div class="signal-header error">
            <div>
                <div class="symbol">{symbol}</div>
            </div>
            <div class="signal-badge" style="background-color: #ffc107;">ERROR</div>
        </div>
        <div class="signal-body">
            <div class="error-message">{signal_data.get('error', 'Unknown error')}</div>
        </div>
    </div>
"""
        
        signal_color = signal_data['signal_color']
        current_price = signal_data['current_price']
        strategy_name = signal_data['strategy_name']
        performance = signal_data['performance']
        details = signal_data['details']
        transitions = signal_data.get('transitions', {})
        
        # Format performance metrics
        avg_return_pct = performance['average_return'] * 100
        win_rate_pct = performance['win_rate'] * 100
        sharpe_ratio = performance['sharpe_ratio']
        roi_pct = performance['return_on_avg_invested_capital'] * 100
        
        # Generate details section
        details_html = ""
        if details:
            if 'rsi_value' in details and details['rsi_value'] is not None:
                details_html = f"""RSI: {details['rsi_value']:.1f} (OS: {details['oversold_threshold']}, OB: {details['overbought_threshold']})"""
            elif 'buy_fast_ma' in details:
                details_html = f"""MA Fast: ${details['buy_fast_ma']:.2f}, MA Slow: ${details['buy_slow_ma']:.2f}"""
            elif 'upper_band' in details:
                details_html = f"""BB Upper: ${details['upper_band']:.2f}, Lower: ${details['lower_band']:.2f}"""
        
        # Generate transitions section (always show for all symbols)
        transitions_html = f"""
            <div class="transitions-section">
                <div class="transitions-title">Price Transitions</div>
                <div class="transitions-grid">"""
        
        # Buy threshold
        if transitions and transitions.get('buy_threshold'):
            buy_price = transitions['buy_threshold']
            buy_change = transitions.get('buy_change_pct', 0)
            change_class = 'negative' if buy_change < 0 else 'positive'
            transitions_html += f"""
                    <div class="transition-card buy-threshold">
                        <span class="transition-label">Buy Signal At</span>
                        <span class="transition-price">${buy_price:.2f}</span>
                        <span class="transition-change {change_class}">{buy_change:+.1f}%</span>
                    </div>"""
        else:
            transitions_html += f"""
                    <div class="transition-card buy-threshold">
                        <span class="transition-label">Buy Signal At</span>
                        <span class="transition-price">No signal in range</span>
                        <span class="transition-change">±20% tested</span>
                    </div>"""
        
        # Sell threshold
        if transitions and transitions.get('sell_threshold'):
            sell_price = transitions['sell_threshold']
            sell_change = transitions.get('sell_change_pct', 0)
            change_class = 'positive' if sell_change > 0 else 'negative'
            transitions_html += f"""
                    <div class="transition-card sell-threshold">
                        <span class="transition-label">Sell Signal At</span>
                        <span class="transition-price">${sell_price:.2f}</span>
                        <span class="transition-change {change_class}">{sell_change:+.1f}%</span>
                    </div>"""
        else:
            transitions_html += f"""
                    <div class="transition-card sell-threshold">
                        <span class="transition-label">Sell Signal At</span>
                        <span class="transition-price">No signal in range</span>
                        <span class="transition-change">±20% tested</span>
                    </div>"""
        
        transitions_html += """
                </div>
            </div>"""
        
        signal_class = signal.lower()
        
        return f"""
    <div class="signal-card">
        <div class="signal-header {signal_class}">
            <div>
                <div class="symbol">{symbol}</div>
                <div class="price">${current_price:.2f}</div>
            </div>
            <div class="signal-badge" style="background-color: {signal_color};">{signal}</div>
        </div>
        <div class="signal-body">
            <div class="strategy-name">{strategy_name}</div>
            <div class="metrics">
                <div class="metric">
                    <span class="metric-label">Avg Return</span>
                    <span class="metric-value">{avg_return_pct:+.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Rate</span>
                    <span class="metric-value">{win_rate_pct:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sharpe</span>
                    <span class="metric-value">{sharpe_ratio:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">ROI</span>
                    <span class="metric-value">{roi_pct:.1f}%</span>
                </div>
            </div>
            {f'<div class="details">{details_html}</div>' if details_html else ''}
        </div>
        {transitions_html}
    </div>
"""


def main():
    """Main function to generate and display mobile trading signals"""
    print("MOBILE TRADING SIGNALS GENERATOR")
    print("=" * 50)
    
    # Configuration
    csv_file_path = os.path.join(project_root, 'results', 'best_strategy_per_symbol.csv')
    
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return
    
    try:
        # Load strategy configurations
        print("Loading strategy configurations...")
        config_loader = StrategyConfigLoader(csv_file_path)
        
        if not config_loader.configurations:
            print("No valid configurations found")
            return
        
        # Generate signals
        print("Generating trading signals...")
        signal_generator = MobileSignalGenerator(config_loader)
        signals_data = signal_generator.generate_all_signals()
        
        # Generate HTML
        print("Creating mobile HTML display...")
        html_content = MobileHTMLGenerator.generate_html(signals_data)
        
        # Save to temporary file and open in browser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_file_path = f.name
        
        print(f"Generated signals for {len(signals_data)} symbols")
        print(f"Opening HTML file: {temp_file_path}")
        
        # Open in default browser
        webbrowser.open('file://' + temp_file_path)
        
        # Also save a permanent copy
        output_file = os.path.join(project_root, 'mobile_signals.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Saved permanent copy: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
