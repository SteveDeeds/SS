#!/usr/bin/env python3
"""
Real-Time Trading Signals Example

This example loads real historical data from the last year, includes current day data,
and runs strategies to return BUY, HOLD, or SELL signals for current market conditions.

Uses the existing src infrastructure:
- Data loader for historical data
- Multiple strategies for signal generation
- Real-time signal evaluation with symbol-specific configurations
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from src infrastructure
from src.data.loader import get_symbol_data, download_symbol_data
from strategies.rsi_strategy import RSIStrategy

# ============================================================================
# CONFIGURATION - Symbol-Specific Strategies and Parameters
# ============================================================================

# Define strategies and parameters for each symbol
SYMBOL_CONFIGURATIONS = {
    "SDIV": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 16,
            'oversold_threshold': 5.0,
            'overbought_threshold': 65.0,
            'cash_percentage': 0.15
        },
        "description": "Global X SuperDividend ETF"
    },
    "SPXL": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 10,
            'oversold_threshold': 10.0,
            'overbought_threshold': 60.0,
            'cash_percentage': 0.15
        },
        "description": "Direxion Daily S&P 500 Bull 3X Shares"
    }    
}

# Default configuration for symbols not explicitly configured
DEFAULT_CONFIGURATION = {
    "strategy_class": RSIStrategy,
    "strategy_params": {
        'rsi_period': 14,
        'oversold_threshold': 30.0,
        'overbought_threshold': 70.0,
        'cash_percentage': 0.15
    },
    "description": "Default RSI strategy parameters"
}

# Global settings
GLOBAL_SETTINGS = {
    "data_period": "2y",  # How much historical data to load
    "analysis_days": 30,  # Days to analyze for signal distribution
    "force_download": False,  # Whether to force fresh data download
    "test_price_adjustment": 0.95  # Multiplier for custom price testing (5% lower)
}


class RealTimeSignalGenerator:
    """
    Generates real-time trading signals using historical data and current market conditions.
    
    This class loads historical data and evaluates strategy signals for current market 
    conditions, returning BUY, HOLD, or SELL recommendations using symbol-specific configurations.
    """
    
    def __init__(self, symbol: str = None, custom_config: Dict = None):
        """
        Initialize the signal generator with symbol-specific configuration.
        
        Args:
            symbol: Stock symbol to analyze (defaults to first symbol in SYMBOL_CONFIGURATIONS)
            custom_config: Optional custom configuration (overrides symbol-specific config)
        """
        # Use first symbol from configurations if none provided
        if symbol is None:
            symbol = list(SYMBOL_CONFIGURATIONS.keys())[0] if SYMBOL_CONFIGURATIONS else "SPY"
        
        self.symbol = symbol.upper()
        
        # Get configuration for this symbol
        if custom_config:
            self.config = custom_config
        else:
            self.config = SYMBOL_CONFIGURATIONS.get(self.symbol, DEFAULT_CONFIGURATION)
        
        self.strategy_params = self.config["strategy_params"]
        self.description = self.config["description"]
        
        # Initialize strategy with symbol-specific parameters
        strategy_class = self.config["strategy_class"]
        self.strategy = strategy_class(**self.strategy_params)
        
        # Load historical data
        # print(f"📡 Loading historical data for {self.symbol}...")
        # print(f"   Strategy: {self.description}")
        self.historical_data = self._load_historical_data()
        
        if not self.historical_data:
            raise ValueError(f"Failed to load historical data for {self.symbol}")
        
        # print(f"✅ Loaded {len(self.historical_data)} days of historical data")
    
    def _load_historical_data(self) -> List[Dict]:
        """Load historical data with sufficient buffer for indicators."""
        try:
            # Use global settings for data loading
            data = get_symbol_data(
                self.symbol, 
                period=GLOBAL_SETTINGS["data_period"], 
                force_download=GLOBAL_SETTINGS["force_download"]
            )
            
            if not data:
                print(f"⚠️ No cached data found, downloading fresh data...")
                data = download_symbol_data(
                    self.symbol, 
                    period=GLOBAL_SETTINGS["data_period"], 
                    save_to_file=True
                )
            
            return data
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return []
    
    def get_current_signal(self, current_price: Optional[float] = None) -> Tuple[str, Dict]:
        """
        Get current trading signal based on latest market data.
        
        Args:
            current_price: Optional current price. If not provided, uses latest historical price.
            
        Returns:
            Tuple of (signal, details) where:
            - signal: "BUY", "SELL", or "HOLD"
            - details: Dictionary with RSI value, thresholds, and analysis
        """
        if not self.historical_data:
            return "HOLD", {"error": "No historical data available"}
        
        # Create current market data point
        latest_data = self.historical_data[-1].copy()
        
        # Update with current price if provided
        if current_price is not None:
            latest_data['close'] = current_price
            latest_data['high'] = max(latest_data['high'], current_price)
            latest_data['low'] = min(latest_data['low'], current_price)
            latest_data['date'] = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Use all historical data for strategy evaluation
        price_history = self.historical_data.copy()
        
        # Add current data point to history for RSI calculation
        if current_price is not None:
            price_history.append(latest_data)
        
        # Check if we have sufficient data for RSI calculation
        min_required_days = self.strategy.rsi_period + 2  # Need extra day for crossover detection
        if len(price_history) < min_required_days:
            return "HOLD", {
                "error": f"Insufficient data for RSI calculation (need {min_required_days}, have {len(price_history)})"
            }
        
        # Calculate current RSI for analysis
        current_rsi = self.strategy._calculate_rsi(price_history)
        
        # Evaluate buy and sell signals
        should_buy = self.strategy.should_buy(price_history, latest_data)
        should_sell = self.strategy.should_sell(price_history, latest_data)
        
        # Determine signal
        if should_buy:
            signal = "BUY"
            reason = f"RSI crossed below oversold threshold ({self.strategy.oversold_threshold})"
        elif should_sell:
            signal = "SELL" 
            reason = f"RSI crossed above overbought threshold ({self.strategy.overbought_threshold})"
        else:
            signal = "HOLD"
            if current_rsi is None:
                reason = "RSI calculation unavailable"
            elif current_rsi < self.strategy.oversold_threshold:
                reason = f"RSI ({current_rsi:.1f}) is oversold but no crossover detected"
            elif current_rsi > self.strategy.overbought_threshold:
                reason = f"RSI ({current_rsi:.1f}) is overbought but no crossover detected"
            else:
                reason = f"RSI ({current_rsi:.1f}) is in neutral zone"
        
        # Prepare detailed analysis
        details = {
            "rsi_value": current_rsi,
            "oversold_threshold": self.strategy.oversold_threshold,
            "overbought_threshold": self.strategy.overbought_threshold,
            "rsi_period": self.strategy.rsi_period,
            "current_price": latest_data['close'],
            "signal_reason": reason,
            "data_points_used": len(price_history),
            "latest_date": latest_data['date'].strftime('%Y-%m-%d %H:%M:%S'),
            "strategy_params": self.strategy_params
        }
        
        return signal, details
    
    def get_signal_analysis(self, days_back: int = 30) -> Dict:
        """
        Get historical signal analysis for the last N days.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with signal statistics and recent signals
        """
        if len(self.historical_data) < days_back + self.strategy.rsi_period:
            days_back = len(self.historical_data) - self.strategy.rsi_period - 1
        
        if days_back <= 0:
            return {"error": "Insufficient historical data for analysis"}
        
        recent_signals = []
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        # Analyze signals for each day in the period
        for i in range(len(self.historical_data) - days_back, len(self.historical_data)):
            if i < self.strategy.rsi_period + 1:
                continue
                
            # Get price history up to this day
            price_history = self.historical_data[:i+1]
            current_data = self.historical_data[i]
            
            # Evaluate signals
            should_buy = self.strategy.should_buy(price_history, current_data)
            should_sell = self.strategy.should_sell(price_history, current_data)
            rsi_value = self.strategy._calculate_rsi(price_history)
            
            if should_buy:
                signal = "BUY"
            elif should_sell:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            signal_counts[signal] += 1
            
            # Store detailed signal info
            recent_signals.append({
                "date": current_data['date'].strftime('%Y-%m-%d'),
                "signal": signal,
                "rsi": rsi_value,
                "price": current_data['close']
            })
        
        return {
            "analysis_period_days": days_back,
            "signal_counts": signal_counts,
            "signal_distribution": {
                signal: round(count / len(recent_signals) * 100, 1) 
                for signal, count in signal_counts.items()
            },
            "recent_signals": recent_signals[-10:],  # Last 10 signals
            "total_signals_analyzed": len(recent_signals)
        }


def demo_real_time_signals():
    """Demonstrate real-time signal generation"""
    print("=" * 60)
    
    # Create signal generator for SPY with default RSI parameters
    try:
        signal_gen = RealTimeSignalGenerator("SPY")
        
        # Get current signal
        print("\n📊 CURRENT MARKET SIGNAL:")
        print("-" * 30)
        
        signal, details = signal_gen.get_current_signal()
        
        print(f"🎯 Signal: {signal}")
        print(f"💹 Current Price: ${details.get('current_price', 'N/A'):.2f}")
        print(f"📈 RSI Value: {details.get('rsi_value', 'N/A'):.1f}" if details.get('rsi_value') else "📈 RSI Value: N/A")
        print(f"📅 Latest Data: {details.get('latest_date', 'N/A')}")
        # print(f"💡 Reason: {details.get('signal_reason', 'N/A')}")
        
        print(f"\n🔧 Strategy Parameters:")
        print(f"   RSI Period: {details.get('rsi_period', 'N/A')}")
        print(f"   Oversold Threshold: {details.get('oversold_threshold', 'N/A')}")
        print(f"   Overbought Threshold: {details.get('overbought_threshold', 'N/A')}")
        
        # Test with custom current price
        print(f"\n🧪 TESTING WITH CUSTOM PRICE:")
        print("-" * 30)
        
        latest_price = details.get('current_price', 400)
        test_price = latest_price * GLOBAL_SETTINGS["test_price_adjustment"]
        
        test_signal, test_details = signal_gen.get_current_signal(current_price=test_price)
        print(f"🎯 Signal with ${test_price:.2f}: {test_signal}")
        print(f"📈 RSI Value: {test_details.get('rsi_value', 'N/A'):.1f}" if test_details.get('rsi_value') else "📈 RSI Value: N/A")
        print(f"💡 Reason: {test_details.get('signal_reason', 'N/A')}")
        
        # Get historical analysis
        print(f"\n📈 RECENT SIGNAL ANALYSIS (Last {GLOBAL_SETTINGS['analysis_days']} Days):")
        print("-" * 30)
        
        analysis = signal_gen.get_signal_analysis(days_back=GLOBAL_SETTINGS["analysis_days"])
        
        if "error" not in analysis:
            signal_counts = analysis.get('signal_counts', {})
            signal_dist = analysis.get('signal_distribution', {})
            
            print(f"📊 Signal Distribution:")
            for signal_type in ['BUY', 'SELL', 'HOLD']:
                count = signal_counts.get(signal_type, 0)
                percent = signal_dist.get(signal_type, 0)
                print(f"   {signal_type}: {count} signals ({percent}%)")
            
            print(f"\n🕐 Recent Signals:")
            recent = analysis.get('recent_signals', [])
            for sig in recent[-5:]:  # Show last 5
                date = sig.get('date', 'N/A')
                signal_type = sig.get('signal', 'N/A')
                rsi = sig.get('rsi', 0)
                price = sig.get('price', 0)
                print(f"   {date}: {signal_type} (RSI: {rsi:.1f}, Price: ${price:.2f})")
        else:
            print(f"❌ Analysis error: {analysis['error']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_multiple_symbols():
    """Demonstrate signals for multiple symbols with their specific configurations"""
    print(f"\n🌟 MULTI-SYMBOL SIGNAL ANALYSIS:")
    print("=" * 60)
    
    # Use configured symbols
    symbols = list(SYMBOL_CONFIGURATIONS.keys())
    
    for symbol in symbols:
        try:
            print(f"\n📊 {symbol} Analysis:")
            print("-" * 20)
            
            signal_gen = RealTimeSignalGenerator(symbol)
            signal, details = signal_gen.get_current_signal()
            
            # Show strategy configuration
            config = SYMBOL_CONFIGURATIONS[symbol]
            params = config["strategy_params"]
            print(f"📋 Strategy: {config['description']}")
            print(f"⚙️  RSI({params['rsi_period']}) - OS:{params['oversold_threshold']}/OB:{params['overbought_threshold']}")
            
            print(f"🎯 Signal: {signal}")
            print(f"💹 Price: ${details.get('current_price', 'N/A'):.2f}")
            print(f"📈 RSI: {details.get('rsi_value', 'N/A'):.1f}" if details.get('rsi_value') else "📈 RSI: N/A")
            print(f"💡 {details.get('signal_reason', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")


def demo_hypothetical_scenarios():
    """Test hypothetical price scenarios from -20% to +20% in 1% steps"""
    print(f"\n🔮 HYPOTHETICAL PRICE SCENARIOS:")
    print("=" * 60)
    
    # Test scenarios for SPY
    symbol = "SPY"
    
    try:
        signal_gen = RealTimeSignalGenerator(symbol)
        
        # Get current price as baseline
        current_signal, current_details = signal_gen.get_current_signal()
        current_price = current_details.get('current_price', 0)
        current_rsi = current_details.get('rsi_value', 0)
        
        print(f"\n📊 {symbol} Hypothetical Analysis:")
        print(f"💹 Current Price: ${current_price:.2f}")
        print(f"📈 Current RSI: {current_rsi:.1f}")
        print(f"🎯 Current Signal: {current_signal}")
        
        print(f"\n🧪 Testing price scenarios (-20% to +20% in 0.1% steps):")
        print("-" * 80)
        
        # Store scenario results for summary
        scenario_results = []
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        # Test scenarios from -20% to +20% in 0.1% steps
        print("⏳ Calculating 401 scenarios...", end="", flush=True)
        
        for i, change_pct_tenths in enumerate(range(-200, 201, 1)):  # -20.0 to +20.0 in 0.1% steps
            change_pct = change_pct_tenths / 10.0  # Convert to percentage
            scenario_price = current_price * (1 + change_pct / 100.0)
            
            # Get signal for this hypothetical price
            scenario_signal, scenario_details = signal_gen.get_current_signal(current_price=scenario_price)
            scenario_rsi = scenario_details.get('rsi_value', 0)
            scenario_reason = scenario_details.get('signal_reason', 'Unknown')
            
            # Store results
            scenario_results.append({
                'change_pct': change_pct,
                'price': scenario_price,
                'rsi': scenario_rsi,
                'signal': scenario_signal,
                'reason': scenario_reason
            })
            
            signal_counts[scenario_signal] += 1
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f" {i+1}...", end="", flush=True)
        
        print(" Done! ✅")
        print("-" * 80)
        print(f"\n📊 SCENARIO SUMMARY (401 scenarios in 0.1% increments):")
        total_scenarios = len(scenario_results)
        
        for signal_type in ['BUY', 'SELL', 'HOLD']:
            count = signal_counts[signal_type]
            percent = (count / total_scenarios) * 100
            print(f"   {signal_type}: {count:3d} scenarios ({percent:5.1f}%)")
        
        # Find interesting thresholds with more precision
        print(f"\n🎯 KEY THRESHOLDS (0.1% precision):")
        
        # Find BUY thresholds
        buy_scenarios = [s for s in scenario_results if s['signal'] == 'BUY']
        if buy_scenarios:
            min_buy = min(buy_scenarios, key=lambda x: x['change_pct'])
            max_buy = max(buy_scenarios, key=lambda x: x['change_pct'])
            print(f"   📈 BUY signals: {min_buy['change_pct']:+5.1f}% to {max_buy['change_pct']:+5.1f}% (RSI: {min_buy['rsi']:.1f} to {max_buy['rsi']:.1f})")
        else:
            print(f"   📈 BUY signals: None in this range")
        
        # Find SELL thresholds
        sell_scenarios = [s for s in scenario_results if s['signal'] == 'SELL']
        if sell_scenarios:
            min_sell = min(sell_scenarios, key=lambda x: x['change_pct'])
            max_sell = max(sell_scenarios, key=lambda x: x['change_pct'])
            print(f"   📉 SELL signals: {min_sell['change_pct']:+5.1f}% to {max_sell['change_pct']:+5.1f}% (RSI: {min_sell['rsi']:.1f} to {max_sell['rsi']:.1f})")
        else:
            print(f"   📉 SELL signals: None in this range")
        
        # Find signal transition points (only show major transitions)
        print(f"\n🔄 PRECISE SIGNAL TRANSITIONS:")
        transitions = []
        for i in range(1, len(scenario_results)):
            prev_signal = scenario_results[i-1]['signal']
            curr_signal = scenario_results[i]['signal']
            
            if prev_signal != curr_signal:
                curr_change = scenario_results[i]['change_pct']
                curr_price = scenario_results[i]['price']
                curr_rsi = scenario_results[i]['rsi']
                transitions.append({
                    'from': prev_signal,
                    'to': curr_signal,
                    'change': curr_change,
                    'price': curr_price,
                    'rsi': curr_rsi
                })
        
        for transition in transitions:
            print(f"   {transition['from']} → {transition['to']} at {transition['change']:+5.1f}% (${transition['price']:.2f}, RSI: {transition['rsi']:.1f})")
        
        # Show some sample scenarios around key points
        print(f"\n📋 SAMPLE SCENARIOS AROUND TRANSITIONS:")
        for transition in transitions[:3]:  # Show first 3 transitions
            change = transition['change']
            print(f"   Around {transition['from']} → {transition['to']} transition ({change:+5.1f}%):")
            
            # Show scenarios around this transition
            for scenario in scenario_results:
                if abs(scenario['change_pct'] - change) <= 0.3:  # Within 0.3% of transition
                    print(f"     {scenario['change_pct']:+5.1f}%: ${scenario['price']:.2f}, RSI {scenario['rsi']:.1f} → {scenario['signal']}")
            print()
                
    except Exception as e:
        print(f"❌ Hypothetical scenarios failed: {e}")
        import traceback
        traceback.print_exc()


def demo_custom_configurations():
    """Demonstrate custom strategy configurations"""
    print(f"\n🛠️  CUSTOM CONFIGURATION DEMO:")
    print("=" * 60)
    
    # Test SPY with very aggressive RSI settings
    custom_config = {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 7,      # Very short period
            'oversold_threshold': 15.0,  # Very aggressive
            'overbought_threshold': 85.0, # Very aggressive
            'cash_percentage': 0.25
        },
        "description": "SPY with ultra-aggressive RSI (7-period, 15/85 thresholds)"
    }
    
    try:
        print(f"\n📊 SPY with Custom Ultra-Aggressive Settings:")
        print("-" * 45)
        
        signal_gen = RealTimeSignalGenerator("SPY", custom_config=custom_config)
        signal, details = signal_gen.get_current_signal()
        
        print(f"🎯 Signal: {signal}")
        print(f"💹 Price: ${details.get('current_price', 'N/A'):.2f}")
        print(f"📈 RSI: {details.get('rsi_value', 'N/A'):.1f}" if details.get('rsi_value') else "📈 RSI: N/A")
        print(f"💡 {details.get('signal_reason', 'N/A')}")
        
        # Compare with standard SPY configuration
        print(f"\n📊 SPY with Standard Settings (for comparison):")
        print("-" * 45)
        
        standard_gen = RealTimeSignalGenerator("SPY")
        std_signal, std_details = standard_gen.get_current_signal()
        
        print(f"🎯 Signal: {std_signal}")
        print(f"📈 RSI: {std_details.get('rsi_value', 'N/A'):.1f}" if std_details.get('rsi_value') else "📈 RSI: N/A")
        print(f"💡 {std_details.get('signal_reason', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Custom configuration demo failed: {e}")


def production_signals_summary(symbol=None):
    """Production-ready simplified signals summary for specific symbol or all symbols"""
    if symbol is None:
        # Run for all configured symbols
        symbols = list(SYMBOL_CONFIGURATIONS.keys())
        
        print("🚀 REAL-TIME TRADING SIGNALS - PRODUCTION SUMMARY")
        print("=" * 80)
        
        for sym in symbols:
            print(f"\n📊 {sym}: {SYMBOL_CONFIGURATIONS[sym]['description']}")
            print("-" * 40)
            production_signals_summary(sym)
        return
    
    # Run for specific symbol
    try:
        # Create signal generator (no verbose output)
        signal_gen = RealTimeSignalGenerator(symbol)
        
        # Get current signal
        signal, details = signal_gen.get_current_signal()
        
        # CURRENT MARKET SIGNAL
        print(f"📊 CURRENT MARKET SIGNAL:")
        print(f"   🎯 Signal: {signal}")
        print(f"   💹 Current Price: ${details.get('current_price', 'N/A'):.2f}")
        print(f"   📈 RSI Value: {details.get('rsi_value', 'N/A'):.1f}" if details.get('rsi_value') else "   📈 RSI Value: N/A")
        print(f"   📅 Latest Data: {details.get('latest_date', 'N/A')}")
        
        # HYPOTHETICAL SCENARIOS (simplified)
        current_price = details.get('current_price', 0)
        scenario_results = []
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        # Calculate scenarios quietly
        for change_pct_tenths in range(-200, 201, 1):  # -20.0 to +20.0 in 0.1% steps
            change_pct = change_pct_tenths / 10.0
            scenario_price = current_price * (1 + change_pct / 100.0)
            scenario_signal, scenario_details = signal_gen.get_current_signal(current_price=scenario_price)
            scenario_rsi = scenario_details.get('rsi_value', 0)
            
            scenario_results.append({
                'change_pct': change_pct,
                'price': scenario_price,
                'rsi': scenario_rsi,
                'signal': scenario_signal
            })
            signal_counts[scenario_signal] += 1
        
        # KEY THRESHOLDS
        print(f"\n🎯 KEY THRESHOLDS (0.1% precision):")
        
        # Find BUY thresholds
        buy_scenarios = [s for s in scenario_results if s['signal'] == 'BUY']
        if buy_scenarios:
            min_buy = min(buy_scenarios, key=lambda x: x['change_pct'])
            print(f"   📈 BUY: {min_buy['change_pct']:+5.1f}% (${min_buy['price']:.2f})")
        else:
            print(f"   📈 BUY: None in range")
        
        # Find SELL thresholds
        sell_scenarios = [s for s in scenario_results if s['signal'] == 'SELL']
        if sell_scenarios:
            min_sell = min(sell_scenarios, key=lambda x: x['change_pct'])
            print(f"   📉 SELL: {min_sell['change_pct']:+5.1f}% (${min_sell['price']:.2f})")
        else:
            print(f"   📉 SELL: None in range")
        
        # Find precise transition points
        transitions = []
        for i in range(1, len(scenario_results)):
            prev_signal = scenario_results[i-1]['signal']
            curr_signal = scenario_results[i]['signal']
            
            if prev_signal != curr_signal:
                curr_change = scenario_results[i]['change_pct']
                transitions.append({
                    'from': prev_signal,
                    'to': curr_signal,
                    'change': curr_change
                })
        
        if transitions:
            print(f"   🔄 Transitions:")
            for t in transitions:
                print(f"      {t['from']}→{t['to']} at {t['change']:+5.1f}%")
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")


if __name__ == "__main__":
    """Run the production signals summary"""
    
    try:
        # Production summary (clean and fast)
        production_signals_summary()
        
        print(f"\n✅ Production signals summary completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
