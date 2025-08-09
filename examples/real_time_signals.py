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
from strategies.adaptive_ma_crossover import AdaptiveMAStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy

# Email functionality
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

# ============================================================================
# CONFIGURATION - Symbol-Specific Strategies and Parameters
# ============================================================================

# Define strategies and parameters for each symbol
SYMBOL_CONFIGURATIONS = {
    "IWMY": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 22,
            'oversold_threshold': 38.0,
            'overbought_threshold': 68.0,
            'cash_percentage': 0.15
        },
        "description": "RSI Defiance R2000 Target 30 Income ETF"
    },
    "IWMY": {
        "strategy_class": BollingerBandsStrategy,
        "strategy_params": {
            'period': 30,
            'std_positive': 1.5,
            'std_negative': 2.0,
            'cash_percentage': 0.15
        },
        "description": "Bollinger Bands - Defiance R2000 Target 30 Income ETF"
    },    
    "MSFT": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 22,
            'oversold_threshold': 35.0,
            'overbought_threshold': 60.0,
            'cash_percentage': 0.15
        },
        "description": "Microsoft Corporation"
    },    
    "MSTY": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 14,
            'oversold_threshold': 27.0,
            'overbought_threshold': 69.0,
            'cash_percentage': 0.15
        },
        "description": "Yieldmax MSTR Option Income Strategy ETF"
    },        
    "NVDY": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 12,
            'oversold_threshold': 25.0,
            'overbought_threshold': 60.0,
            'cash_percentage': 0.15
        },
        "description": "YieldMax NVDA Option Income Strategy ETF"
    },
    "SDIV": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 13,
            'oversold_threshold': 23.0,
            'overbought_threshold': 66.0,
            'cash_percentage': 0.15
        },
        "description": "Global X SuperDividend ETF"
    },
    "SPXL": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 12,
            'oversold_threshold': 21.0,
            'overbought_threshold': 65.0,
            'cash_percentage': 0.15
        },
        "description": "Direxion Daily S&P 500 Bull 3X Shares"
    },
    "ULTY": {
        "strategy_class": AdaptiveMAStrategy,
        "strategy_params": {
            'buy_slow_period': 13,
            'buy_fast_period': 5,
            'sell_slow_period': 13,
            'sell_fast_period': 5,
            'cash_percentage': 0.15
        },
        "description": "YieldMax Ultra Option Income Strategy ETF"
    },
    "YMAX": {
        "strategy_class": RSIStrategy,
        "strategy_params": {
            'rsi_period': 16,
            'oversold_threshold': 35.0,
            'overbought_threshold': 65.0,
            'cash_percentage': 0.15
        },
        "description": "YieldMax Universe Fund of Option Income ETF"
    },

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
    "data_period": "1y",  # How much historical data to load
    "analysis_days": 30,  # Days to analyze for signal distribution
    "force_download": False,  # Whether to force fresh data download
    "test_price_adjustment": 0.95  # Multiplier for custom price testing (5% lower)
}

# Email configuration for Brevo
EMAIL_CONFIG = {
    "api_key": os.getenv("BREVO_API_KEY"),  # Set this environment variable
    "sender_email": "stevendeeds@yahoo.com",
    "sender_name": "Trading Signals Bot",
    "recipient_email": "stevendeeds@yahoo.com",
    "enabled": True
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
            - details: Dictionary with strategy-specific analysis
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
        
        # Add current data point to history for calculations
        if current_price is not None:
            price_history.append(latest_data)
        
        # Determine minimum required days based on strategy type
        if hasattr(self.strategy, 'rsi_period'):
            # RSI Strategy
            min_required_days = self.strategy.rsi_period + 2
            strategy_type = "RSI"
        elif hasattr(self.strategy, 'buy_slow_period'):
            # Adaptive MA Strategy with separate buy/sell periods
            min_required_days = max(self.strategy.buy_slow_period, self.strategy.sell_slow_period) + 2
            strategy_type = "Adaptive_MA"
        elif hasattr(self.strategy, 'slow_period'):
            # Standard MA Strategy
            min_required_days = self.strategy.slow_period + 2
            strategy_type = "MA"
        elif hasattr(self.strategy, 'period'):
            # Bollinger Bands Strategy
            min_required_days = self.strategy.period + 2
            strategy_type = "Bollinger_Bands"
        else:
            # Unknown strategy type
            min_required_days = 50  # Conservative default
            strategy_type = "Unknown"
        
        if len(price_history) < min_required_days:
            return "HOLD", {
                "error": f"Insufficient data for {strategy_type} calculation (need {min_required_days}, have {len(price_history)})"
            }
        
        # Evaluate buy and sell signals
        should_buy = self.strategy.should_buy(price_history, latest_data)
        should_sell = self.strategy.should_sell(price_history, latest_data)
        
        # Determine signal and reason based on strategy type
        if should_buy:
            signal = "BUY"
            reason = self._get_buy_reason(strategy_type)
        elif should_sell:
            signal = "SELL"
            reason = self._get_sell_reason(strategy_type)
        else:
            signal = "HOLD"
            reason = self._get_hold_reason(strategy_type, price_history)
        
        # Prepare detailed analysis based on strategy type
        details = self._get_strategy_details(strategy_type, price_history, latest_data, reason)
        
        return signal, details
    
    def _get_buy_reason(self, strategy_type: str) -> str:
        """Get buy reason based on strategy type"""
        if strategy_type == "RSI":
            return f"RSI crossed below oversold threshold ({self.strategy.oversold_threshold})"
        elif strategy_type in ["MA", "Adaptive_MA"]:
            return "Fast MA crossed above slow MA (Golden Cross)"
        elif strategy_type == "Bollinger_Bands":
            return "Price touched or went below lower Bollinger Band"
        else:
            return "Buy signal detected"
    
    def _get_sell_reason(self, strategy_type: str) -> str:
        """Get sell reason based on strategy type"""
        if strategy_type == "RSI":
            return f"RSI crossed above overbought threshold ({self.strategy.overbought_threshold})"
        elif strategy_type in ["MA", "Adaptive_MA"]:
            return "Fast MA crossed below slow MA (Death Cross)"
        elif strategy_type == "Bollinger_Bands":
            return "Price touched or went above upper Bollinger Band"
        else:
            return "Sell signal detected"
    
    def _get_hold_reason(self, strategy_type: str, price_history: List[Dict]) -> str:
        """Get hold reason based on strategy type"""
        if strategy_type == "RSI":
            try:
                current_rsi = self.strategy._calculate_rsi(price_history)
                if current_rsi is None:
                    return "RSI calculation unavailable"
                elif current_rsi < self.strategy.oversold_threshold:
                    return f"RSI ({current_rsi:.1f}) is oversold but no crossover detected"
                elif current_rsi > self.strategy.overbought_threshold:
                    return f"RSI ({current_rsi:.1f}) is overbought but no crossover detected"
                else:
                    return f"RSI ({current_rsi:.1f}) is in neutral zone"
            except:
                return "RSI analysis unavailable"
        elif strategy_type in ["MA", "Adaptive_MA"]:
            return "No MA crossover detected"
        elif strategy_type == "Bollinger_Bands":
            return "Price within Bollinger Bands range"
        else:
            return "No signal detected"
    
    def _get_strategy_details(self, strategy_type: str, price_history: List[Dict], latest_data: Dict, reason: str) -> Dict:
        """Get strategy-specific details"""
        base_details = {
            "current_price": latest_data['close'],
            "signal_reason": reason,
            "data_points_used": len(price_history),
            "latest_date": latest_data['date'].strftime('%Y-%m-%d %H:%M:%S'),
            "strategy_params": self.strategy_params,
            "strategy_type": strategy_type
        }
        
        if strategy_type == "RSI":
            try:
                current_rsi = self.strategy._calculate_rsi(price_history)
                base_details.update({
                    "rsi_value": current_rsi,
                    "oversold_threshold": self.strategy.oversold_threshold,
                    "overbought_threshold": self.strategy.overbought_threshold,
                    "rsi_period": self.strategy.rsi_period
                })
            except:
                base_details["rsi_value"] = None
        elif strategy_type == "Adaptive_MA":
            try:
                # Calculate current MAs for display
                buy_fast_ma = self.strategy._calculate_moving_average(price_history, self.strategy.buy_fast_period)
                buy_slow_ma = self.strategy._calculate_moving_average(price_history, self.strategy.buy_slow_period)
                sell_fast_ma = self.strategy._calculate_moving_average(price_history, self.strategy.sell_fast_period)
                sell_slow_ma = self.strategy._calculate_moving_average(price_history, self.strategy.sell_slow_period)
                
                base_details.update({
                    "buy_fast_ma": buy_fast_ma,
                    "buy_slow_ma": buy_slow_ma,
                    "sell_fast_ma": sell_fast_ma,
                    "sell_slow_ma": sell_slow_ma,
                    "buy_fast_period": self.strategy.buy_fast_period,
                    "buy_slow_period": self.strategy.buy_slow_period,
                    "sell_fast_period": self.strategy.sell_fast_period,
                    "sell_slow_period": self.strategy.sell_slow_period
                })
            except:
                pass
        elif strategy_type == "MA":
            try:
                # Calculate current MAs for display
                fast_ma = self.strategy._calculate_moving_average(price_history, self.strategy.fast_period)
                slow_ma = self.strategy._calculate_moving_average(price_history, self.strategy.slow_period)
                
                base_details.update({
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "fast_period": self.strategy.fast_period,
                    "slow_period": self.strategy.slow_period
                })
            except:
                pass
        elif strategy_type == "Bollinger_Bands":
            try:
                # Calculate current Bollinger Bands for display
                bands = self.strategy._calculate_bollinger_bands(price_history)
                if bands:
                    middle, upper, lower = bands
                    base_details.update({
                        "bb_middle": middle,
                        "bb_upper": upper,
                        "bb_lower": lower,
                        "bb_period": self.strategy.period,
                        "std_positive": self.strategy.std_positive,
                        "std_negative": self.strategy.std_negative
                    })
            except:
                pass
        
        return base_details
    
    def get_signal_analysis(self, days_back: int = 30) -> Dict:
        """
        Get historical signal analysis for the last N days.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with signal statistics and recent signals
        """
        # Determine minimum required days based on strategy type
        if hasattr(self.strategy, 'rsi_period'):
            min_required_days = self.strategy.rsi_period
        elif hasattr(self.strategy, 'buy_slow_period'):
            min_required_days = max(self.strategy.buy_slow_period, self.strategy.sell_slow_period)
        elif hasattr(self.strategy, 'slow_period'):
            min_required_days = self.strategy.slow_period
        elif hasattr(self.strategy, 'period'):
            min_required_days = self.strategy.period
        else:
            min_required_days = 20  # Conservative default
        
        if len(self.historical_data) < days_back + min_required_days:
            days_back = len(self.historical_data) - min_required_days - 1
        
        if days_back <= 0:
            return {"error": "Insufficient historical data for analysis"}
        
        recent_signals = []
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        # Analyze signals for each day in the period
        for i in range(len(self.historical_data) - days_back, len(self.historical_data)):
            if i < min_required_days + 1:
                continue
                
            # Get price history up to this day
            price_history = self.historical_data[:i+1]
            current_data = self.historical_data[i]
            
            # Evaluate signals
            should_buy = self.strategy.should_buy(price_history, current_data)
            should_sell = self.strategy.should_sell(price_history, current_data)
            
            # Get indicator value if available
            indicator_value = None
            if hasattr(self.strategy, '_calculate_rsi'):
                try:
                    indicator_value = self.strategy._calculate_rsi(price_history)
                except:
                    indicator_value = None
            elif hasattr(self.strategy, '_calculate_moving_average'):
                try:
                    # For MA strategies, show the fast MA value
                    if hasattr(self.strategy, 'buy_fast_period'):
                        indicator_value = self.strategy._calculate_moving_average(price_history, self.strategy.buy_fast_period)
                    elif hasattr(self.strategy, 'fast_period'):
                        indicator_value = self.strategy._calculate_moving_average(price_history, self.strategy.fast_period)
                except:
                    indicator_value = None
            
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
                "indicator": indicator_value,
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


class EmailNotifier:
    """
    Handles email notifications for trading signals using Brevo API.
    """
    
    def __init__(self):
        """Initialize email configuration"""
        self.api_key = EMAIL_CONFIG["api_key"]
        self.sender_email = EMAIL_CONFIG["sender_email"]
        self.sender_name = EMAIL_CONFIG["sender_name"]
        self.recipient_email = EMAIL_CONFIG["recipient_email"]
        self.enabled = EMAIL_CONFIG["enabled"]
        
        if self.enabled and not self.api_key:
            print("⚠️ Email is enabled but BREVO_API_KEY environment variable is not set")
            self.enabled = False
        
        if self.enabled:
            # Configure Brevo API
            configuration = sib_api_v3_sdk.Configuration()
            configuration.api_key['api-key'] = self.api_key
            self.api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
    
    def format_trading_signals_email(self, signals_data: List[Dict]) -> Tuple[str, str]:
        """
        Format trading signals data into email subject and HTML body.
        
        Args:
            signals_data: List of signal dictionaries from multiple symbols
            
        Returns:
            Tuple of (subject, html_body)
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Count signals by type
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        buy_symbols = []
        sell_symbols = []
        
        for signal_data in signals_data:
            signal = signal_data.get("signal", "HOLD")
            symbol = signal_data.get("symbol", "Unknown")
            signal_counts[signal] += 1
            
            if signal == "BUY":
                buy_symbols.append(symbol)
            elif signal == "SELL":
                sell_symbols.append(symbol)
        
        # Create subject line
        if buy_symbols:
            subject = f"🔥 BUY Signals: {', '.join(buy_symbols)} - {current_time}"
        elif sell_symbols:
            subject = f"📉 SELL Signals: {', '.join(sell_symbols)} - {current_time}"
        else:
            subject = f"📊 Trading Signals Update - All HOLD - {current_time}"
        
        # Create HTML body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                .signal-card {{ margin: 15px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #ddd; }}
                .signal-buy {{ border-left-color: #28a745; background-color: #f8fff9; }}
                .signal-sell {{ border-left-color: #dc3545; background-color: #fff8f8; }}
                .signal-hold {{ border-left-color: #6c757d; background-color: #f8f9fa; }}
                .signal-title {{ font-size: 18px; font-weight: bold; margin-bottom: 8px; }}
                .signal-buy .signal-title {{ color: #28a745; }}
                .signal-sell .signal-title {{ color: #dc3545; }}
                .signal-hold .signal-title {{ color: #6c757d; }}
                .details {{ color: #666; margin: 5px 0; }}
                .thresholds {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; margin-top: 10px; }}
                .summary {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .footer {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Real-Time Trading Signals Report</h1>
                    <p>Generated: {current_time}</p>
                </div>
                
                <div class="summary">
                    <h2>📈 Signal Summary</h2>
                    <p><strong>BUY Signals:</strong> {signal_counts['BUY']} ({', '.join(buy_symbols) if buy_symbols else 'None'})</p>
                    <p><strong>SELL Signals:</strong> {signal_counts['SELL']} ({', '.join(sell_symbols) if sell_symbols else 'None'})</p>
                    <p><strong>HOLD Signals:</strong> {signal_counts['HOLD']}</p>
                    <p><strong>Total Symbols Analyzed:</strong> {len(signals_data)}</p>
                </div>
        """
        
        # Add individual signal cards
        for signal_data in signals_data:
            symbol = signal_data.get("symbol", "Unknown")
            signal = signal_data.get("signal", "HOLD")
            details = signal_data.get("details", {})
            description = signal_data.get("description", "")
            
            current_price = details.get("current_price", 0)
            rsi_value = details.get("rsi_value", 0)
            signal_reason = details.get("signal_reason", "No reason provided")
            oversold_threshold = details.get("oversold_threshold", 0)
            overbought_threshold = details.get("overbought_threshold", 0)
            rsi_period = details.get("rsi_period", 0)
            latest_date = details.get("latest_date", "Unknown")
            
            # Get buy/sell thresholds
            buy_threshold = signal_data.get("buy_threshold", "N/A")
            sell_threshold = signal_data.get("sell_threshold", "N/A")
            
            signal_class = f"signal-{signal.lower()}"
            signal_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}[signal]
            
            html_body += f"""
                <div class="signal-card {signal_class}">
                    <div class="signal-title">{signal_emoji} {symbol} - {signal}</div>
                    <div class="details"><strong>Description:</strong> {description}</div>
                    <div class="details"><strong>Current Price:</strong> ${current_price:.2f}</div>
                    <div class="details"><strong>RSI({rsi_period}):</strong> {rsi_value:.1f}</div>
                    <div class="details"><strong>Latest Data:</strong> {latest_date}</div>
                    <div class="details"><strong>Reason:</strong> {signal_reason}</div>
                    
                    <div class="thresholds">
                        <strong>Strategy Parameters:</strong><br>
                        • Oversold Threshold: {oversold_threshold}<br>
                        • Overbought Threshold: {overbought_threshold}<br>
                        • RSI Period: {rsi_period}<br>
                        <br>
                        <strong>Price Thresholds:</strong><br>
                        • BUY Trigger: {buy_threshold}<br>
                        • SELL Trigger: {sell_threshold}
                    </div>
                </div>
            """
        
        html_body += f"""
                <div class="footer">
                    <p>This is an automated trading signals report generated by the Real-Time Signals system.</p>
                    <p>⚠️ <strong>Disclaimer:</strong> This information is for educational purposes only and should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return subject, html_body
    
    def send_trading_signals_email(self, signals_data: List[Dict]) -> bool:
        """
        Send trading signals email via Brevo.
        
        Args:
            signals_data: List of signal dictionaries from multiple symbols
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            print("📧 Email notifications are disabled")
            return False
        
        try:
            # Format email content
            subject, html_body = self.format_trading_signals_email(signals_data)
            
            # Create email object
            send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
                to=[{"email": self.recipient_email}],
                sender={"email": self.sender_email, "name": self.sender_name},
                subject=subject,
                html_content=html_body
            )
            
            # Send email
            api_response = self.api_instance.send_transac_email(send_smtp_email)
            print(f"✅ Email sent successfully! Message ID: {api_response.message_id}")
            return True
            
        except ApiException as e:
            print(f"❌ Brevo API error sending email: {e}")
            return False
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return False


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


def production_signals_summary(symbol=None, collect_for_email=False):
    """Production-ready simplified signals summary for specific symbol or all symbols"""
    email_data = []  # Collect data for email if requested
    
    if symbol is None:
        # Run for all configured symbols
        symbols = list(SYMBOL_CONFIGURATIONS.keys())
        
        if not collect_for_email:
            print("🚀 REAL-TIME TRADING SIGNALS - PRODUCTION SUMMARY")
            print("=" * 80)
        
        for sym in symbols:
            if not collect_for_email:
                print(f"\n📊 {sym}: {SYMBOL_CONFIGURATIONS[sym]['description']}")
                print("-" * 40)
            
            symbol_data = production_signals_summary(sym, collect_for_email=True)
            if collect_for_email and symbol_data:
                email_data.append(symbol_data)
            elif not collect_for_email:
                production_signals_summary(sym)
        
        return email_data if collect_for_email else None
    
    # Run for specific symbol
    try:
        # Create signal generator (no verbose output)
        signal_gen = RealTimeSignalGenerator(symbol)
        
        # Get current signal
        signal, details = signal_gen.get_current_signal()
        
        if not collect_for_email:
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
        
        # Find precise transition points first
        transitions = []
        for i in range(1, len(scenario_results)):
            prev_signal = scenario_results[i-1]['signal']
            curr_signal = scenario_results[i]['signal']
            
            if prev_signal != curr_signal:
                curr_change = scenario_results[i]['change_pct']
                curr_price = scenario_results[i]['price']
                transitions.append({
                    'from': prev_signal,
                    'to': curr_signal,
                    'change': curr_change,
                    'price': curr_price
                })
        
        # Find BUY and SELL thresholds
        buy_threshold = "None in range"
        sell_threshold = "None in range"
        
        # Find BUY threshold - any transition involving BUY (either starting or ending)
        buy_transitions = [t for t in transitions if t['to'] == 'BUY' or t['from'] == 'BUY']
        if buy_transitions:
            # Get the full range of BUY activity
            buy_changes = [t['change'] for t in buy_transitions]
            min_buy = min(buy_changes)  # When BUY signals start
            max_buy = max(buy_changes)  # When BUY signals end
            
            if min_buy == max_buy:
                # Single transition point
                buy_threshold = f"{min_buy:+5.1f}% (${buy_transitions[0]['price']:.2f})"
            else:
                # Range of BUY activity
                min_price = next(t['price'] for t in buy_transitions if t['change'] == min_buy)
                max_price = next(t['price'] for t in buy_transitions if t['change'] == max_buy)
                buy_threshold = f"{min_buy:+5.1f}% to {max_buy:+5.1f}% (${min_price:.2f} to ${max_price:.2f})"
        
        # Find SELL threshold - any transition involving SELL (either starting or ending)
        sell_transitions = [t for t in transitions if t['to'] == 'SELL' or t['from'] == 'SELL']
        if sell_transitions:
            # Get the full range of SELL activity
            sell_changes = [t['change'] for t in sell_transitions]
            min_sell = min(sell_changes)  # When SELL signals start
            max_sell = max(sell_changes)  # When SELL signals end
            
            if min_sell == max_sell:
                # Single transition point
                sell_threshold = f"{min_sell:+5.1f}% (${sell_transitions[0]['price']:.2f})"
            else:
                # Range of SELL activity
                min_price = next(t['price'] for t in sell_transitions if t['change'] == min_sell)
                max_price = next(t['price'] for t in sell_transitions if t['change'] == max_sell)
                sell_threshold = f"{min_sell:+5.1f}% to {max_sell:+5.1f}% (${min_price:.2f} to ${max_price:.2f})"
        
        if not collect_for_email:
            # KEY THRESHOLDS - Based on actual transition points
            print(f"\n🎯 KEY THRESHOLDS (0.1% precision):")
            print(f"   📈 BUY: {buy_threshold}")
            print(f"   📉 SELL: {sell_threshold}")
            
            if transitions:
                print(f"   🔄 All Transitions:")
                for t in transitions:
                    print(f"      {t['from']}→{t['to']} at {t['change']:+5.1f}%")
        
        # If collecting for email, return structured data
        if collect_for_email:
            return {
                "symbol": symbol,
                "signal": signal,
                "details": details,
                "description": SYMBOL_CONFIGURATIONS.get(symbol, DEFAULT_CONFIGURATION)["description"],
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "transitions": transitions
            }
        
    except Exception as e:
        if not collect_for_email:
            print(f"❌ Error analyzing {symbol}: {e}")
        return None


def send_trading_signals_email():
    """Send trading signals via email using Brevo"""
    print("📧 PREPARING EMAIL NOTIFICATION")
    print("=" * 40)
    
    try:
        # Initialize email notifier
        email_notifier = EmailNotifier()
        
        if not email_notifier.enabled:
            print("❌ Email notifications are not enabled")
            print("   To enable emails, set these environment variables:")
            print("   - BREVO_API_KEY: Your Brevo API key")
            print("   - SENDER_EMAIL: Your verified sender email")
            print("   - RECIPIENT_EMAIL: Email to receive notifications")
            print("   - EMAIL_ENABLED: Set to 'true'")
            return False
        
        print(f"✅ Email notifications enabled")
        print(f"   From: {email_notifier.sender_name} <{email_notifier.sender_email}>")
        print(f"   To: {email_notifier.recipient_email}")
        
        # Collect signals data for all symbols
        print(f"\n📊 Collecting signals data...")
        signals_data = production_signals_summary(collect_for_email=True)
        
        if not signals_data:
            print("❌ No signals data collected")
            return False
        
        print(f"✅ Collected data for {len(signals_data)} symbols")
        
        # Count signals by type for preview
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for signal_data in signals_data:
            signal = signal_data.get("signal", "HOLD")
            signal_counts[signal] += 1
        
        print(f"   📈 BUY: {signal_counts['BUY']} symbols")
        print(f"   📉 SELL: {signal_counts['SELL']} symbols")
        print(f"   ⏸️ HOLD: {signal_counts['HOLD']} symbols")
        
        # Send email
        print(f"\n📤 Sending email...")
        success = email_notifier.send_trading_signals_email(signals_data)
        
        if success:
            print(f"✅ Trading signals email sent successfully!")
            return True
        else:
            print(f"❌ Failed to send email")
            return False
            
    except Exception as e:
        print(f"❌ Error preparing email: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run the production signals summary with optional email notifications"""
    
    try:
        # Check if email should be sent (based on command line argument or environment variable)
        send_email = len(sys.argv) > 1 and sys.argv[1].lower() in ['email', '--email', '-e']
        send_email = send_email or os.getenv("SEND_EMAIL", "false").lower() == "true"
        
        if send_email:
            print("📧 EMAIL MODE: Generating signals and sending email notification")
            print("=" * 60)
            
            # Send email with signals
            email_success = send_trading_signals_email()
            
            if email_success:
                print(f"\n✅ Email notification sent successfully!")
            else:
                print(f"\n❌ Email notification failed")
                # Still run console summary as fallback
                print(f"\n📊 FALLBACK: Running console summary...")
                production_signals_summary()
        else:
            print("📊 CONSOLE MODE: Displaying signals summary")
            print("=" * 50)
            print("💡 Tip: Run with 'email' argument or set SEND_EMAIL=true to send email notifications")
            print()
            
            # Production summary (clean and fast)
            production_signals_summary()
        
        print(f"\n✅ Analysis completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
