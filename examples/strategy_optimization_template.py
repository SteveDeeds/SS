#!/usr/bin/env python3
"""
Strategy Optimization Template

Template for creating strategy-specific optimization scripts.
Copy this file and modify the STRATEGY_CONFIG section for your strategy.

This provides the same functionality as generic_strategy_optimization.py
but focused on a single strategy with easy customization.
"""

import os
import sys
from itertools import product
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.single_parameter_optimizer import SingleParameterOptimizer
from src.data.loader import get_symbol_data

# ============================================================================
# STRATEGY CONFIGURATION - MODIFY THIS SECTION FOR YOUR STRATEGY
# ============================================================================

# Strategy file path (relative to project root)
STRATEGY_FILE = "strategies/bollinger_bands_strategy.py"

# Strategy display name
STRATEGY_NAME = "Bollinger Bands"

# Strategy type (used for file naming)
STRATEGY_TYPE = "bollinger_bands"

# Parameter grid to explore
PARAMETER_GRID = {
    'period': [5, 10, 15, 20, 25, 30],
    'std_positive': [0.5, 1.0, 1.5, 2.0, 2.5],
    'std_negative': [0.5, 1.0, 1.5, 2.0, 2.5],
    'cash_percentage': [0.10]
}

# Custom validation function for parameter combinations
def validate_parameters(params: Dict[str, Any]) -> bool:
    """
    Validate parameter combinations.
    
    Args:
        params: Parameter dictionary to validate
        
    Returns:
        True if parameters are valid, False otherwise
    """
    # For Bollinger Bands, all combinations are valid
    return True
    
    # Example validation for Moving Average strategy:
    # buy_fast = params.get('buy_fast_period', 0)
    # buy_slow = params.get('buy_slow_period', 999)
    # sell_fast = params.get('sell_fast_period', 0)
    # sell_slow = params.get('sell_slow_period', 999)
    # return buy_fast < buy_slow and sell_fast < sell_slow
    
    # Example validation for RSI strategy:
    # oversold = params.get('oversold_threshold', 0)
    # overbought = params.get('overbought_threshold', 100)
    # return oversold < overbought

# ============================================================================
# END STRATEGY CONFIGURATION
# ============================================================================

# List of stocks to analyze
SYMBOLS = ["IWMY", "AMDY", "YMAX", "MSFT", "MSTY", "ULTY", "NVDY", "SPXL", "AAPL", 
           "AVGO", "GOOGL", "GOOG", "META", "NFLX", "TMUS", 
           "AMZN", "TSLA", "HD", "SBUX", "LLY", "UNH", "JNJ", "ABBV", "JPM", "V", 
           "MA", "BAC", "GE", "RTX", "HON", "CAT", "WMT", "PG", "KO", "PEP", "XOM", 
           "CVX", "COP", "EOG", "NEE", "DUK", "SRE", "SO", "LIN", "SHW", "APD", 
           "ECL", "PLD", "AMT", "EQIX", "CCI", "NVDA"]


def get_existing_results() -> Dict[str, str]:
    """Get existing result files for this strategy"""
    import glob
    results_pattern = f"results/{STRATEGY_TYPE}_optimization_*_results.csv"
    existing_files = glob.glob(results_pattern)
    
    symbol_files = {}
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        prefix = f"{STRATEGY_TYPE}_optimization_"
        suffix = "_results.csv"
        
        if filename.startswith(prefix) and filename.endswith(suffix):
            symbol = filename[len(prefix):-len(suffix)]
            symbol_files[symbol] = file_path
    
    return symbol_files


def prioritize_symbols() -> List[str]:
    """Prioritize symbols based on existing results"""
    existing_files = get_existing_results()
    
    # Find symbols without existing results
    missing_symbols = [symbol for symbol in SYMBOLS if symbol not in existing_files]
    
    # Find symbols with existing results and sort by file age (oldest first)
    symbols_with_files = []
    for symbol in SYMBOLS:
        if symbol in existing_files:
            try:
                file_path = existing_files[symbol]
                file_time = os.path.getmtime(file_path)
                symbols_with_files.append((symbol, file_time))
            except OSError:
                missing_symbols.append(symbol)
    
    symbols_with_files.sort(key=lambda x: x[1])
    ordered_existing_symbols = [symbol for symbol, _ in symbols_with_files]
    
    prioritized_list = missing_symbols + ordered_existing_symbols
    
    print(f"📋 {STRATEGY_NAME} Symbol processing order:")
    if missing_symbols:
        print(f"   🆕 {len(missing_symbols)} symbols without results: {missing_symbols}")
    if ordered_existing_symbols:
        print(f"   🔄 {len(ordered_existing_symbols)} symbols with results (oldest first): {ordered_existing_symbols}")
    
    return prioritized_list


def generate_valid_combinations() -> List[Dict[str, Any]]:
    """Generate valid parameter combinations"""
    # Generate all possible combinations
    param_names = list(PARAMETER_GRID.keys())
    param_values = list(PARAMETER_GRID.values())
    all_combinations = list(product(*param_values))
    
    # Convert to parameter dictionaries and validate
    valid_combinations = []
    for combination in all_combinations:
        param_dict = dict(zip(param_names, combination))
        
        if validate_parameters(param_dict):
            valid_combinations.append(param_dict)
    
    return valid_combinations


def optimize_strategy():
    """Run strategy optimization across all symbols with smart prioritization"""
    print(f"🔄 {STRATEGY_NAME.upper()} PARAMETER OPTIMIZATION")
    print("=" * 60)
    print("Processing ALL symbols in smart priority order:")
    print("1. Symbols without results first")
    print("2. Symbols with results, oldest first")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Get prioritized symbols and valid combinations
    prioritized_symbols = prioritize_symbols()
    valid_combinations = generate_valid_combinations()
    
    if not prioritized_symbols:
        print("❌ No symbols to process")
        return
    
    if not valid_combinations:
        print("❌ No valid parameter combinations found")
        return
    
    # Initialize optimizer
    optimizer = SingleParameterOptimizer(STRATEGY_FILE)
    
    print(f"\n📊 {STRATEGY_NAME} parameter space to explore:")
    for param_name, param_values in PARAMETER_GRID.items():
        print(f"   {param_name}: {param_values}")
    print(f"   Symbols to process: {len(prioritized_symbols)}")
    print(f"   Valid combinations per stock: {len(valid_combinations)}")
    print(f"   Simulations per combination: 100")
    print()
    
    # Run optimization for prioritized symbols
    total_combinations = 0
    
    try:
        for stock_num, symbol in enumerate(prioritized_symbols, 1):
            print(f"\n🚀 Analyzing Stock {stock_num}/{len(prioritized_symbols)}: {symbol}")
            print("=" * 50)
            
            # Load historical data once for this symbol
            print(f"📡 Loading historical data for {symbol}...")
            historical_data = get_symbol_data(symbol, period="4y", force_download=False)
            
            if not historical_data:
                print(f"❌ Failed to load data for {symbol}, skipping...")
                continue
            
            print(f"✅ Loaded {len(historical_data)} days of historical data")
            
            # Output CSV file for this stock
            output_csv = f"results/{STRATEGY_TYPE}_optimization_{symbol}_results.csv"
            
            print(f"🔀 Testing {len(valid_combinations)} {STRATEGY_NAME} combinations for {symbol}")
            
            # Run through all valid combinations for this stock
            for combination_num, parameters in enumerate(valid_combinations, 1):
                print(f"🔬 {symbol} Combo {combination_num}/{len(valid_combinations)}: {parameters}")
                
                # Test this parameter combination
                results = optimizer.test_single_parameters(
                    historical_data=historical_data,
                    parameters=parameters,
                    symbol=symbol,
                    initial_capital=10000,
                    test_scenarios=100,
                    scenario_length=352,
                    warmup_days=100
                )
                
                # Export results to CSV file for this stock
                optimizer.export_results_to_csv(results, output_csv, append_mode=True)
                
                # Show progress
                roaic = results.get('roaic')
                if roaic is not None:
                    print(f"   ✅ ROAIC: {roaic:.2%}")
                else:
                    print(f"   ✅ ROAIC: None (no valid trades)")
                
                total_combinations += 1
                
                # Show CSV file status periodically
                if combination_num % max(1, len(valid_combinations) // 5) == 0:
                    print(f"📄 CSV Update: {combination_num}/{len(valid_combinations)} combinations for {symbol}")
                    if os.path.exists(output_csv):
                        with open(output_csv, 'r') as f:
                            lines = f.readlines()
                        print(f"   File contains {len(lines)} lines (including header)")
            
            print(f"\n✅ {symbol} completed! ({len(valid_combinations)} combinations)")
            print(f"   Results saved to: {output_csv}")
            print(f"📊 Total combinations tested so far: {total_combinations}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Optimization stopped by user after {total_combinations} combinations")
    
    print(f"\n✅ {STRATEGY_NAME} optimization completed!")
    print(f"   Total combinations tested: {total_combinations}")
    print(f"   Stocks analyzed: {stock_num if 'stock_num' in locals() else len(prioritized_symbols)}")
    print(f"   Results saved in results/ directory")
    
    # Show summary of generated files
    show_results_summary(prioritized_symbols)


def show_results_summary(processed_symbols: List[str]):
    """Show summary of all result files"""
    print(f"\n📋 Generated {STRATEGY_NAME} Results Files:")
    for symbol in processed_symbols:
        output_csv = f"results/{STRATEGY_TYPE}_optimization_{symbol}_results.csv"
        if os.path.exists(output_csv):
            with open(output_csv, 'r') as f:
                lines = f.readlines()
            print(f"   {symbol}: {output_csv} ({len(lines)} lines)")
        else:
            print(f"   {symbol}: No results file generated")
    
    # Show summary of all existing files for this strategy
    all_existing = get_existing_results()
    if all_existing:
        print(f"\n📊 All Existing {STRATEGY_NAME} Results:")
        for symbol, file_path in sorted(all_existing.items()):
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                file_age = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"   {symbol}: {len(lines)} lines (modified: {file_age.strftime('%Y-%m-%d %H:%M:%S')})")
            except (OSError, IOError):
                print(f"   {symbol}: File exists but cannot read details")


if __name__ == "__main__":
    """Run strategy optimization across all symbols and save results to CSV files"""
    optimize_strategy()
