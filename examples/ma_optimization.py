#!/usr/bin/env python3
"""
Moving Average Strategy Multi-Stock Optimization Demo

This example demonstrates how to run MA strategy optimization across multiple stocks,
constantly updating CSV files with results. Each parameter combination
gets added as a new row to enable real-time monitoring of optimization progress.

Perfect for long-running MA parameter searches that need to be monitored
while running.
"""

import os
import sys
import time
import random
from datetime import datetime
from itertools import product
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.grid_search import GridSearchOptimizer

# Fix data directory path issue for caching
def setup_data_directory():
    """Ensure we use the main data directory for caching"""
    import sys
    import os
    
    # Get the project root directory
    current_file = os.path.abspath(__file__)  # examples/ma_optimization.py
    examples_dir = os.path.dirname(current_file)  # examples/
    project_root = os.path.dirname(examples_dir)  # SS/
    main_data_dir = os.path.join(project_root, 'data')
    
    # Ensure main data directory exists
    os.makedirs(main_data_dir, exist_ok=True)
    
    # Set environment variable for data directory (if the optimizer respects it)
    os.environ['TRADING_DATA_DIR'] = main_data_dir
    
    return main_data_dir

# Setup data directory before optimization
MAIN_DATA_DIR = setup_data_directory()

# Define parameter ranges to explore (must match strategy's valid ranges)
slow_periods = range(5,20,1)
fast_periods = range(1,16,1)
cash_percentages = [0.10]

# List of stocks to analyze
symbols = ["IWMY", "AMDY", "YMAX", "MSFT", "MSTY", "ULTY", "NVDY", "SPXL"]


def multi_stock_ma_optimization_demo():
    """
    Demonstrate MA optimization across multiple stocks with comprehensive parameter testing
    """
    print("🔄 MULTI-STOCK MA OPTIMIZATION DEMO")
    print("=" * 60)
    print("This demo runs MA optimization for multiple stocks,")
    print("testing all parameter combinations for each stock.")
    print("Results are saved to individual CSV files per stock.")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Initialize optimizer
    optimizer = GridSearchOptimizer("strategies/adaptive_ma_crossover.py")

    
    print(f"📊 MA parameter space to explore:")
    print(f"   Slow periods: {list(slow_periods)}")
    print(f"   Fast periods: {list(fast_periods)}")
    print(f"   Cash percentages: {cash_percentages}")
    print(f"   Stocks to analyze: {symbols}")
    print(f"   Total combinations per stock: {len(slow_periods) * len(fast_periods) * len(cash_percentages)}")
    print(f"   Simulations per combination: 100")
    print(f"   Exploration order: Sequential by stock")
    print()
    
    # Pre-load data for all symbols to ensure proper caching
    print("📥 Pre-loading data for all symbols to ensure cache consistency...")
    from src.data.loader import get_symbol_data
    
    for symbol in symbols:
        print(f"   Loading {symbol}...")
        try:
            data = get_symbol_data(symbol, period="1y", force_download=False)
            if data:
                print(f"   ✅ {symbol}: {len(data)} days loaded")
            else:
                print(f"   ⚠️  {symbol}: No data loaded, will download during optimization")
        except Exception as e:
            print(f"   ❌ {symbol}: Error loading data - {e}")
    
    print("✅ Data pre-loading completed\n")
    
    # Run optimization for each stock
    total_combinations = 0
    
    try:
        for stock_num, symbol in enumerate(symbols, 1):
            print(f"\n🚀 Analyzing Stock {stock_num}/{len(symbols)}: {symbol}")
            print("=" * 50)
            
            # Output CSV file for this stock
            output_csv = f"results/ma_optimization_{symbol}_results.csv"
            
            # Generate all MA parameter combinations for this stock
            all_combinations = list(product(slow_periods, fast_periods, cash_percentages))
            
            # Filter out invalid combinations (fast must be less than slow)
            valid_combinations = [(slow, fast, cash) for slow, fast, cash in all_combinations if fast < slow]
            
            # Shuffle the combinations for random exploration
            random.shuffle(valid_combinations)
            
            print(f"🔀 Testing {len(valid_combinations)} MA combinations for {symbol}")
            
            # Run through all combinations for this stock
            for combination_num, (slow_period, fast_period, cash_pct) in enumerate(valid_combinations, 1):
                # Create a small parameter grid for this combination
                parameter_grid = {
                    'slow_period': [slow_period],
                    'fast_period': [fast_period],
                    'cash_percentage': [cash_pct]
                }
                
                print(f"🔬 {symbol} Combo {combination_num}/{len(valid_combinations)}: "
                      f"slow={slow_period}, fast={fast_period}, cash={cash_pct}")
                
                # Run optimization for this parameter combination
                results = optimizer.optimize(
                    symbol=symbol,
                    parameter_grid=parameter_grid,
                    initial_capital=10000,
                    test_scenarios=100,  # Increased to 100 simulations
                    export_csv=False  # We'll handle CSV export manually
                )
                
                # Append results to CSV file for this stock
                optimizer.export_to_csv(results, output_csv, append_mode=True)
                
                # Show progress
                roaic = results['best_strategy']['roaic']
                if roaic is not None:
                    print(f"   ✅ ROAIC: {roaic:.2%}")
                else:
                    print(f"   ✅ ROAIC: None (no valid trades)")
                
                total_combinations += 1
                
                # Brief pause to simulate real-world processing time
                time.sleep(0.1)
                
                # Show CSV file status every 20 combinations
                if combination_num % 20 == 0:
                    print(f"📄 CSV Update: {combination_num}/{len(valid_combinations)} combinations for {symbol}")
                    if os.path.exists(output_csv):
                        with open(output_csv, 'r') as f:
                            lines = f.readlines()
                        print(f"   File contains {len(lines)} lines (including header)")
            
            print(f"\n✅ {symbol} completed! ({len(valid_combinations)} combinations)")
            print(f"   Results saved to: {output_csv}")
            print(f"� Total combinations tested so far: {total_combinations}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Optimization stopped by user after {total_combinations} combinations")
    
    print(f"\n🎉 Multi-stock optimization session completed!")
    print(f"   Total combinations tested: {total_combinations}")
    print(f"   Stocks analyzed: {stock_num if 'stock_num' in locals() else len(symbols)}")
    print(f"   Results files in results/ directory")
    
    # Show summary of all generated files
    print(f"\n📋 Generated Results Files:")
    for symbol in symbols:
        output_csv = f"results/ma_optimization_{symbol}_results.csv"
        if os.path.exists(output_csv):
            with open(output_csv, 'r') as f:
                lines = f.readlines()
            print(f"   {symbol}: {output_csv} ({len(lines)} lines)")
        else:
            print(f"   {symbol}: No results file generated")


def analyze_multi_stock_ma_results():
    """
    Analyze the results from multi-stock MA optimization
    """
    print(f"\n📊 ANALYZING MULTI-STOCK MA RESULTS")
    print("=" * 60)
    
    import csv
    
    # Analyze results for each stock
    for symbol in symbols:
        output_csv = f"results/ma_optimization_{symbol}_results.csv"
        
        if not os.path.exists(output_csv):
            print(f"❌ Results file not found for {symbol}: {output_csv}")
            continue
        
        print(f"\n📈 {symbol} Analysis:")
        print("-" * 30)
        
        with open(output_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        if not results:
            print(f"❌ No results found in CSV file for {symbol}")
            continue
        
        print(f"📊 Found {len(results)} optimization results for {symbol}")
        
        # Convert ROAIC to float and filter valid results
        valid_results = []
        for result in results:
            try:
                roaic = float(result.get('return_on_avg_invested_capital', 0))
                if roaic != 0:  # Filter out zero/invalid results
                    result['roaic_float'] = roaic
                    valid_results.append(result)
            except ValueError:
                continue
        
        if not valid_results:
            print(f"❌ No valid ROAIC results found for {symbol}")
            continue
        
        print(f"✅ Found {len(valid_results)} valid results with non-zero ROAIC for {symbol}")
        
        # Find best and worst results
        best_result = max(valid_results, key=lambda x: x['roaic_float'])
        worst_result = min(valid_results, key=lambda x: x['roaic_float'])
        
        print(f"\n🏆 BEST RESULT for {symbol}:")
        print(f"   ROAIC: {best_result['roaic_float']:.2%}")
        print(f"   Parameters: {best_result.get('parameter_combination', 'unknown')}")
        print(f"   datetime: {best_result.get('datetime', 'unknown')}")
        
        print(f"\n📉 WORST RESULT for {symbol}:")
        print(f"   ROAIC: {worst_result['roaic_float']:.2%}")
        print(f"   Parameters: {worst_result.get('parameter_combination', 'unknown')}")
        print(f"   datetime: {worst_result.get('datetime', 'unknown')}")
        
        # Calculate statistics
        roaic_values = [r['roaic_float'] for r in valid_results]
        avg_roaic = sum(roaic_values) / len(roaic_values)
        
        print(f"\n📊 STATISTICS for {symbol}:")
        print(f"   Average ROAIC: {avg_roaic:.2%}")
        print(f"   Best ROAIC: {max(roaic_values):.2%}")
        print(f"   Worst ROAIC: {min(roaic_values):.2%}")
        print(f"   Range: {max(roaic_values) - min(roaic_values):.2%}")
    
    # Create summary comparison across all stocks
    print(f"\n🌟 CROSS-STOCK SUMMARY:")
    print("=" * 40)
    
    all_best_results = []
    
    for symbol in symbols:
        output_csv = f"results/ma_optimization_{symbol}_results.csv"
        
        if not os.path.exists(output_csv):
            continue
            
        with open(output_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        valid_results = []
        for result in results:
            try:
                roaic = float(result.get('return_on_avg_invested_capital', 0))
                if roaic != 0:
                    result['roaic_float'] = roaic
                    result['symbol'] = symbol
                    valid_results.append(result)
            except ValueError:
                continue
        
        if valid_results:
            best_for_symbol = max(valid_results, key=lambda x: x['roaic_float'])
            all_best_results.append(best_for_symbol)
    
    if all_best_results:
        # Find overall best performer
        overall_best = max(all_best_results, key=lambda x: x['roaic_float'])
        
        print(f"\n🏆 OVERALL BEST PERFORMER:")
        print(f"   Symbol: {overall_best['symbol']}")
        print(f"   ROAIC: {overall_best['roaic_float']:.2%}")
        print(f"   Parameters: {overall_best.get('parameter_combination', 'unknown')}")
        
        # Show top 3 performers
        sorted_results = sorted(all_best_results, key=lambda x: x['roaic_float'], reverse=True)
        print(f"\n🥇 TOP 3 PERFORMING STOCKS:")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"   {i}. {result['symbol']}: {result['roaic_float']:.2%}")


def analyze_continuous_results():
    """
    Analyze the results from continuous optimization (legacy function for backward compatibility)
    """
    output_csv = "results/continuous_optimization_results.csv"
    
    if not os.path.exists(output_csv):
        print(f"❌ Results file not found: {output_csv}")
        print("Run the multi-stock optimization demo first.")
        return
    
    print(f"\n📊 ANALYZING CONTINUOUS RESULTS")
    print("=" * 60)
    
    import csv
    
    with open(output_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if not results:
        print("❌ No results found in CSV file")
        return
    
    print(f"📈 Analysis of {len(results)} optimization results:")
    
    # Convert ROAIC to float and filter valid results
    valid_results = []
    for result in results:
        try:
            roaic = float(result.get('return_on_avg_invested_capital', 0))
            if roaic != 0:  # Filter out zero/invalid results
                result['roaic_float'] = roaic
                valid_results.append(result)
        except ValueError:
            continue
    
    if not valid_results:
        print("❌ No valid ROAIC results found")
        return
    
    print(f"✅ Found {len(valid_results)} valid results with non-zero ROAIC")
    
    # Find best and worst results
    best_result = max(valid_results, key=lambda x: x['roaic_float'])
    worst_result = min(valid_results, key=lambda x: x['roaic_float'])
    
    print(f"\n🏆 BEST RESULT:")
    print(f"   ROAIC: {best_result['roaic_float']:.2%}")
    print(f"   Parameters: {best_result.get('parameter_combination', 'unknown')}")
    print(f"   datetime: {best_result.get('datetime', 'unknown')}")
    
    print(f"\n📉 WORST RESULT:")
    print(f"   ROAIC: {worst_result['roaic_float']:.2%}")
    print(f"   Parameters: {worst_result.get('parameter_combination', 'unknown')}")
    print(f"   datetime: {worst_result.get('datetime', 'unknown')}")
    
    # Calculate statistics
    roaic_values = [r['roaic_float'] for r in valid_results]
    avg_roaic = sum(roaic_values) / len(roaic_values)
    
    print(f"\n📊 STATISTICS:")
    print(f"   Average ROAIC: {avg_roaic:.2%}")
    print(f"   Best ROAIC: {max(roaic_values):.2%}")
    print(f"   Worst ROAIC: {min(roaic_values):.2%}")
    print(f"   Range: {max(roaic_values) - min(roaic_values):.2%}")
    
    # Show parameter distribution if available
    param_columns = [col for col in results[0].keys() if col.startswith('param_')]
    if param_columns:
        print(f"\n🔧 PARAMETER ANALYSIS:")
        for param_col in param_columns:
            param_name = param_col.replace('param_', '')
            param_values = []
            param_roaic = []
            
            for result in valid_results:
                try:
                    param_val = float(result.get(param_col, 0))
                    param_values.append(param_val)
                    param_roaic.append(result['roaic_float'])
                except ValueError:
                    continue
            
            if param_values:
                # Find parameter value with best average ROAIC
                from collections import defaultdict
                param_performance = defaultdict(list)
                
                for val, roaic in zip(param_values, param_roaic):
                    param_performance[val].append(roaic)
                
                avg_performance = {val: sum(roaics)/len(roaics) 
                                 for val, roaics in param_performance.items()}
                
                best_param_val = max(avg_performance.keys(), key=lambda x: avg_performance[x])
                
                print(f"   {param_name}: best value = {best_param_val} "
                      f"(avg ROAIC: {avg_performance[best_param_val]:.2%})")


def main():
    """Run the multi-stock MA optimization demo"""
    print("🚀 MULTI-STOCK MA OPTIMIZATION WITH COMPREHENSIVE TESTING")
    print("This demo runs MA optimizations across multiple stocks with")
    print("100 simulations per parameter combination for robust results.")
    print()
    
    choice = input("Choose demo:\n1. Run multi-stock MA optimization\n2. Analyze existing MA results\n3. Both\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        multi_stock_ma_optimization_demo()
    elif choice == "2":
        analyze_multi_stock_ma_results()
    elif choice == "3":
        multi_stock_ma_optimization_demo()
        analyze_multi_stock_ma_results()
    else:
        print("Invalid choice. Running multi-stock MA optimization by default.")
        multi_stock_ma_optimization_demo()
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETED!")


if __name__ == "__main__":
    main()
