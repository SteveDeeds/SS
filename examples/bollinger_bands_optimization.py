#!/usr/bin/env python3
"""
Bollinger Bands Strategy Continuous Optimization Demo

This example demonstrates how to run Bollinger Bands strategy optimization continuously                 # Show CSV file status every 20 combinations
                if combination_num % 20 == 0:
                    print(f"📄 CSV Update: {combination_num}/{len(all_combinations)} combinations for {symbol}")
                    if os.path.exists(output_csv):
                        with open(output_csv, 'r') as f:
                            lines = f.readlines()
                        print(f"   File contains {len(lines)} lines (including header)")
                    
                    # Show timing statistics for last 20 combinations
                    print(f"   ⏱️  Last combination took {duration:.1f}s")
            
            print(f"\n✅ {symbol} completed! ({len(all_combinations)} combinations)")
            print(f"   Results saved to: {output_csv}")
            print(f"📊 Total combinations tested so far: {total_combinations}")
            
            # Log stock completion
            with open(timing_log_file, 'a') as log_file:
                log_file.write(f"\n--- {symbol} COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
constantly updating a CSV file with results. Each parameter combination
gets added as a new row to enable real-time monitoring of optimization progress.

Perfect for long-running Bollinger Bands parameter searches that need to be monitored
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
    current_file = os.path.abspath(__file__)  # examples/rsi_optimization.py
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

# Define Bollinger Bands parameter ranges to explore (must match Bollinger Bands strategy's valid ranges)
periods = range(5, 50, 5)  # Period: 10, 15, 20, 25, 30, 35, 40, 45
std_positive_values = [0.5, 1.0, 1.5, 2.0, 2.5]  # Standard deviations for upper band
std_negative_values = [1.0, 1.5, 2.0, 2.5, 3.0]  # Standard deviations for lower band
cash_percentages = [0.10]

# List of stocks to analyze
symbols = ["IWMY", "AMDY", "YMAX", "MSFT", "MSTY", "ULTY", "NVDY"]


def multi_stock_bollinger_bands_optimization_demo():
    """
    Demonstrate Bollinger Bands optimization across multiple stocks with comprehensive parameter testing
    """
    print("🔄 MULTI-STOCK BOLLINGER BANDS OPTIMIZATION DEMO")
    print("=" * 60)
    print("This demo runs Bollinger Bands optimization for multiple stocks,")
    print("testing all parameter combinations for each stock.")
    print("Results are saved to individual CSV files per stock.")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Create timing log file
    timing_log_file = "results/bollinger_bands_optimization_timing_log.txt"
    
    # Initialize optimizer
    optimizer = GridSearchOptimizer("strategies/bollinger_bands_strategy.py")

    
    print(f"📊 Bollinger Bands parameter space to explore:")
    print(f"   Periods: {list(periods)}")
    print(f"   Std Positive (Upper Band): {std_positive_values}")
    print(f"   Std Negative (Lower Band): {std_negative_values}")
    print(f"   Cash percentages: {cash_percentages}")
    print(f"   Stocks to analyze: {symbols}")
    print(f"   Total combinations per stock: {len(periods) * len(std_positive_values) * len(std_negative_values) * len(cash_percentages)}")
    print(f"   Simulations per combination: 100")
    print(f"   Exploration order: Sequential by stock")
    print(f"   Timing log: {timing_log_file}")
    print()
    
    # Initialize timing log
    with open(timing_log_file, 'w') as log_file:
        log_file.write("=== BOLLINGER BANDS OPTIMIZATION TIMING LOG ===\n")
        log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("Format: [Timestamp] Symbol Combo# period=X std_pos=Y std_neg=Z cash=W -> Duration: Xs ROAIC: Y%\n\n")
    
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
            output_csv = f"results/bollinger_bands_optimization_{symbol}_results.csv"
            
            # Generate all Bollinger Bands parameter combinations for this stock
            all_combinations = list(product(periods, std_positive_values, std_negative_values, cash_percentages))
            
            # Shuffle the combinations for random exploration
            random.shuffle(all_combinations)
            
            print(f"🔀 Testing {len(all_combinations)} Bollinger Bands combinations for {symbol}")
            
            # Run through all combinations for this stock
            for combination_num, (period, std_pos, std_neg, cash_pct) in enumerate(all_combinations, 1):
                # Record start time for this combination
                start_time = time.time()
                start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Create a small parameter grid for this combination
                parameter_grid = {
                    'period': [period],
                    'std_positive': [std_pos],
                    'std_negative': [std_neg],
                    'cash_percentage': [cash_pct]
                }
                
                print(f"🔬 {symbol} Combo {combination_num}/{len(all_combinations)}: "
                      f"period={period}, std_pos={std_pos}, std_neg={std_neg}, cash={cash_pct}")
                
                # Run optimization for this parameter combination
                results = optimizer.optimize(
                    symbol=symbol,
                    parameter_grid=parameter_grid,
                    initial_capital=10000,
                    test_scenarios=100,  # Increased to 100 simulations
                    export_csv=False  # We'll handle CSV export manually
                )
                
                # Calculate duration
                end_time = time.time()
                duration = end_time - start_time
                
                # Append results to CSV file for this stock
                optimizer.export_to_csv(results, output_csv, append_mode=True)
                
                # Show progress
                roaic = results['best_strategy']['roaic']
                if roaic is not None:
                    roaic_str = f"{roaic:.2%}"
                    print(f"   ✅ ROAIC: {roaic_str} (Duration: {duration:.1f}s)")
                else:
                    roaic_str = "None"
                    print(f"   ✅ ROAIC: None (no valid trades) (Duration: {duration:.1f}s)")
                
                # Log timing information
                with open(timing_log_file, 'a') as log_file:
                    log_file.write(f"[{start_timestamp}] {symbol} Combo#{combination_num} "
                                 f"period={period} std_pos={std_pos} std_neg={std_neg} cash={cash_pct} -> "
                                 f"Duration: {duration:.1f}s ROAIC: {roaic_str}\n")
                
                total_combinations += 1
                
                # Brief pause to simulate real-world processing time
                time.sleep(0.1)
                
                # Show CSV file status every 20 combinations
                if combination_num % 20 == 0:
                    print(f"📄 CSV Update: {combination_num}/{len(all_combinations)} combinations for {symbol}")
                    if os.path.exists(output_csv):
                        with open(output_csv, 'r') as f:
                            lines = f.readlines()
                        print(f"   File contains {len(lines)} lines (including header)")
            
            print(f"\n✅ {symbol} completed! ({len(all_combinations)} combinations)")
            print(f"   Results saved to: {output_csv}")
            print(f"� Total combinations tested so far: {total_combinations}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Optimization stopped by user after {total_combinations} combinations")
        # Log interruption
        with open(timing_log_file, 'a') as log_file:
            log_file.write(f"\n!!! INTERRUPTED by user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} !!!\n")
    
    print(f"\n🎉 Multi-stock optimization session completed!")
    print(f"   Total combinations tested: {total_combinations}")
    print(f"   Stocks analyzed: {stock_num if 'stock_num' in locals() else len(symbols)}")
    print(f"   Results files in results/ directory")
    print(f"   Timing log: {timing_log_file}")
    
    # Log session completion
    with open(timing_log_file, 'a') as log_file:
        log_file.write(f"\n=== SESSION COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.write(f"Total combinations tested: {total_combinations}\n")
        if 'stock_num' in locals():
            log_file.write(f"Stocks completed: {stock_num}/{len(symbols)}\n")
    
    # Show summary of all generated files
    print(f"\n📋 Generated Results Files:")
    for symbol in symbols:
        output_csv = f"results/bollinger_bands_optimization_{symbol}_results.csv"
        if os.path.exists(output_csv):
            with open(output_csv, 'r') as f:
                lines = f.readlines()
            print(f"   {symbol}: {output_csv} ({len(lines)} lines)")
        else:
            print(f"   {symbol}: No results file generated")
    
    # Show timing log summary
    if os.path.exists(timing_log_file):
        print(f"\n⏱️  TIMING ANALYSIS:")
        with open(timing_log_file, 'r') as f:
            lines = f.readlines()
        
        # Extract duration information from log
        durations = []
        for line in lines:
            if "Duration:" in line:
                try:
                    # Extract duration from "Duration: X.Xs"
                    duration_part = line.split("Duration: ")[1].split("s")[0]
                    durations.append(float(duration_part))
                except (IndexError, ValueError):
                    continue
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            print(f"   Average time per combination: {avg_duration:.1f}s")
            print(f"   Fastest combination: {min_duration:.1f}s")
            print(f"   Slowest combination: {max_duration:.1f}s")
            print(f"   Total logged combinations: {len(durations)}")
            
            # Estimate remaining time if not all stocks completed
            if 'stock_num' in locals() and stock_num < len(symbols):
                remaining_stocks = len(symbols) - stock_num
                combinations_per_stock = len(periods) * len(std_positive_values) * len(std_negative_values) * len(cash_percentages)
                remaining_combinations = remaining_stocks * combinations_per_stock
                estimated_remaining_time = remaining_combinations * avg_duration
                
                hours = int(estimated_remaining_time // 3600)
                minutes = int((estimated_remaining_time % 3600) // 60)
                print(f"   Estimated time for remaining stocks: {hours}h {minutes}m")
        
        print(f"   Detailed timing log: {timing_log_file}")


def analyze_multi_stock_bollinger_bands_results():
    """
    Analyze the results from multi-stock Bollinger Bands optimization
    """
    print(f"\n📊 ANALYZING MULTI-STOCK BOLLINGER BANDS RESULTS")
    print("=" * 60)
    
    import csv
    
    # Analyze results for each stock
    for symbol in symbols:
        output_csv = f"results/bollinger_bands_optimization_{symbol}_results.csv"
        
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
        output_csv = f"results/rsi_optimization_{symbol}_results.csv"
        
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


def analyze_continuous_bollinger_bands_results():
    """
    Analyze the results from continuous Bollinger Bands optimization
    """
    output_csv = "results/continuous_bollinger_bands_optimization_results.csv"
    
    if not os.path.exists(output_csv):
        print(f"❌ Results file not found: {output_csv}")
        print("Run the continuous optimization demo first.")
        return
    
    print(f"\n📊 ANALYZING CONTINUOUS BOLLINGER BANDS RESULTS")
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
    """Run the multi-stock Bollinger Bands optimization demo"""
    print("🚀 MULTI-STOCK BOLLINGER BANDS OPTIMIZATION WITH COMPREHENSIVE TESTING")
    print("This demo runs Bollinger Bands optimizations across multiple stocks with")
    print("100 simulations per parameter combination for robust results.")
    print()
    
    choice = input("Choose demo:\n1. Run multi-stock Bollinger Bands optimization\n2. Analyze existing Bollinger Bands results\n3. Both\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        multi_stock_bollinger_bands_optimization_demo()
    elif choice == "2":
        analyze_multi_stock_bollinger_bands_results()
    elif choice == "3":
        multi_stock_bollinger_bands_optimization_demo()
        analyze_multi_stock_bollinger_bands_results()
    else:
        print("Invalid choice. Running multi-stock Bollinger Bands optimization by default.")
        multi_stock_bollinger_bands_optimization_demo()
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETED!")


if __name__ == "__main__":
    main()
