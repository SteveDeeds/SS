#!/usr/bin/env python3
"""
Continuous Optimization Demo

This example demonstrates how to run optimization continuously for hours,
constantly updating a CSV file with results. Each parameter combination
gets added as a new row to enable real-time monitoring of optimization progress.

Perfect for long-running parameter searches that need to be monitored
while running.
"""

import os
import sys
import time
import random
from itertools import product
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.grid_search import GridSearchOptimizer

# Define parameter ranges to explore (must match strategy's valid ranges)
slow_periods = range(5,20,1)
fast_periods = range(1,16,1)
cash_percentages = [0.10]
symbol="SDIV"


def continuous_optimization_demo():
    """
    Demonstrate continuous optimization with real-time CSV updates
    """
    print("🔄 CONTINUOUS OPTIMIZATION DEMO")
    print("=" * 60)
    print("This demo runs multiple optimization rounds CONTINUOUSLY,")
    print("updating a CSV file with results. After completing all")
    print("parameter combinations, it reshuffles and starts again.")
    print("Perfect for long-running parameter searches.")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Output CSV file that will be continuously updated
    output_csv = "results/continuous_optimization_results.csv"
    
    # Initialize optimizer
    optimizer = GridSearchOptimizer("strategies/adaptive_ma_crossover.py")

    
    print(f"📊 Parameter space to explore:")
    print(f"   Slow periods: {slow_periods}")
    print(f"   Fast periods: {fast_periods}")
    print(f"   Cash percentages: {cash_percentages}")
    print(f"   Total combinations per cycle: {len(slow_periods) * len(fast_periods) * len(cash_percentages)}")
    print(f"   Exploration order: Shuffled (random)")
    print(f"   Output file: {output_csv}")
    print(f"   🔄 Mode: CONTINUOUS (infinite cycles)")
    print()
    
    # Run optimization continuously
    total_combinations = 0
    cycle_count = 0
    
    try:
        while True:  # Infinite loop
            cycle_count += 1
            print(f"\n🚀 Starting Cycle {cycle_count}")
            print("=" * 40)
            
            # Generate all parameter combinations for this cycle
            all_combinations = list(product(slow_periods, fast_periods, cash_percentages))
            
            # Shuffle the combinations for random exploration
            random.shuffle(all_combinations)
            
            print(f"🔀 Shuffled {len(all_combinations)} combinations for Cycle {cycle_count}")
            
            # Run through all combinations in this cycle
            for combination_num, (slow_period, fast_period, cash_pct) in enumerate(all_combinations, 1):
                # Create a small parameter grid for this combination
                parameter_grid = {
                    'slow_period': [slow_period],
                    'fast_period': [fast_period],
                    'cash_percentage': [cash_pct]
                }
                
                print(f"🔬 Cycle {cycle_count}, Combo {combination_num}/{len(all_combinations)}: "
                      f"slow={slow_period}, fast={fast_period}, cash={cash_pct}")
                
                # Run optimization for this parameter combination
                results = optimizer.optimize(
                    symbol=symbol,
                    parameter_grid=parameter_grid,
                    initial_capital=10000,
                    test_scenarios=20,  # 20 gives us 5% resolution for percentile
                    export_csv=False  # We'll handle CSV export manually
                )
                
                # Append results to CSV file
                optimizer.export_to_csv(results, output_csv, append_mode=True)
                
                # Show progress
                roaic = results['best_strategy']['roaic']
                if roaic is not None:
                    print(f"   ✅ ROAIC: {roaic:.2%}")
                else:
                    print(f"   ✅ ROAIC: None (no valid trades)")
                
                total_combinations += 1
                
                # Brief pause to simulate real-world processing time
                # In real scenarios, this would be the natural processing time
                time.sleep(0.5)
                
                # Show CSV file status every few combinations
                if combination_num % 10 == 0:
                    print(f"📄 CSV Update: {total_combinations} total combinations tested")
                    if os.path.exists(output_csv):
                        with open(output_csv, 'r') as f:
                            lines = f.readlines()
                        print(f"   File contains {len(lines)} lines (including header)")
            
            print(f"\n✅ Cycle {cycle_count} completed! ({len(all_combinations)} combinations)")
            print(f"📊 Total combinations tested so far: {total_combinations}")
            print(f"🔄 Starting next cycle in 2 seconds...")
            time.sleep(2)  # Brief pause between cycles
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Optimization stopped by user after {total_combinations} combinations")
        print(f"   Completed {cycle_count} full cycles")
    
    print(f"\n🎉 Continuous optimization session ended!")
    print(f"   Total combinations tested: {total_combinations}")
    print(f"   Full cycles completed: {cycle_count}")
    print(f"   Results file: {output_csv}")
    
    # Show final CSV stats
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            lines = f.readlines()
        print(f"   Final CSV contains {len(lines)} lines")
        
        # Show last few lines as preview
        print(f"\n📋 Last few results:")
        for line in lines[-3:]:
            if ',' in line:  # Skip header if it's in the last few lines
                parts = line.strip().split(',')
                if len(parts) >= 7:  # Ensure we have enough columns
                    datetime = parts[0]
                    params = parts[5] if len(parts) > 5 else "unknown"
                    roaic = parts[8] if len(parts) > 8 else "0"
                    print(f"   {datetime}: {params} → ROAIC: {roaic}")


def analyze_continuous_results():
    """
    Analyze the results from continuous optimization
    """
    output_csv = "results/continuous_optimization_results.csv"
    
    if not os.path.exists(output_csv):
        print(f"❌ Results file not found: {output_csv}")
        print("Run the continuous optimization demo first.")
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
    """Run the continuous optimization demo"""
    print("🚀 CONTINUOUS OPTIMIZATION WITH REAL-TIME CSV UPDATES")
    print("This demo shows how to run optimizations for hours while")
    print("continuously updating a CSV file for real-time monitoring.")
    print()
    
    choice = input("Choose demo:\n1. Run continuous optimization\n2. Analyze existing results\n3. Both\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        continuous_optimization_demo()
    elif choice == "2":
        analyze_continuous_results()
    elif choice == "3":
        continuous_optimization_demo()
        analyze_continuous_results()
    else:
        print("Invalid choice. Running continuous optimization by default.")
        continuous_optimization_demo()
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETED!")


if __name__ == "__main__":
    main()
