#!/usr/bin/env python3
"""
Generic Strategy Parameter Optimization

This script provides a unified framework for optimizing any strategy that implements
the OptimizableStrategy interface. It automatically discovers strategy parameters
and generates appropriate parameter combinations for testing.

Supports Bollinger Bands, Moving Average, RSI, and any future strategies.
Results are saved as {strategy_type}_optimization_{SYMBOL}_results.csv
"""

import os
import sys
from itertools import product
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.single_parameter_optimizer import SingleParameterOptimizer
from src.data.loader import get_symbol_data


class StrategyConfig:
    """Configuration for a specific strategy optimization"""
    
    def __init__(self, strategy_file: str, strategy_name: str, parameter_grid: Dict[str, List]):
        self.strategy_file = strategy_file
        self.strategy_name = strategy_name
        self.parameter_grid = parameter_grid
        self.strategy_type = strategy_name.lower().replace(' ', '_')


# Pre-defined strategy configurations
STRATEGY_CONFIGS = {
    'bollinger_bands': StrategyConfig(
        strategy_file="strategies/bollinger_bands_strategy.py",
        strategy_name="Bollinger Bands",
        parameter_grid={
            'period': [5, 10, 15, 20, 25, 30],
            'std_positive': [0.5, 1.0, 1.5, 2.0, 2.5],
            'std_negative': [0.5, 1.0, 1.5, 2.0, 2.5],
            'cash_percentage': [0.10]
        }
    ),
    
    'moving_average': StrategyConfig(
        strategy_file="strategies/adaptive_ma_crossover.py",
        strategy_name="Moving Average",
        parameter_grid={
            'buy_slow_period': [10, 15, 20, 25, 30, 35],
            'buy_fast_period': [3, 5, 7, 9, 11, 13],
            'sell_slow_period': [15, 20, 25, 30, 35, 40],
            'sell_fast_period': [5, 10, 15, 20, 25, 30],
            'cash_percentage': [0.10]
        }
    ),
    
    'rsi': StrategyConfig(
        strategy_file="strategies/rsi_strategy.py",
        strategy_name="RSI",
        parameter_grid={
            'rsi_period': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            'oversold_threshold': [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
            'overbought_threshold': [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
            'cash_percentage': [0.10]
        }
    )
}

# List of stocks to analyze
SYMBOLS = ["IWMY", "AMDY", "YMAX", "MSFT", "MSTY", "ULTY", "NVDY", "SPXL", "AAPL", 
           "AVGO", "GOOGL", "GOOG", "META", "NFLX", "TMUS", 
           "AMZN", "TSLA", "HD", "SBUX", "LLY", "UNH", "JNJ", "ABBV", "JPM", "V", 
           "MA", "BAC", "GE", "RTX", "HON", "CAT", "WMT", "PG", "KO", "PEP", "XOM", 
           "CVX", "COP", "EOG", "NEE", "DUK", "SRE", "SO", "LIN", "SHW", "APD", 
           "ECL", "PLD", "AMT", "EQIX", "CCI", "NVDA"]


def get_all_existing_results() -> Dict[str, Dict[str, str]]:
    """
    Get existing result files for ALL strategies.
    
    Returns:
        Dictionary with structure: {strategy_type: {symbol: file_path}}
    """
    import glob
    all_results = {}
    
    for strategy_key, config in STRATEGY_CONFIGS.items():
        strategy_type = config.strategy_type
        results_pattern = f"results/{strategy_type}_optimization_*_results.csv"
        existing_files = glob.glob(results_pattern)
        
        symbol_files = {}
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            prefix = f"{strategy_type}_optimization_"
            suffix = "_results.csv"
            
            if filename.startswith(prefix) and filename.endswith(suffix):
                symbol = filename[len(prefix):-len(suffix)]
                symbol_files[symbol] = file_path
        
        all_results[strategy_type] = symbol_files
    
    return all_results


def analyze_symbol_strategy_coverage() -> Dict[str, Any]:
    """
    Analyze which symbols have which strategies and identify priorities.
    
    Returns:
        Dictionary with prioritized work items
    """
    all_results = get_all_existing_results()
    strategy_types = list(STRATEGY_CONFIGS.keys())
    
    # Categorize symbols by their coverage
    completely_missing = []      # Symbols with no strategy files at all
    partially_missing = []       # Symbols missing some strategies
    fully_covered = []          # Symbols with all strategies (may need refresh)
    
    # Work items to process
    work_items = []
    
    for symbol in SYMBOLS:
        strategies_present = []
        strategies_missing = []
        file_ages = []
        
        for strategy_key in strategy_types:
            config = STRATEGY_CONFIGS[strategy_key]
            strategy_type = config.strategy_type
            
            if symbol in all_results.get(strategy_type, {}):
                strategies_present.append(strategy_key)
                file_path = all_results[strategy_type][symbol]
                try:
                    file_age = os.path.getmtime(file_path)
                    file_ages.append((strategy_key, file_age, file_path))
                except OSError:
                    # File exists in glob but can't get age - treat as missing
                    strategies_missing.append(strategy_key)
            else:
                strategies_missing.append(strategy_key)
        
        # Categorize this symbol
        if not strategies_present:
            # No strategies at all - highest priority
            completely_missing.append(symbol)
            # Add work items for ALL strategies for this symbol
            for strategy_key in strategy_types:
                work_items.append({
                    'type': 'missing_symbol',
                    'priority': 1,
                    'symbol': symbol,
                    'strategy': strategy_key,
                    'reason': f'New symbol - missing all strategies'
                })
        elif strategies_missing:
            # Some strategies missing - second priority
            partially_missing.append(symbol)
            # Add work items for missing strategies
            for strategy_key in strategies_missing:
                work_items.append({
                    'type': 'missing_strategy',
                    'priority': 2,
                    'symbol': symbol,
                    'strategy': strategy_key,
                    'reason': f'Missing {strategy_key} strategy'
                })
        else:
            # All strategies present - may need refresh (lowest priority)
            fully_covered.append(symbol)
            # Add refresh work items based on file age (oldest first)
            for strategy_key, file_age, file_path in sorted(file_ages, key=lambda x: x[1]):
                work_items.append({
                    'type': 'refresh',
                    'priority': 3,
                    'symbol': symbol,
                    'strategy': strategy_key,
                    'file_age': file_age,
                    'file_path': file_path,
                    'reason': f'Refresh old results (age: {datetime.fromtimestamp(file_age).strftime("%Y-%m-%d")})'
                })
    
    return {
        'completely_missing': completely_missing,
        'partially_missing': partially_missing,
        'fully_covered': fully_covered,
        'work_items': sorted(work_items, key=lambda x: (x['priority'], x.get('file_age', 0))),
        'all_results': all_results
    }


class GenericStrategyOptimizer:
    """Generic optimizer for any strategy that implements OptimizableStrategy interface"""
    
    def __init__(self, strategy_config: StrategyConfig):
        self.config = strategy_config
        self.optimizer = SingleParameterOptimizer(strategy_config.strategy_file)
    
    def generate_valid_combinations(self) -> List[Dict[str, Any]]:
        """Generate valid parameter combinations with strategy-specific validation"""
        # Generate all possible combinations
        param_names = list(self.config.parameter_grid.keys())
        param_values = list(self.config.parameter_grid.values())
        all_combinations = list(product(*param_values))
        
        # Convert to parameter dictionaries
        param_dicts = []
        for combination in all_combinations:
            param_dict = dict(zip(param_names, combination))
            
            # Apply strategy-specific validation
            if self._is_valid_combination(param_dict):
                param_dicts.append(param_dict)
        
        return param_dicts
    
    def _is_valid_combination(self, params: Dict[str, Any]) -> bool:
        """Validate parameter combinations based on strategy type"""
        strategy_type = self.config.strategy_type
        
        if strategy_type == 'moving_average':
            # For MA: fast periods must be less than slow periods
            buy_fast = params.get('buy_fast_period', 0)
            buy_slow = params.get('buy_slow_period', 999)
            sell_fast = params.get('sell_fast_period', 0)
            sell_slow = params.get('sell_slow_period', 999)
            
            return buy_fast < buy_slow and sell_fast < sell_slow
        
        elif strategy_type == 'rsi':
            # For RSI: oversold threshold must be less than overbought threshold
            oversold = params.get('oversold_threshold', 0)
            overbought = params.get('overbought_threshold', 100)
            
            return oversold < overbought
        
        # For other strategies (like Bollinger Bands), all combinations are valid
        return True
    
    def optimize_single_work_item(self, work_item: Dict[str, Any], test_scenarios: int = 100, scenario_length: int = 352) -> bool:
        """
        Run optimization for a single work item (symbol + strategy combination).
        
        Args:
            work_item: Work item dictionary with symbol, strategy, and other details
            test_scenarios: Number of scenarios to test
            scenario_length: Length of each scenario
            
        Returns:
            True if successful, False if failed
        """
        symbol = work_item['symbol']
        strategy_key = work_item['strategy']
        
        # Get the strategy config for this work item
        if strategy_key not in STRATEGY_CONFIGS:
            print(f"❌ Unknown strategy: {strategy_key}")
            return False
        
        strategy_config = STRATEGY_CONFIGS[strategy_key]
        
        # Create optimizer for this strategy if needed
        if self.config.strategy_type != strategy_config.strategy_type:
            temp_optimizer = GenericStrategyOptimizer(strategy_config)
        else:
            temp_optimizer = self
        
        print(f"\n🔬 {work_item['type'].upper()}: {symbol} - {strategy_config.strategy_name}")
        print(f"   Reason: {work_item['reason']}")
        
        # Delete existing file if this is a refresh operation
        output_csv = f"results/{strategy_config.strategy_type}_optimization_{symbol}_results.csv"
        if work_item['type'] == 'refresh' and os.path.exists(output_csv):
            print(f"🗑️ Deleting old results file: {output_csv}")
            try:
                os.remove(output_csv)
            except OSError as e:
                print(f"⚠️ Could not delete old file: {e}")
        
        # Load historical data once for this symbol
        print(f"📡 Loading historical data for {symbol}...")
        historical_data = get_symbol_data(symbol, period="4y", force_download=False)
        
        if not historical_data:
            print(f"❌ Failed to load data for {symbol}, skipping...")
            return False
        
        print(f"✅ Loaded {len(historical_data)} days of historical data")
        
        # Generate valid combinations for this strategy
        valid_combinations = temp_optimizer.generate_valid_combinations()
        
        if not valid_combinations:
            print(f"❌ No valid parameter combinations for {strategy_config.strategy_name}")
            return False
        
        print(f"🔀 Testing {len(valid_combinations)} {strategy_config.strategy_name} combinations")
        
        # Run through all valid combinations for this symbol
        total_combinations = 0
        try:
            for combination_num, parameters in enumerate(valid_combinations, 1):
                print(f"🔬 {symbol} Combo {combination_num}/{len(valid_combinations)}: {parameters}")
                
                # Test this parameter combination
                results = temp_optimizer.optimizer.test_single_parameters(
                    historical_data=historical_data,
                    parameters=parameters,
                    symbol=symbol,
                    initial_capital=10000,
                    test_scenarios=test_scenarios,
                    scenario_length=scenario_length,
                    warmup_days=100
                )
                
                # Export results to CSV file for this stock
                temp_optimizer.optimizer.export_results_to_csv(results, output_csv, append_mode=True)
                
                # Show progress
                roaic = results.get('roaic')
                if roaic is not None:
                    print(f"   ✅ ROAIC: {roaic:.2%}")
                else:
                    print(f"   ✅ ROAIC: None (no valid trades)")
                
                total_combinations += 1
                
                # Show CSV file status periodically
                if combination_num % max(1, len(valid_combinations) // 5) == 0:
                    print(f"📄 CSV Update: {combination_num}/{len(valid_combinations)} combinations")
                    if os.path.exists(output_csv):
                        with open(output_csv, 'r') as f:
                            lines = f.readlines()
                        print(f"   File contains {len(lines)} lines (including header)")
            
            print(f"✅ {symbol} - {strategy_config.strategy_name} completed! ({total_combinations} combinations)")
            print(f"   Results saved to: {output_csv}")
            return True
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Work item interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Error processing work item: {e}")
            return False
    
    def _show_results_summary(self, processed_symbols: List[str]):
        """Show summary of all result files"""
        print(f"\n📋 Generated {self.config.strategy_name} Results Files:")
        for symbol in processed_symbols:
            output_csv = f"results/{self.config.strategy_type}_optimization_{symbol}_results.csv"
            if os.path.exists(output_csv):
                with open(output_csv, 'r') as f:
                    lines = f.readlines()
                print(f"   {symbol}: {output_csv} ({len(lines)} lines)")
            else:
                print(f"   {symbol}: No results file generated")
        
        # Show summary of all existing files for this strategy
        all_existing = self.get_existing_results()
        if all_existing:
            print(f"\n📊 All Existing {self.config.strategy_name} Results:")
            for symbol, file_path in sorted(all_existing.items()):
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    file_age = datetime.fromtimestamp(os.path.getmtime(file_path))
                    print(f"   {symbol}: {len(lines)} lines (modified: {file_age.strftime('%Y-%m-%d %H:%M:%S')})")
                except (OSError, IOError):
                    print(f"   {symbol}: File exists but cannot read details")


def main():
    """Main function - analyze coverage and process work items in priority order"""
    print("🚀 INTELLIGENT MULTI-STRATEGY OPTIMIZATION")
    print("=" * 60)
    print("Analyzing symbol-strategy coverage and prioritizing work...")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Analyze current coverage and get prioritized work items
    coverage_analysis = analyze_symbol_strategy_coverage()
    work_items = coverage_analysis['work_items']
    
    if not work_items:
        print("✅ All symbols have complete and up-to-date strategy coverage!")
        return
    
    # Show coverage summary
    print(f"\n📊 COVERAGE ANALYSIS:")
    print(f"   🆕 Completely missing symbols: {len(coverage_analysis['completely_missing'])}")
    if coverage_analysis['completely_missing']:
        print(f"      {coverage_analysis['completely_missing'][:10]}{'...' if len(coverage_analysis['completely_missing']) > 10 else ''}")
    
    print(f"   🔄 Partially missing symbols: {len(coverage_analysis['partially_missing'])}")
    if coverage_analysis['partially_missing']:
        print(f"      {coverage_analysis['partially_missing'][:10]}{'...' if len(coverage_analysis['partially_missing']) > 10 else ''}")
    
    print(f"   ✅ Fully covered symbols: {len(coverage_analysis['fully_covered'])}")
    if coverage_analysis['fully_covered']:
        print(f"      {coverage_analysis['fully_covered'][:10]}{'...' if len(coverage_analysis['fully_covered']) > 10 else ''}")
    
    # Show work items by priority
    priority_counts = {}
    for item in work_items:
        priority = item['priority']
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print(f"\n📋 WORK ITEMS BY PRIORITY:")
    priority_names = {1: "Missing Symbols", 2: "Missing Strategies", 3: "Refresh Old Results"}
    for priority in sorted(priority_counts.keys()):
        count = priority_counts[priority]
        name = priority_names.get(priority, f"Priority {priority}")
        print(f"   {priority}. {name}: {count} items")
    
    print(f"\n🎯 PROCESSING {len(work_items)} WORK ITEMS IN PRIORITY ORDER")
    print("=" * 60)
    
    # Process work items in priority order
    completed_items = 0
    
    # Create a dummy optimizer for the first strategy (we'll switch as needed)
    first_strategy = list(STRATEGY_CONFIGS.keys())[0]
    optimizer = GenericStrategyOptimizer(STRATEGY_CONFIGS[first_strategy])
    
    try:
        for item_num, work_item in enumerate(work_items, 1):
            print(f"\n📍 Work Item {item_num}/{len(work_items)} (Priority {work_item['priority']})")
            print(f"   {work_item['symbol']} - {work_item['strategy']} - {work_item['reason']}")
            
            success = optimizer.optimize_single_work_item(work_item, test_scenarios=100, scenario_length=352)
            
            if success:
                completed_items += 1
                print(f"✅ Work item completed successfully")
            else:
                print(f"❌ Work item failed")
            
            print(f"📊 Progress: {completed_items}/{item_num} successful, {len(work_items) - item_num} remaining")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Processing stopped by user after {completed_items} completed items")
    
    print(f"\n✅ OPTIMIZATION SESSION COMPLETED!")
    print(f"   Work items completed: {completed_items}/{len(work_items)}")
    print(f"   Success rate: {completed_items/len(work_items)*100:.1f}%" if work_items else "100%")
    
    # Show final coverage summary
    print(f"\n📋 FINAL RESULTS SUMMARY:")
    all_results = get_all_existing_results()
    for strategy_key, config in STRATEGY_CONFIGS.items():
        strategy_type = config.strategy_type
        strategy_results = all_results.get(strategy_type, {})
        coverage = len(strategy_results)
        print(f"   {config.strategy_name}: {coverage}/{len(SYMBOLS)} symbols ({coverage/len(SYMBOLS)*100:.1f}%)")
        
        if coverage > 0:
            # Show sample of recent files
            sample_files = list(strategy_results.items())[:5]
            for symbol, file_path in sample_files:
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    file_age = datetime.fromtimestamp(os.path.getmtime(file_path))
                    print(f"      {symbol}: {len(lines)} lines (modified: {file_age.strftime('%Y-%m-%d %H:%M:%S')})")
                except (OSError, IOError):
                    print(f"      {symbol}: File exists but cannot read details")


if __name__ == "__main__":
    main()
