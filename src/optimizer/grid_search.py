"""
Grid Search Optimizer

Implements systematic grid search over strategy parameters.
Tests all combinations and exports results to CSV for analysis.
"""
import sys
import os
from typing import Dict, Any, List, Optional
import itertools

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .base import BaseOptimizer
from data.loader import get_symbol_data
from data.augmentation import DataAugmentationEngine
from simulator.engine import TradingSimulator
from metrics.strategy_metrics import StrategyMetrics
from .strategy_adapter import StrategyAdapter


class GridSearchOptimizer(BaseOptimizer):
    """
    Grid search optimizer for systematic parameter exploration.
    
    Tests all combinations of specified parameter values and
    exports results in standardized CSV format.
    """
    
    def optimize(self, symbol: str = "SPXL", 
                 parameter_grid: Optional[Dict[str, List]] = None,
                 test_scenarios: int = 10,
                 scenario_length: int = 352,  # ~1.4 years
                 warmup_days: int = 100,
                 initial_capital: float = 10000,
                 export_csv: bool = True,
                 filename_suffix: str = "new_arch",
                 **kwargs) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Args:
            symbol: Trading symbol to test on
            parameter_grid: Dict mapping parameter names to value lists
            test_scenarios: Number of augmented scenarios to test
            scenario_length: Length of each test scenario
            warmup_days: Days for strategy warmup
            initial_capital: Starting capital for backtests
            export_csv: Whether to export results to CSV
            filename_suffix: Suffix for result files
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
        """
        print(f"🚀 Grid Search Optimization: {self.strategy_info['name']}")
        print("=" * 60)
        
        # Use default parameter grid if none provided
        if parameter_grid is None:
            parameter_grid = self._create_default_parameter_grid()
        
        # Validate parameter grid
        self._validate_parameter_grid(parameter_grid)
        
        # Get historical data
        # print(f"\n📊 Step 1: Loading data for {symbol}...")
        historical_data = get_symbol_data(symbol, period="2y")
        if not historical_data:
            raise ValueError(f"Failed to load data for {symbol}")
        print(f"✅ Loaded {len(historical_data)} days of data")
        
        # Generate test scenarios
        # print(f"\n🎲 Step 2: Generating {test_scenarios} test scenarios...")
        augmentation_engine = DataAugmentationEngine()
        scenarios = augmentation_engine.generate_augmented_data(
            historical_data,
            n_scenarios=test_scenarios,
            scenario_length=scenario_length,
            market_regime='all'
        )
        print(f"✅ Generated {len(scenarios)} scenarios")
        
        # Generate parameter combinations
        # print(f"\n🎯 Step 3: Generating parameter combinations...")
        combinations = self.get_parameter_combinations(parameter_grid)
        # print(f"🎯   Parameter grid:")
        # for param, values in parameter_grid.items():
        #     print(f"     {param}: {values}")
        # print(f"✅ Total combinations: {len(combinations)}")
        
        # Run grid search
        # print(f"\n🧪 Step 4: Testing strategies...")
        results = self._run_grid_search(
            combinations, scenarios, initial_capital, warmup_days, symbol
        )
        
        # Add symbol to results for CSV export
        results['symbol'] = symbol
        results['scenarios'] = scenarios  # Store scenarios for visualization access
        
        # Export results
        if export_csv:
            print(f"\n💾 Step 5: Exporting results...")
            csv_path = self._export_results(
                results, symbol, len(scenarios), filename_suffix
            )
            results['csv_path'] = csv_path
        
        # Summary
        print(f"\n🎉 Grid Search Complete!")
        print(f"   Strategies tested: {len(results['strategy_results'])}")
        
        # Handle case where ROAIC might be None
        best_roaic = results['best_strategy']['roaic']
        if best_roaic is not None:
            print(f"   Best ROAIC: {best_roaic:.3f}")
        else:
            print(f"   Best ROAIC: None (no valid trades)")
        print(f"   Best strategy: {results['best_strategy']['name']}")
        
        return results
    
    def _create_default_parameter_grid(self) -> Dict[str, List]:
        """Create default parameter grid based on strategy ranges"""
        grid = {}
        
        for param_name, param_range in self.parameter_ranges.items():
            min_val, max_val = param_range
            param_config = self.strategy_info['parameters'][param_name]
            param_type = param_config['type']
            
            if param_type == int:
                # Create 4-5 integer values across range
                step = max(1, (max_val - min_val) // 4)
                values = list(range(min_val, max_val + 1, step))
                if max_val not in values:
                    values.append(max_val)
                grid[param_name] = values
            
            elif param_type == float:
                # Create 4-5 float values across range
                step = (max_val - min_val) / 4
                values = [round(min_val + i * step, 3) for i in range(5)]
                values[-1] = max_val  # Ensure max is included
                grid[param_name] = values
        
        return grid
    
    def _validate_parameter_grid(self, parameter_grid: Dict[str, List]) -> None:
        """Validate that parameter grid is compatible with strategy"""
        for param_name in parameter_grid.keys():
            if param_name not in self.parameter_ranges:
                raise ValueError(f"Parameter '{param_name}' not found in strategy parameters")
        
        # Check that all values are within valid ranges
        for param_name, values in parameter_grid.items():
            param_range = self.parameter_ranges[param_name]
            min_val, max_val = param_range
            
            for value in values:
                if value < min_val or value > max_val:
                    raise ValueError(f"Parameter '{param_name}' value {value} outside valid range {param_range}")
    
    def _run_grid_search(self, combinations: List[Dict[str, Any]], 
                        scenarios: List[List[Dict]], initial_capital: float,
                        warmup_days: int, symbol: str) -> Dict[str, Any]:
        """Run grid search across all parameter combinations"""
        simulator = TradingSimulator()
        strategy_results = {}
        scenario_results_map = {}
        parameter_mapping = {}  # Track which strategy corresponds to which parameters
        
        for i, params in enumerate(combinations):
            print(f"   Testing {i+1}/{len(combinations)}: {params}")
            
            # Create strategy instance
            strategy = self.create_strategy_instance(**params)
            strategy_name = str(strategy)
            
            # Store parameter mapping for CSV export
            parameter_mapping[strategy_name] = params
            
            # Wrap strategy with adapter for simulator compatibility
            adapted_strategy = StrategyAdapter(strategy)
            
            # Test on all scenarios
            scenario_results = []
            for scenario in scenarios:
                result = simulator.run_single_scenario(
                    composite_strategy=adapted_strategy, 
                    price_history=scenario, 
                    initial_capital=initial_capital, 
                    symbol=symbol,
                    warmup_days=warmup_days
                )
                scenario_results.append(result)
            
            # Calculate metrics
            metrics = StrategyMetrics.calculate_scenario_metrics(scenario_results)
            strategy_results[strategy_name] = metrics
            scenario_results_map[strategy_name] = scenario_results
        
        # Find best strategy (handle None values)
        valid_strategies = {
            name: metrics for name, metrics in strategy_results.items()
            if metrics.get('return_on_average_invested_capital') is not None
        }
        
        if valid_strategies:
            best_strategy_name = max(
                valid_strategies.keys(), 
                key=lambda name: valid_strategies[name]['return_on_average_invested_capital']
            )
        else:
            # No valid strategies, just pick the first one
            best_strategy_name = list(strategy_results.keys())[0] if strategy_results else 'None'
        
        return {
            'strategy_results': strategy_results,
            'scenario_results_map': scenario_results_map,
            'best_strategy': {
                'name': best_strategy_name,
                'roaic': strategy_results[best_strategy_name]['return_on_average_invested_capital'],
                'metrics': strategy_results[best_strategy_name]
            },
            'parameter_grid': combinations,
            'parameter_mapping': parameter_mapping  # Add this for CSV export
        }
    
    def _export_results(self, results: Dict[str, Any], symbol: str, 
                       scenario_count: int, filename_suffix: str) -> str:
        """Export results to CSV using existing exporter"""
        from metrics.grid_search_exporter import GridSearchResultsExporter
        
        exporter = GridSearchResultsExporter(
            symbol, 
            self.strategy_info['strategy_type'], 
            results_dir="../results"
        )
        
        return exporter.export_results(
            results['strategy_results'],
            scenario_count,
            filename_suffix,
            results['scenario_results_map']
        )
    
    def export_to_csv(self, results: Dict[str, Any], output_file: str, 
                      append_mode: bool = True) -> str:
        """
        Export optimization results to CSV file.
        
        For continuous runs, this method will append new rows to existing CSV
        files, allowing long-running optimizations to continuously update output.
        
        Args:
            results: Results dictionary from optimize() method
            output_file: Path to output CSV file
            append_mode: If True, append to existing file; if False, overwrite
            
        Returns:
            Path to the CSV file
        """
        import csv
        from datetime import datetime
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else "results", exist_ok=True)
        
        # Prepare CSV headers (standardized format)
        headers = [
            'datetime',
            'strategy_file', 
            'strategy_name',
            'symbol',
            'test_scenarios',
            'parameter_combination',
            'average_return',
            'return_std_dev',
            'return_on_avg_invested_capital',
            'win_rate',
            'sharpe_ratio',
            'percentile_25th_return',
            'percentile_5th_return',
            'best_case_return',
            'worst_case_return',
            'median_return',
            'total_trades'
        ]
        
        # Add dynamic parameter columns
        if results.get('parameter_grid'):
            # Get all parameter names from the first combination
            first_combo = results['parameter_grid'][0] if results['parameter_grid'] else {}
            for param_name in first_combo.keys():
                headers.append(f'param_{param_name}')
        
        # Check if file exists and has correct headers
        file_exists = os.path.exists(output_file)
        write_header = not file_exists or not append_mode
        
        # If file exists and we're appending, verify headers match
        if file_exists and append_mode:
            try:
                with open(output_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    existing_headers = next(reader, [])
                    if existing_headers != headers:
                        print(f"⚠️ Header mismatch detected. Existing: {len(existing_headers)} cols, New: {len(headers)} cols")
                        # Continue anyway - CSV readers can handle extra/missing columns
            except:
                write_header = True  # If we can't read the file, write header
        
        # Open file in appropriate mode
        mode = 'a' if (file_exists and append_mode) else 'w'
        
        datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows_written = 0
        
        with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Write header if needed
            if write_header:
                writer.writeheader()
            
            # Write data rows - one row per parameter combination tested
            strategy_results = results.get('strategy_results', {})
            parameter_mapping = results.get('parameter_mapping', {})
            scenario_results_map = results.get('scenario_results_map', {})
            
            for strategy_name, metrics in strategy_results.items():
                # Get the actual parameter combination for this strategy
                param_combo = parameter_mapping.get(strategy_name, {})
                param_combo_str = str(param_combo) if param_combo else "unknown"
                
                # Get scenario results for trade counting
                scenario_results = scenario_results_map.get(strategy_name, [])
                total_trades = sum(len(getattr(result, 'trades_executed', [])) for result in scenario_results)
                
                # Create row data
                row_data = {
                    'datetime': datetime,
                    'strategy_file': os.path.basename(self.strategy_path),
                    'strategy_name': strategy_name,
                    'symbol': results.get('symbol', 'UNKNOWN'),
                    'test_scenarios': len(scenario_results),
                    'parameter_combination': param_combo_str,
                    'average_return': metrics.get('average_return', 0),
                    'return_std_dev': metrics.get('return_standard_deviation', 0),
                    'return_on_avg_invested_capital': metrics.get('return_on_average_invested_capital', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'percentile_25th_return': metrics.get('percentile_25th_return', 0),
                    'percentile_5th_return': metrics.get('percentile_5th_return', 0),
                    'best_case_return': metrics.get('max_return', 0),
                    'worst_case_return': metrics.get('min_return', 0),
                    'median_return': metrics.get('median_return', 0),
                    'total_trades': total_trades
                }
                
                # Add parameter values as separate columns
                for param_name, param_value in param_combo.items():
                    row_data[f'param_{param_name}'] = param_value
                
                # Write the row
                writer.writerow(row_data)
                rows_written += 1
        
        action = "Appended" if (file_exists and append_mode) else "Created"
        print(f"✅ {action} {rows_written} rows to CSV: {output_file}")
        
        return output_file
