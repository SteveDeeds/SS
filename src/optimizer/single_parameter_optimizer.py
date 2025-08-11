"""
Single Parameter Optimizer

A streamlined optimizer for testing single parameter combinations efficiently.
Loads data once per symbol and generates multiple scenarios for robust testing.
"""

import sys
import os
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.augmentation import DataAugmentationEngine
from simulator.engine import TradingSimulator
from optimizer.strategy_adapter import StrategyAdapter


class SingleParameterOptimizer:
    """
    Efficient optimizer for single parameter combinations.
    
    Unlike GridSearchOptimizer, this loads data once and tests a single
    parameter combination across multiple augmented scenarios.
    """
    
    def __init__(self, strategy_file_path: str):
        """
        Initialize optimizer with strategy file.
        
        Args:
            strategy_file_path: Path to strategy Python file
        """
        self.strategy_file_path = strategy_file_path
        self.strategy_class = self._load_strategy_class()
        self.simulator = TradingSimulator(commission_per_trade=1.0, slippage_pct=0.001)
        self.augmentation_engine = DataAugmentationEngine()
    
    def _load_strategy_class(self):
        """Load strategy class from file"""
        # Get absolute path
        abs_path = os.path.abspath(self.strategy_file_path)
        
        # Extract module name and add to sys.path
        strategy_dir = os.path.dirname(abs_path)
        strategy_filename = os.path.basename(abs_path)
        strategy_module_name = strategy_filename[:-3]  # Remove .py extension
        
        if strategy_dir not in sys.path:
            sys.path.insert(0, strategy_dir)
        
        # Import the module
        strategy_module = __import__(strategy_module_name)
        
        # Get strategy class from STRATEGY_INFO
        if hasattr(strategy_module, 'STRATEGY_INFO'):
            return strategy_module.STRATEGY_INFO['strategy_class']
        else:
            # Fallback: look for class that inherits from OptimizableStrategy
            for attr_name in dir(strategy_module):
                attr = getattr(strategy_module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, 'parameters') and 
                    attr_name != 'OptimizableStrategy'):
                    return attr
        
        raise ValueError(f"Could not find strategy class in {self.strategy_file_path}")
    
    def test_single_parameters(self, 
                              historical_data: List[Dict],
                              parameters: Dict[str, Any],
                              symbol: str,
                              initial_capital: float = 10000,
                              test_scenarios: int = 100,
                              scenario_length: int = 352,
                              warmup_days: int = 100) -> Dict[str, Any]:
        """
        Test a single set of parameters across multiple scenarios.
        
        Args:
            historical_data: Historical OHLCV data (loaded once)
            parameters: Single set of strategy parameters
            symbol: Stock symbol being tested
            initial_capital: Starting capital
            test_scenarios: Number of augmented scenarios to test
            scenario_length: Length of each scenario in days
            warmup_days: Days for indicator warmup
            
        Returns:
            Dictionary with test results and statistics
        """
        # Create strategy instance with parameters
        strategy_instance = self.strategy_class(**parameters)
        
        # Wrap strategy with adapter
        composite_strategy = StrategyAdapter(strategy_instance)
        
        # Generate augmented scenarios from historical data
        scenarios = self.augmentation_engine.generate_augmented_data(
            historical_data,
            n_scenarios=test_scenarios,
            scenario_length=scenario_length,
            market_regime='all'
        )
        
        # Run simulation on each scenario
        scenario_results = []
        total_returns = []
        
        for scenario_idx, scenario_data in enumerate(scenarios):
            try:
                # Run single scenario simulation
                result = self.simulator.run_single_scenario(
                    composite_strategy=composite_strategy,
                    price_history=scenario_data,
                    initial_capital=initial_capital,
                    symbol=symbol,
                    warmup_days=warmup_days
                )
                
                scenario_results.append(result)
                total_returns.append(result.total_return)
                
            except Exception as e:
                print(f"⚠️ Scenario {scenario_idx + 1} failed: {e}")
                # Add failed scenario with zero return
                total_returns.append(0.0)
        
        # Calculate statistics
        if total_returns:
            average_return = sum(total_returns) / len(total_returns)
            positive_returns = [r for r in total_returns if r > 0]
            win_rate = len(positive_returns) / len(total_returns) if total_returns else 0
            
            # Calculate additional metrics
            max_return = max(total_returns) if total_returns else 0
            min_return = min(total_returns) if total_returns else 0
            
            # Calculate return on average invested capital (ROAIC)
            total_trades = sum(len(result.trades_executed) for result in scenario_results if result)
            roaic = average_return if average_return != 0 else None
        else:
            average_return = 0
            win_rate = 0
            max_return = 0
            min_return = 0
            total_trades = 0
            roaic = None
        
        # Return comprehensive results
        return {
            'symbol': symbol,
            'parameters': parameters,
            'strategy_name': strategy_instance.name,
            'scenarios_tested': len(scenarios),
            'scenarios_successful': len(scenario_results),
            'average_return': average_return,
            'win_rate': win_rate,
            'max_return': max_return,
            'min_return': min_return,
            'total_trades': total_trades,
            'roaic': roaic,
            'scenario_results': scenario_results,
            'all_returns': total_returns,
            'scenario_data': scenarios  # Store the actual scenario data used
        }
    
    def format_results_for_csv(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results for CSV export (matching GridSearchOptimizer format).
        
        Args:
            results: Results from test_single_parameters
            
        Returns:
            Dictionary formatted for CSV export
        """
        return {
            'symbol': results['symbol'],
            'strategy_name': results['strategy_name'],
            'parameter_combination': str(results['parameters']),
            'average_return': results['average_return'],
            'win_rate': results['win_rate'],
            'max_return': results['max_return'],
            'min_return': results['min_return'],
            'total_trades': results['total_trades'],
            'return_on_avg_invested_capital': results['roaic'] or 0,
            'sharpe_ratio': 0,  # Not calculated in simple version
            'scenarios_tested': results['scenarios_tested'],
            'scenarios_successful': results['scenarios_successful']
        }
    
    def export_results_to_csv(self, results: Dict[str, Any], output_file: str, append_mode: bool = True):
        """
        Export results to CSV file.
        
        Args:
            results: Results from test_single_parameters
            output_file: Output CSV file path
            append_mode: Whether to append to existing file
        """
        csv_data = self.format_results_for_csv(results)
        
        # Determine write mode
        mode = 'a' if append_mode and os.path.exists(output_file) else 'w'
        write_header = mode == 'w' or not os.path.exists(output_file)
        
        # Write to CSV
        with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'symbol', 'strategy_name', 'parameter_combination',
                'average_return', 'win_rate', 'max_return', 'min_return', 
                'total_trades', 'return_on_avg_invested_capital', 'sharpe_ratio',
                'scenarios_tested', 'scenarios_successful'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            writer.writerow(csv_data)
