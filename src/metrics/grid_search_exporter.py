"""
Grid Search Results Export System

Provides standardized CSV export functionality for grid search results
that is model-independent and suitable for Excel analysis and programmatic processing.
"""
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


class GridSearchResultsExporter:
    """
    Exports grid search results to standardized CSV format
    """
    
    # Fixed columns that appear in every export (never change)
    FIXED_COLUMNS = [
        'symbol',
        'strategy_type', 
        'run_datetime',
        'test_scenarios',
        'average_return',
        'return_std_dev',
        'roaic',
        'percentile_25_return',
        'win_rate',
        'sharpe_ratio',
        'max_drawdown',
        'total_trades',
        'avg_trades_per_scenario',
        'best_case_return',
        'worst_case_return',
        'median_return'
    ]
    
    def __init__(self, symbol: str, strategy_type: str, results_dir: str = "results"):
        """
        Initialize exporter
        
        Args:
            symbol: Trading symbol (e.g., 'SPXL')
            strategy_type: Strategy type (e.g., 'MA_Crossover', 'RSI_Momentum')
            results_dir: Directory to save results files
        """
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.results_dir = results_dir
        self.datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
    
    def export_results(self, grid_results: Dict[str, Any], 
                      scenario_count: int,
                      filename_suffix: str = "",
                      scenario_results_map: Dict[str, List] = None) -> str:
        """
        Export grid search results to CSV
        
        Args:
            grid_results: Dictionary mapping strategy names to metrics
            scenario_count: Number of test scenarios used
            filename_suffix: Optional suffix for filename
            scenario_results_map: Optional dict mapping strategy names to scenario results for additional metrics
            
        Returns:
            Path to exported CSV file
        """
        # Generate filename
        filename = f"{self.symbol}_{self.strategy_type}_grid_search_{self.datetime}"
        if filename_suffix:
            filename += f"_{filename_suffix}"
        filename += ".csv"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare data for export
        export_data = []
        
        for strategy_name, metrics in grid_results.items():
            # Extract parameters from strategy name
            parameters = self._extract_parameters(strategy_name)
            
            # Get scenario results for this strategy if available
            strategy_scenario_results = scenario_results_map.get(strategy_name, []) if scenario_results_map else []
            
            # Calculate total trades across all scenarios for this strategy
            total_trades = sum([len(result.trades_executed) for result in strategy_scenario_results]) if strategy_scenario_results else 0
            
            # Create row with fixed columns - map to actual metric names
            row = {
                'symbol': self.symbol,
                'strategy_type': self.strategy_type,
                'run_datetime': self.datetime,
                'test_scenarios': scenario_count,
                'average_return': metrics.get('average_return', 0),
                'return_std_dev': metrics.get('return_standard_deviation', 0),
                'roaic': metrics.get('return_on_average_invested_capital', 0),
                'percentile_25_return': metrics.get('percentile_25th_return', 0),  # Fixed: was percentile_25_return
                'win_rate': metrics.get('win_rate', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),  # Note: Not implemented in StrategyMetrics yet
                'total_trades': total_trades,
                'avg_trades_per_scenario': metrics.get('average_trades_per_scenario', 0),
                'best_case_return': metrics.get('max_return', 0),  # Fixed: was best_case_return
                'worst_case_return': metrics.get('min_return', 0),  # Fixed: was worst_case_return
                'median_return': metrics.get('median_return', 0)
            }
            
            # Add parameter columns
            for param_name, param_value in parameters.items():
                row[f"param_{param_name}"] = param_value
            
            export_data.append(row)
        
        # Get all column names (fixed + dynamic parameters)
        all_columns = self.FIXED_COLUMNS.copy()
        
        # Find all parameter columns across all strategies
        param_columns = set()
        for row in export_data:
            param_columns.update([col for col in row.keys() if col.startswith('param_')])
        
        # Sort parameter columns for consistency
        param_columns = sorted(param_columns)
        all_columns.extend(param_columns)
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_columns)
            writer.writeheader()
            
            for row in export_data:
                # Ensure all columns are present (fill missing with None)
                complete_row = {col: row.get(col, None) for col in all_columns}
                writer.writerow(complete_row)
        
        print(f"✅ Grid search results exported to: {filepath}")
        print(f"   • Strategies tested: {len(export_data)}")
        print(f"   • Columns: {len(all_columns)} ({len(self.FIXED_COLUMNS)} fixed + {len(param_columns)} parameters)")
        
        return filepath
    
    def _extract_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """
        Extract parameters from strategy name
        
        Args:
            strategy_name: Strategy name like "MA(10,30) Crossover - Cash(0.15)"
            
        Returns:
            Dictionary of parameter names and values
        """
        parameters = {}
        
        # Extract MA parameters from names like "MA(10,30) Crossover"
        if "MA(" in strategy_name and ")" in strategy_name:
            params_str = strategy_name.split("MA(")[1].split(")")[0]
            if "," in params_str:
                parts = params_str.split(",")
                if len(parts) >= 2:
                    parameters['fast_period'] = int(parts[0])
                    parameters['slow_period'] = int(parts[1])
        
        # Extract cash percentage from strategy names (common pattern)
        # Look for patterns like "Cash(0.15)" or similar
        import re
        
        # Pattern for cash percentage in parentheses
        cash_match = re.search(r'Cash\(([0-9.]+)\)', strategy_name)
        if cash_match:
            parameters['cash_percentage'] = float(cash_match.group(1))
        
        # Pattern for percentage values like "15%" or "0.15"
        percent_matches = re.findall(r'([0-9.]+)%', strategy_name)
        if percent_matches:
            parameters['cash_percentage'] = float(percent_matches[0]) / 100
        
        # If no cash percentage found but we know it's an MA strategy,
        # try to infer from common patterns in ParameterizedStrategyFactory
        if 'cash_percentage' not in parameters and 'fast_period' in parameters:
            # This is a fallback - in practice, we should encode parameters more explicitly
            # For now, we'll leave it as unknown
            pass
        
        return parameters
    
    @staticmethod
    def load_results(filepath: str) -> pd.DataFrame:
        """
        Load grid search results from CSV for analysis
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Pandas DataFrame with results
        """
        return pd.read_csv(filepath)
    
    @staticmethod
    def find_optimal_strategies(df: pd.DataFrame, 
                               metric: str = 'roaic',
                               top_n: int = 10) -> pd.DataFrame:
        """
        Find top performing strategies by specified metric
        
        Args:
            df: DataFrame with grid search results
            metric: Metric to optimize for
            top_n: Number of top strategies to return
            
        Returns:
            DataFrame with top strategies sorted by metric
        """
        return df.nlargest(top_n, metric)
    
    @staticmethod
    def parameter_sensitivity_analysis(df: pd.DataFrame, 
                                     metric: str = 'roaic') -> Dict[str, Dict]:
        """
        Analyze parameter sensitivity
        
        Args:
            df: DataFrame with grid search results
            metric: Metric to analyze
            
        Returns:
            Dictionary with parameter sensitivity analysis
        """
        param_columns = [col for col in df.columns if col.startswith('param_')]
        sensitivity = {}
        
        for param_col in param_columns:
            param_name = param_col.replace('param_', '')
            param_analysis = df.groupby(param_col)[metric].agg(['mean', 'std', 'count'])
            sensitivity[param_name] = param_analysis.to_dict('index')
        
        return sensitivity


class HeatmapGenerator:
    """
    Generates heatmap visualizations from grid search results
    """
    
    @staticmethod
    def create_parameter_heatmap(df: pd.DataFrame, 
                               param1: str, param2: str, 
                               metric: str = 'roaic',
                               save_path: str = None) -> str:
        """
        Create heatmap showing performance across two parameters
        
        Args:
            df: DataFrame with grid search results
            param1: First parameter (x-axis)
            param2: Second parameter (y-axis)
            metric: Performance metric to visualize
            save_path: Path to save heatmap image
            
        Returns:
            Path to saved heatmap
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data for heatmap
        param1_col = f"param_{param1}"
        param2_col = f"param_{param2}"
        
        if param1_col not in df.columns or param2_col not in df.columns:
            raise ValueError(f"Parameters {param1} or {param2} not found in data")
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values=metric, 
            index=param2_col, 
            columns=param1_col, 
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn')
        plt.title(f'{metric.upper()} Heatmap: {param1} vs {param2}')
        plt.xlabel(param1.replace('_', ' ').title())
        plt.ylabel(param2.replace('_', ' ').title())
        
        # Save heatmap
        if save_path is None:
            datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"heatmap_{param1}_{param2}_{metric}_{datetime}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmap saved to: {save_path}")
        return save_path


# Example usage and demo functions
def demo_grid_search_export():
    """
    Demonstrate grid search export functionality
    """
    print("📊 Grid Search Export Demo")
    print("=" * 40)
    
    # Example grid search results
    example_results = {
        "MA(10,25) Crossover": {
            'average_return': 0.15,
            'return_standard_deviation': 0.08,
            'return_on_average_invested_capital': 0.32,
            'percentile_25_return': 0.05,
            'win_rate': 0.65,
            'sharpe_ratio': 1.2
        },
        "MA(15,30) Crossover": {
            'average_return': 0.12,
            'return_standard_deviation': 0.06,
            'return_on_average_invested_capital': 0.28,
            'percentile_25_return': 0.04,
            'win_rate': 0.62,
            'sharpe_ratio': 1.5
        }
    }
    
    # Export results
    exporter = GridSearchResultsExporter("SPXL", "MA_Crossover")
    filepath = exporter.export_results(example_results, scenario_count=50)
    
    # Load and analyze
    df = GridSearchResultsExporter.load_results(filepath)
    print(f"\n📋 Loaded data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Find optimal strategies
    top_strategies = GridSearchResultsExporter.find_optimal_strategies(df, 'roaic', 5)
    print(f"\n🏆 Top strategies by ROAIC:")
    print(top_strategies[['average_return', 'roaic', 'param_fast_period', 'param_slow_period']].to_string())


if __name__ == "__main__":
    demo_grid_search_export()
