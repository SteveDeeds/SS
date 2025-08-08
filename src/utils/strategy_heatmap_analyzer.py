#!/usr/bin/env python3
"""
Strategy Heat Map Analyzer

Utility for generating sequential parameter heat maps from optimal strategy results.
Creates heat maps for consecutive parameter pairs to visualize the optimization space.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.heatmap_generator import HeatMapGenerator


class StrategyHeatMapAnalyzer:
    """
    Analyzes optimal strategy results and generates sequential parameter heat maps.
    """
    
    def __init__(self, output_folder: str = "analysis_output"):
        """
        Initialize the heat map analyzer.
        
        Args:
            output_folder: Folder to save generated heat maps
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.heatmap_generator = HeatMapGenerator()
        
    def get_parameter_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Extract sequential parameter pairs from the DataFrame columns.
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            List of (param1, param2) tuples for consecutive parameters
        """
        # Find all parameter columns
        param_columns = [col for col in df.columns if col.startswith('param_')]
        
        if len(param_columns) < 2:
            print(f"⚠️ Only {len(param_columns)} parameter columns found. Need at least 2 for heat maps.")
            return []
        
        # Create sequential pairs
        param_pairs = []
        for i in range(len(param_columns) - 1):
            param1 = param_columns[i].replace('param_', '')
            param2 = param_columns[i + 1].replace('param_', '')
            param_pairs.append((param1, param2))
        
        print(f"📊 Found {len(param_columns)} parameters: {[col.replace('param_', '') for col in param_columns]}")
        print(f"🔗 Will generate {len(param_pairs)} sequential parameter pair heat maps")
        
        return param_pairs
    
    def generate_strategy_heatmaps(self, 
                                 df: pd.DataFrame,
                                 strategy_info: Dict,
                                 optimal_params: Dict = None,
                                 metrics: List[str] = None,
                                 save_plots: bool = True,
                                 show_plots: bool = False) -> Dict[str, List[str]]:
        """
        Generate heat maps for the optimal strategy results.
        
        Args:
            df: DataFrame with ALL optimization results (not filtered)
            strategy_info: Dictionary with strategy metadata (symbol, strategy_name, etc.)
            optimal_params: Dictionary with optimal strategy parameters to highlight
            metrics: List of metrics to visualize (default: key performance metrics)
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots (default: False to avoid too many popups)
            
        Returns:
            Dictionary mapping parameter pairs to list of generated file paths
        """
        if df.empty:
            print("❌ No data provided for heat map generation")
            return {}
        
        # Default metrics if not specified - focus on key performance indicators
        if metrics is None:
            metrics = ['return_on_avg_invested_capital', 'percentile_25th_return']
        
        # Filter to only available metrics
        available_metrics = [metric for metric in metrics if metric in df.columns]
        if not available_metrics:
            print("❌ No suitable metrics found for visualization")
            return {}
        
        # Get parameter pairs first 
        param_pairs = self.get_parameter_pairs(df)
        
        if not param_pairs:
            return {}
        
        # Create strategy-specific folder
        strategy_folder_name = f"{strategy_info.get('symbol', 'unknown')}_{strategy_info.get('strategy_name', 'strategy')}"
        
        # Clean folder name to remove invalid characters for Windows
        import re
        strategy_folder_name = re.sub(r'[<>:"/\\|?*]', '_', strategy_folder_name)  # Replace invalid chars
        strategy_folder_name = re.sub(r'[(),\s]+', '_', strategy_folder_name)      # Replace spaces, parens, commas
        strategy_folder_name = re.sub(r'_+', '_', strategy_folder_name)            # Collapse multiple underscores
        strategy_folder_name = strategy_folder_name.strip('_')                     # Remove leading/trailing underscores
        
        strategy_output_folder = self.output_folder / strategy_folder_name
        strategy_output_folder.mkdir(exist_ok=True)
        
        print(f"\n🎨 GENERATING HEAT MAPS")
        print("=" * 60)
        print(f"📈 Strategy: {strategy_info.get('strategy_name', 'Unknown')}")
        print(f"🎯 Symbol: {strategy_info.get('symbol', 'Unknown')}")
        print(f"📊 Total data points: {len(df)}")
        print(f"📋 Metrics: {', '.join(available_metrics)}")
        print(f"📁 Output folder: {strategy_folder_name}")
        if optimal_params:
            print(f"🎯 Optimal params: {optimal_params}")
            print(f"🔧 Will filter data to match optimal parameter values for parameter slices")
        
        generated_files = {}
        
        # Generate heat maps for each parameter pair and metric combination
        for i, (param1, param2) in enumerate(param_pairs, 1):
            print(f"\n🔍 Parameter Pair {i}/{len(param_pairs)}: {param1} vs {param2}")
            
            pair_key = f"{param1}_vs_{param2}"
            generated_files[pair_key] = []
            
            # Filter data to create a slice where other parameters match the optimal strategy
            filtered_df = df.copy()
            if optimal_params:
                # Get all parameter columns except the two we're plotting
                param_columns = [col for col in df.columns if col.startswith('param_')]
                # Convert param1 and param2 back to column names for comparison
                param1_col = f'param_{param1}'
                param2_col = f'param_{param2}'
                other_params = [col for col in param_columns if col not in [param1_col, param2_col]]
                
                print(f"   🔧 Filtering data: fixing {len(other_params)} other parameters (with ±1 neighbors)")
                for param in other_params:
                    # Convert column name to parameter name (remove 'param_' prefix)
                    param_name = param.replace('param_', '')
                    if param_name in optimal_params:
                        if pd.notna(optimal_params[param_name]):
                            opt_value = optimal_params[param_name]
                            
                            # Get all unique values for this parameter and sort them
                            param_values = sorted(filtered_df[param].dropna().unique())
                            
                            # Find the position of the optimal value
                            try:
                                opt_index = param_values.index(opt_value)
                            except ValueError:
                                # If exact value not found, find closest
                                closest_idx = min(range(len(param_values)), 
                                                 key=lambda i: abs(param_values[i] - opt_value))
                                opt_index = closest_idx
                            
                            # Get neighboring values (previous, current, next)
                            values_to_include = []
                            
                            # Add previous value if exists
                            if opt_index > 0:
                                values_to_include.append(param_values[opt_index - 1])
                            
                            # Add current optimal value
                            values_to_include.append(param_values[opt_index])
                            
                            # Add next value if exists
                            if opt_index < len(param_values) - 1:
                                values_to_include.append(param_values[opt_index + 1])
                            
                            initial_count = len(filtered_df)
                            filtered_df = filtered_df[filtered_df[param].isin(values_to_include)]
                            final_count = len(filtered_df)
                            
                            print(f"      📌 {param} = {opt_value} ±1 ({values_to_include}) → {final_count} rows (filtered out {initial_count - final_count})")
                        else:
                            print(f"      ⏭️  {param}: skipped (NaN value)")
                    else:
                        print(f"      ⏭️  {param}: skipped (not in optimal_params)")
                
                print(f"   📊 Using {len(filtered_df)} data points (filtered from {len(df)} total)")
            else:
                print(f"   📊 Using all {len(filtered_df)} data points (no optimal parameters provided)")
            
            # Load the filtered data into heat map generator for this parameter pair
            self.heatmap_generator.load_data(filtered_df)
            
            for metric in available_metrics:
                try:
                    # Generate the heat map with average aggregation
                    fig, ax = self.heatmap_generator.generate_heatmap(
                        x_param=param1,
                        y_param=param2,
                        metric=metric,
                        colormap='RdYlGn',
                        figsize=(10, 8),
                        title=f"{strategy_info.get('symbol', '')} - {metric.replace('_', ' ').title()}\n{param1.title()} vs {param2.title()}",
                        aggfunc='mean'  # Explicitly use mean for averaging
                    )
                    
                    # Add box around optimal strategy cell if parameters provided
                    if optimal_params and param1 in optimal_params and param2 in optimal_params:
                        self._add_optimal_strategy_box(ax, param1, param2, optimal_params)
                    
                    # Save plot if requested
                    if save_plots:
                        filename = f"{param1}_vs_{param2}_{metric}.png"
                        filepath = strategy_output_folder / filename
                        
                        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                        generated_files[pair_key].append(str(filepath))
                        print(f"   💾 Saved: {filename}")
                    
                    # Show plot if requested (default: False)
                    if show_plots:
                        plt.show()
                    else:
                        plt.close(fig)  # Close to free memory
                        
                except Exception as e:
                    print(f"   ❌ Error generating {param1} vs {param2} for {metric}: {e}")
                    continue
        
        return generated_files
    
    def _add_optimal_strategy_box(self, ax, param1: str, param2: str, optimal_params: Dict):
        """
        Add a box around the optimal strategy cell in the heat map.
        
        Args:
            ax: Matplotlib axes object
            param1: X-axis parameter name
            param2: Y-axis parameter name
            optimal_params: Dictionary with optimal parameter values
        """
        try:
            # Get the optimal parameter values
            opt_x_val = optimal_params.get(param1)
            opt_y_val = optimal_params.get(param2)
            
            if opt_x_val is None or opt_y_val is None:
                return
            
            # Convert numpy types to native Python types for comparison
            opt_x_val = float(opt_x_val)
            opt_y_val = float(opt_y_val)
            
            # Get the actual data from the heat map to find the correct positions
            # The heat map is created from a pivot table, so we need to get the index/column values
            
            # Get the tick positions and labels
            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()
            x_labels = [float(label.get_text()) for label in ax.get_xticklabels() if label.get_text()]
            y_labels = [float(label.get_text()) for label in ax.get_yticklabels() if label.get_text()]
            
            # Find the position of our optimal values in the label arrays
            x_pos = None
            y_pos = None
            
            # Find closest match for x-axis
            for i, x_label in enumerate(x_labels):
                if abs(x_label - opt_x_val) < 0.001:  # Small tolerance for floating point comparison
                    x_pos = i
                    break
            
            # Find closest match for y-axis
            for i, y_label in enumerate(y_labels):
                if abs(y_label - opt_y_val) < 0.001:  # Small tolerance for floating point comparison
                    y_pos = i
                    break
            
            if x_pos is not None and y_pos is not None:
                # In heat maps, y-axis has lower values at the top (index 0)
                # and higher values at the bottom, so use y_pos directly
                
                # Add a rectangle around the cell
                # Heat map cells are centered at integer positions
                rect = plt.Rectangle((x_pos, y_pos), 1, 1, 
                                   linewidth=4, edgecolor='red', facecolor='none', 
                                   linestyle='-', alpha=0.9, zorder=10)
                ax.add_patch(rect)
                
                print(f"   🎯 Added optimal strategy box at ({param1}={opt_x_val}, {param2}={opt_y_val}) -> grid position ({x_pos}, {y_pos})")
            else:
                print(f"   ⚠️ Optimal values not found in heat map: {param1}={opt_x_val} (available: {x_labels}), {param2}={opt_y_val} (available: {y_labels})")
                
        except Exception as e:
            print(f"   ❌ Error adding optimal strategy box: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_optimal_strategy(self, 
                               optimal_strategy_row: pd.Series,
                               all_results_df: pd.DataFrame = None,
                               results_folder: str = "results",
                               metrics: List[str] = None) -> Dict[str, List[str]]:
        """
        Generate heat maps for a specific optimal strategy.
        
        Args:
            optimal_strategy_row: Series with the optimal strategy information
            all_results_df: DataFrame with all optimization results (not used - kept for compatibility)
            results_folder: Folder containing the original CSV files
            metrics: List of metrics to visualize
            
        Returns:
            Dictionary with generated file paths
        """
        symbol = optimal_strategy_row.get('symbol', 'Unknown')
        strategy_name = optimal_strategy_row.get('strategy_name', 'Unknown')
        source_file = optimal_strategy_row.get('source_file', None)
        
        print(f"\n🔬 ANALYZING OPTIMAL STRATEGY")
        print("=" * 60)
        print(f"📈 Strategy: {strategy_name}")
        print(f"🎯 Symbol: {symbol}")
        print(f"📊 ROIC: {optimal_strategy_row.get('return_on_avg_invested_capital', 'N/A'):.4f}")
        print(f"📁 Source file: {source_file}")
        
        if not source_file:
            print("❌ No source file information available")
            return {}
        
        # Load data from the specific source file
        try:
            # Get the project root and construct the full path
            project_root = Path(__file__).parent.parent.parent
            source_file_path = project_root / results_folder / source_file
            
            if not source_file_path.exists():
                print(f"❌ Source file not found: {source_file_path}")
                return {}
            
            analysis_df = pd.read_csv(source_file_path)
            print(f"🔍 Loaded {len(analysis_df)} results from {source_file} (full parameter space)")
            
        except Exception as e:
            print(f"❌ Error loading source file {source_file}: {e}")
            return {}
        
        # Extract optimal parameters from the row
        optimal_params = {}
        for col in optimal_strategy_row.index:
            if col.startswith('param_'):
                param_name = col.replace('param_', '')
                optimal_params[param_name] = optimal_strategy_row[col]
        
        print(f"🎯 Optimal parameters: {optimal_params}")
        
        if analysis_df.empty:
            print("❌ No data available for analysis")
            return {}
        
        # Prepare strategy info
        strategy_info = {
            'symbol': symbol,
            'strategy_name': strategy_name,
            'roic': optimal_strategy_row.get('return_on_avg_invested_capital', 0),
            'source_file': source_file
        }
        
        # Generate heat maps with optimal parameter highlighting
        return self.generate_strategy_heatmaps(
            df=analysis_df,
            strategy_info=strategy_info,
            optimal_params=optimal_params,  # Pass optimal parameters for highlighting
            metrics=metrics,
            save_plots=True,
            show_plots=False  # Don't show plots to avoid too many popups
        )
    
    def analyze_top_strategies(self, 
                             top_strategies_df: pd.DataFrame,
                             results_folder: str = "results",
                             max_strategies: int = 3,
                             metrics: List[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate heat maps for multiple top strategies.
        
        Args:
            top_strategies_df: DataFrame with top strategies
            results_folder: Folder containing the original CSV files
            max_strategies: Maximum number of strategies to analyze
            metrics: List of metrics to visualize (default: ['return_on_avg_invested_capital', 'percentile_25th_return'])
            
        Returns:
            Dictionary mapping strategy info to generated file paths
        """
        if top_strategies_df.empty:
            print("❌ No top strategies provided")
            return {}
        
        # Default metrics focusing on key performance indicators
        if metrics is None:
            metrics = ['return_on_avg_invested_capital', 'percentile_25th_return']
        
        print(f"\n🏆 ANALYZING TOP {min(max_strategies, len(top_strategies_df))} STRATEGIES")
        print("=" * 80)
        
        all_results = {}
        
        for idx, (_, strategy_row) in enumerate(top_strategies_df.head(max_strategies).iterrows(), 1):
            print(f"\n{'='*20} STRATEGY #{idx} {'='*20}")
            
            strategy_key = f"strategy_{idx}_{strategy_row.get('symbol', 'unknown')}"
            results = self.analyze_optimal_strategy(
                optimal_strategy_row=strategy_row,
                results_folder=results_folder,
                metrics=metrics
            )
            
            if results:
                all_results[strategy_key] = results
                print(f"✅ Generated {sum(len(files) for files in results.values())} heat map files")
            else:
                print(f"❌ No heat maps generated for strategy #{idx}")
        
        return all_results
    
    def create_summary_report(self, 
                            analysis_results: Dict[str, Dict[str, List[str]]],
                            top_strategies_df: pd.DataFrame) -> str:
        """
        Create a summary report of the heat map analysis.
        
        Args:
            analysis_results: Results from analyze_top_strategies
            top_strategies_df: DataFrame with top strategies
            
        Returns:
            Path to the summary report file
        """
        report_path = self.output_folder / "heat_map_analysis_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🎨 STRATEGY HEAT MAP ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"📅 Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📊 Strategies Analyzed: {len(analysis_results)}\n")
            f.write(f"📁 Output Folder: {self.output_folder}\n\n")
            
            # Strategy details
            for idx, (strategy_key, results) in enumerate(analysis_results.items(), 1):
                strategy_row = top_strategies_df.iloc[idx-1] if idx <= len(top_strategies_df) else None
                
                f.write(f"🏆 STRATEGY #{idx}\n")
                f.write("-" * 40 + "\n")
                
                if strategy_row is not None:
                    f.write(f"📈 Symbol: {strategy_row.get('symbol', 'Unknown')}\n")
                    f.write(f"🎯 Strategy: {strategy_row.get('strategy_name', 'Unknown')}\n")
                    f.write(f"💰 ROIC: {strategy_row.get('return_on_avg_invested_capital', 0):.4f}\n")
                    f.write(f"📊 Win Rate: {strategy_row.get('win_rate', 0):.3f}\n")
                    f.write(f"📈 Sharpe Ratio: {strategy_row.get('sharpe_ratio', 0):.3f}\n")
                
                f.write(f"🎨 Heat Maps Generated:\n")
                total_files = 0
                for param_pair, files in results.items():
                    f.write(f"   📋 {param_pair}: {len(files)} files\n")
                    total_files += len(files)
                
                f.write(f"📁 Total Files: {total_files}\n\n")
            
            # File listing
            f.write("📂 GENERATED FILES\n")
            f.write("=" * 40 + "\n")
            for strategy_key, results in analysis_results.items():
                f.write(f"\n{strategy_key}:\n")
                for param_pair, files in results.items():
                    for file_path in files:
                        f.write(f"   📄 {Path(file_path).name}\n")
        
        print(f"📋 Summary report saved: {report_path}")
        return str(report_path)


def demo_heat_map_analysis():
    """Demonstrate heat map analysis with sample data"""
    print("🎨 DEMO: Strategy Heat Map Analysis")
    print("=" * 60)
    
    # Create sample optimal strategy data
    import numpy as np
    import tempfile
    import os
    
    # Sample parameter combinations
    rsi_periods = [10, 12, 14, 16, 18, 20, 22]
    oversold_thresholds = [15, 20, 25, 30, 35, 40]
    overbought_thresholds = [60, 65, 70, 75, 80, 85, 90, 95]
    cash_percentages = [0.05, 0.1, 0.15, 0.2]
    
    data = []
    for rsi_period in rsi_periods:
        for oversold in oversold_thresholds:
            for overbought in overbought_thresholds:
                for cash_pct in cash_percentages:
                    if oversold < overbought:  # Valid combination
                        # Simulate performance metrics
                        base_return = np.random.normal(0.08, 0.12)
                        roic = np.random.normal(0.15, 0.25)
                        win_rate = np.random.uniform(0.3, 0.8)
                        sharpe = np.random.normal(0.6, 0.4)
                        percentile_25th = np.random.normal(0.02, 0.08)  # Add this metric
                        
                        data.append({
                            'symbol': 'DEMO',
                            'strategy_name': f'RSI({rsi_period}) Demo Strategy',
                            'return_on_avg_invested_capital': roic,
                            'average_return': base_return,
                            'win_rate': win_rate,
                            'sharpe_ratio': sharpe,
                            'percentile_25th_return': percentile_25th,  # Include this
                            'param_rsi_period': rsi_period,
                            'param_oversold_threshold': oversold,
                            'param_overbought_threshold': overbought,
                            'param_cash_percentage': cash_pct
                        })
    
    df = pd.DataFrame(data)
    print(f"📊 Created {len(df)} sample strategy results")
    
    # Create temporary CSV file for demo
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_csv_path = f.name
    
    try:
        # Get top strategy
        top_strategy = df.nlargest(1, 'return_on_avg_invested_capital').iloc[0]
        print(f"🏆 Top strategy: {top_strategy['strategy_name']}")
        print(f"📊 ROIC: {top_strategy['return_on_avg_invested_capital']:.4f}")
        
        # Create analyzer and generate heat maps
        analyzer = StrategyHeatMapAnalyzer(output_folder="demo_heatmaps")
        
        results = analyzer.analyze_optimal_strategy(
            optimal_strategy_row=top_strategy,
            metrics=['return_on_avg_invested_capital', 'percentile_25th_return']
        )
        
        print(f"\n✅ Demo completed! Generated heat maps in: demo_heatmaps/")
        return results
    finally:
        # Clean up temp file
        os.unlink(temp_csv_path)


if __name__ == "__main__":
    """Run demo heat map analysis"""
    try:
        demo_heat_map_analysis()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
