#!/usr/bin/env python3
"""
Optimal Strategy Finder

This script analyzes optimization results from CSV files in the results folder to find
the best performing strategies based on multiple criteria.

Filtering Criteria:
1. average_return > 0
2. win_rate > 0.25 
3. sharpe_ratio >= 0.5
4. percentile_25th_return > 0
5. total_trades >= 4 * test_scenarios

Final Selection: Highest return_on_avg_invested_capital from remaining candidates
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import glob

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.strategy_heatmap_analyzer import StrategyHeatMapAnalyzer


class OptimalStrategyFinder:
    """
    Analyzes optimization results to find the best performing strategies.
    """
    
    def __init__(self, results_folder: str = "results"):
        """
        Initialize the strategy finder.
        
        Args:
            results_folder: Path to folder containing CSV result files
        """
        self.results_folder = Path(project_root) / results_folder
        self.criteria = {
            'average_return': "> 0",
            'win_rate': "> 0.25", 
            'sharpe_ratio': ">= 0.5",
            'percentile_25th_return': "> 0",
            'total_trades': ">= 4 * test_scenarios"
        }
        
    def load_all_results(self) -> pd.DataFrame:
        """
        Load and combine all CSV result files from the results folder.
        
        Returns:
            Combined DataFrame with all results
        """
        print(f"🔍 Loading results from: {self.results_folder}")
        
        # Find all CSV files in results folder
        csv_files = list(self.results_folder.glob("*_results.csv"))
        
        if not csv_files:
            print(f"❌ No CSV result files found in {self.results_folder}")
            return pd.DataFrame()
        
        print(f"📁 Found {len(csv_files)} result files:")
        
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add source file information
                df['source_file'] = csv_file.name
                print(f"   ✅ {csv_file.name}: {len(df)} rows")
                all_data.append(df)
            except Exception as e:
                print(f"   ❌ {csv_file.name}: Error loading - {e}")
        
        if not all_data:
            print(f"❌ No valid data loaded")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined dataset: {len(combined_df)} total rows")
        
        return combined_df
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filtering criteria to find qualified strategies.
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            Filtered DataFrame meeting all criteria
        """
        print(f"\n🔧 APPLYING FILTERS")
        print("=" * 50)
        
        original_count = len(df)
        initial_symbols = df['symbol'].nunique() if not df.empty else 0
        print(f"📊 Starting with {original_count} strategies across {initial_symbols} symbols")
        
        # Filter 1: average_return > 0
        df = df[df['average_return'] > 0]
        unique_symbols = df['symbol'].nunique() if not df.empty else 0
        print(f"   Filter 1 (average_return > 0): {len(df)} remaining ({original_count - len(df)} removed) - {unique_symbols} symbols")
        
        # Filter 2: win_rate > 0.25
        df = df[df['win_rate'] > 0.25]
        unique_symbols = df['symbol'].nunique() if not df.empty else 0
        print(f"   Filter 2 (win_rate > 0.25): {len(df)} remaining - {unique_symbols} symbols")
        
        # Filter 3: sharpe_ratio >= 0.5
        df = df[df['sharpe_ratio'] >= 0.5]
        unique_symbols = df['symbol'].nunique() if not df.empty else 0
        print(f"   Filter 3 (sharpe_ratio >= 0.5): {len(df)} remaining - {unique_symbols} symbols")
        
        # Filter 4: percentile_25th_return > 0
        df = df[df['percentile_25th_return'] > 0]
        unique_symbols = df['symbol'].nunique() if not df.empty else 0
        print(f"   Filter 4 (percentile_25th_return > 0): {len(df)} remaining - {unique_symbols} symbols")
        
        # Filter 5: total_trades >= 4 * test_scenarios
        df = df[df['total_trades'] >= 4 * df['test_scenarios']]
        unique_symbols = df['symbol'].nunique() if not df.empty else 0
        print(f"   Filter 5 (total_trades >= 4 * test_scenarios): {len(df)} remaining - {unique_symbols} symbols")
        
        print(f"\n✅ Final qualified strategies: {len(df)} across {unique_symbols} symbols")
        
        return df
    
    def find_optimal_strategies(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Find the top N strategies by return_on_avg_invested_capital.
        
        Args:
            df: Filtered DataFrame with qualified strategies
            top_n: Number of top strategies to return
            
        Returns:
            DataFrame with top strategies sorted by return_on_avg_invested_capital
        """
        if df.empty:
            print(f"❌ No qualified strategies found")
            return pd.DataFrame()
        
        print(f"\n🏆 FINDING OPTIMAL STRATEGIES")
        print("=" * 50)
        
        # Sort by return_on_avg_invested_capital (descending)
        top_strategies = df.nlargest(top_n, 'return_on_avg_invested_capital')
        
        print(f"📈 Top {min(top_n, len(top_strategies))} strategies by return_on_avg_invested_capital:")
        
        return top_strategies
    
    def display_strategy_summary(self, df: pd.DataFrame, detailed: bool = True):
        """
        Display a formatted summary of strategies.
        
        Args:
            df: DataFrame with strategies to display
            detailed: Whether to show detailed information
        """
        if df.empty:
            print("❌ No strategies to display")
            return
        
        print(f"\n📋 STRATEGY SUMMARY")
        print("=" * 80)
        
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            print(f"\n🏅 #{idx} - {row['symbol']}")
            print(f"   Strategy: {row['strategy_name']}")
            print(f"   📈 Return on Invested Capital: {row['return_on_avg_invested_capital']:.4f}")
            print(f"   💰 Average Return: {row['average_return']:.4f} ({row['average_return']*100:.2f}%)")
            print(f"   🎯 Win Rate: {row['win_rate']:.3f} ({row['win_rate']*100:.1f}%)")
            print(f"   📊 Sharpe Ratio: {row['sharpe_ratio']:.3f}")
            print(f"   📉 25th Percentile Return: {row['percentile_25th_return']:.4f}")
            print(f"   🔄 Total Trades: {row['total_trades']} (scenarios: {row['test_scenarios']})")
            
            if detailed:
                print(f"   📐 Std Dev: {row['return_std_dev']:.4f}")
                print(f"   🔝 Best Case: {row['best_case_return']:.4f} ({row['best_case_return']*100:.2f}%)")
                print(f"   🔻 Worst Case: {row['worst_case_return']:.4f} ({row['worst_case_return']*100:.2f}%)")
                print(f"   📊 Median Return: {row['median_return']:.4f}")
    
    def analyze_by_symbol(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group analysis by symbol and find best strategy for each.
        
        Args:
            df: DataFrame with qualified strategies
            
        Returns:
            Dictionary mapping symbol to best strategy DataFrame
        """
        print(f"\n🎯 ANALYSIS BY SYMBOL")
        print("=" * 50)
        
        symbol_results = {}
        
        if df.empty:
            print("❌ No qualified strategies for symbol analysis")
            return symbol_results
        
        symbols = df['symbol'].unique()
        print(f"📊 Analyzing {len(symbols)} symbols: {', '.join(symbols)}")
        
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol]
            if not symbol_df.empty:
                best_strategy = symbol_df.nlargest(1, 'return_on_avg_invested_capital')
                symbol_results[symbol] = best_strategy
                
                strategy_info = best_strategy.iloc[0]
                print(f"\n🏆 {symbol}: {strategy_info['strategy_name']}")
                print(f"   📈 ROIC: {strategy_info['return_on_avg_invested_capital']:.4f}")
                print(f"   🎯 Win Rate: {strategy_info['win_rate']:.3f}")
                print(f"   📊 Sharpe: {strategy_info['sharpe_ratio']:.3f}")
            else:
                print(f"\n❌ {symbol}: No qualified strategies")
        
        return symbol_results
    
    def export_results(self, df: pd.DataFrame, filename: str = "optimal_strategies.csv"):
        """
        Export optimal strategies to CSV file.
        
        Args:
            df: DataFrame with optimal strategies
            filename: Output filename
        """
        if df.empty:
            print("❌ No data to export")
            return
        
        output_path = self.results_folder / filename
        
        try:
            df.to_csv(output_path, index=False)
            print(f"\n💾 Results exported to: {output_path}")
            print(f"   📁 {len(df)} strategies saved")
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
    
    def run_analysis(self, top_n: int = 10, export: bool = True, detailed: bool = True, generate_heatmaps: bool = True):
        """
        Run complete analysis workflow.
        
        Args:
            top_n: Number of top strategies to find
            export: Whether to export results to CSV
            detailed: Whether to show detailed information
            generate_heatmaps: Whether to generate parameter space heat maps
        """
        print("🚀 OPTIMAL STRATEGY FINDER")
        print("=" * 80)
        print("Finding best strategies based on multiple performance criteria")
        print()
        
        # Load all results
        all_results = self.load_all_results()
        
        if all_results.empty:
            print("❌ No data loaded. Exiting.")
            return
        
        # Apply filters
        qualified_strategies = self.apply_filters(all_results)
        
        # Find optimal strategies
        optimal_strategies = self.find_optimal_strategies(qualified_strategies, top_n)
        
        # Display results
        self.display_strategy_summary(optimal_strategies, detailed=detailed)
        
        # Analyze by symbol (PRIMARY FOCUS - One optimal strategy per symbol)
        symbol_analysis = self.analyze_by_symbol(qualified_strategies)
        
        # Display summary of per-symbol results
        if symbol_analysis:
            best_per_symbol = pd.concat(symbol_analysis.values(), ignore_index=True)
            print(f"\n🎯 PER-SYMBOL OPTIMAL STRATEGIES SUMMARY")
            print("=" * 60)
            print(f"✅ Found optimal strategies for {len(symbol_analysis)} symbols")
            print(f"📊 Average ROIC: {best_per_symbol['return_on_avg_invested_capital'].mean():.4f}")
            print(f"🎯 Average Win Rate: {best_per_symbol['win_rate'].mean():.3f}")
            print(f"📈 Average Sharpe: {best_per_symbol['sharpe_ratio'].mean():.3f}")
        
        # Export results (DISABLED - no saving)
        if export and not optimal_strategies.empty:
            print(f"\n💾 EXPORT DISABLED - Results not saved to CSV files")
            # Commented out to disable saving
            # self.export_results(optimal_strategies, "top_optimal_strategies.csv")
            # 
            # # Also export best strategy per symbol
            # if symbol_analysis:
            #     best_per_symbol = pd.concat(symbol_analysis.values(), ignore_index=True)
            #     self.export_results(best_per_symbol, "best_strategy_per_symbol.csv")
        
        # Generate heat maps for optimal strategies
        if generate_heatmaps and not optimal_strategies.empty:
            print(f"\n🎨 GENERATING PARAMETER SPACE HEAT MAPS")
            print("=" * 80)
            
            try:
                # Create heat map analyzer
                heatmap_analyzer = StrategyHeatMapAnalyzer(output_folder="strategy_heatmaps")
                
                # Generate heat maps for best strategy per symbol instead of top overall strategies
                if symbol_analysis:
                    print(f"🎯 GENERATING HEAT MAPS FOR BEST STRATEGY PER SYMBOL")
                    print("=" * 60)
                    
                    best_per_symbol = pd.concat(symbol_analysis.values(), ignore_index=True)
                    print(f"📊 Analyzing {len(best_per_symbol)} symbol-specific optimal strategies")
                    
                    heatmap_results = heatmap_analyzer.analyze_top_strategies(
                        top_strategies_df=best_per_symbol,
                        results_folder="results",
                        max_strategies=len(best_per_symbol),  # Analyze all symbol strategies
                        metrics=['return_on_avg_invested_capital', 'percentile_25th_return']
                    )
                else:
                    # Fallback to top overall strategies if no symbol analysis available
                    print(f"🏆 GENERATING HEAT MAPS FOR TOP {min(3, len(optimal_strategies))} OVERALL STRATEGIES")
                    print("=" * 60)
                    
                    max_heatmap_strategies = min(3, len(optimal_strategies))
                    
                    heatmap_results = heatmap_analyzer.analyze_top_strategies(
                        top_strategies_df=optimal_strategies,
                        results_folder="results",
                        max_strategies=max_heatmap_strategies,
                        metrics=['return_on_avg_invested_capital', 'percentile_25th_return']
                    )
                
                if heatmap_results:
                    # Create summary report using appropriate DataFrame
                    strategies_for_report = (best_per_symbol if symbol_analysis 
                                           else optimal_strategies.head(max_heatmap_strategies))
                    
                    report_path = heatmap_analyzer.create_summary_report(
                        analysis_results=heatmap_results,
                        top_strategies_df=strategies_for_report
                    )
                    
                    total_files = sum(len(files) for strategy_results in heatmap_results.values() 
                                    for files in strategy_results.values())
                    
                    analysis_type = "symbol-specific" if symbol_analysis else "top overall"
                    print(f"\n✅ Heat map analysis completed for {analysis_type} strategies!")
                    print(f"📁 Generated {total_files} heat map files in 'strategy_heatmaps/' folder")
                    print(f"📋 Summary report: {Path(report_path).name}")
                else:
                    print(f"⚠️ No heat maps generated - check data availability")
                    
            except Exception as e:
                print(f"❌ Heat map generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS")
        print("=" * 50)
        print(f"📁 Total strategies analyzed: {len(all_results)}")
        print(f"✅ Qualified strategies: {len(qualified_strategies)}")
        print(f"🏆 Top strategies shown: {len(optimal_strategies)}")
        print(f"🎯 Symbols with qualified strategies: {len(symbol_analysis)}")
        
        if not qualified_strategies.empty:
            print(f"\n📈 Qualified Strategy Statistics:")
            print(f"   Return on Invested Capital: {qualified_strategies['return_on_avg_invested_capital'].mean():.4f} ± {qualified_strategies['return_on_avg_invested_capital'].std():.4f}")
            print(f"   Average Return: {qualified_strategies['average_return'].mean():.4f} ± {qualified_strategies['average_return'].std():.4f}")
            print(f"   Win Rate: {qualified_strategies['win_rate'].mean():.3f} ± {qualified_strategies['win_rate'].std():.3f}")
            print(f"   Sharpe Ratio: {qualified_strategies['sharpe_ratio'].mean():.3f} ± {qualified_strategies['sharpe_ratio'].std():.3f}")
        
        # Return results for further analysis if needed
        return {
            'all_results': all_results,
            'qualified_strategies': qualified_strategies,
            'optimal_strategies': optimal_strategies,
            'symbol_analysis': symbol_analysis
        }


def demo_basic_analysis():
    """Demonstrate basic optimal strategy finding"""
    print("🎯 BASIC OPTIMAL STRATEGY ANALYSIS")
    print("=" * 60)
    
    finder = OptimalStrategyFinder()
    finder.run_analysis(top_n=5, detailed=False, generate_heatmaps=False)


def demo_detailed_analysis():
    """Demonstrate detailed analysis with exports"""
    print("🔬 DETAILED OPTIMAL STRATEGY ANALYSIS")
    print("=" * 60)
    
    finder = OptimalStrategyFinder()
    finder.run_analysis(top_n=10, export=True, detailed=True, generate_heatmaps=False)


def demo_heatmap_analysis():
    """Demonstrate analysis with heat map generation"""
    print("🎨 HEAT MAP STRATEGY ANALYSIS")
    print("=" * 60)
    
    finder = OptimalStrategyFinder()
    results = finder.run_analysis(top_n=5, export=True, detailed=True, generate_heatmaps=False)
    
    if results and results['optimal_strategies'] is not None and not results['optimal_strategies'].empty:
        print(f"\n🎨 Heat maps generated for parameter space visualization!")
        print(f"📁 Check the 'strategy_heatmaps/' folder for generated plots")
        print(f"📊 Sequential parameter pairs analyzed based on column order")
    
    return results


def demo_custom_analysis():
    """Demonstrate custom analysis with modified criteria"""
    print("⚙️ CUSTOM CRITERIA ANALYSIS")
    print("=" * 60)
    
    # You could modify criteria here if needed
    finder = OptimalStrategyFinder()
    
    # Load and show filtering process step by step
    all_results = finder.load_all_results()
    
    if not all_results.empty:
        print(f"\n📊 Dataset Overview:")
        print(f"   Symbols: {', '.join(all_results['symbol'].unique())}")
        print(f"   Date range: {all_results['datetime'].min()} to {all_results['datetime'].max()}")
        print(f"   Strategy types: {len(all_results['strategy_file'].unique())}")
        
        # Show distribution of key metrics
        print(f"\n📈 Metric Distributions:")
        print(f"   Return on Invested Capital: {all_results['return_on_avg_invested_capital'].describe()}")
        print(f"   Win Rate: {all_results['win_rate'].describe()}")
        print(f"   Sharpe Ratio: {all_results['sharpe_ratio'].describe()}")
    
    # Run standard analysis
    finder.run_analysis(top_n=15, export=True, detailed=True)


if __name__ == "__main__":
    """Run optimal strategy analysis"""
    
    try:
        # Check if results folder exists
        results_path = Path(project_root) / "results"
        if not results_path.exists():
            print(f"❌ Results folder not found: {results_path}")
            print("   Make sure you have run optimization scripts first")
            sys.exit(1)
        
        # Run heat map analysis by default (includes detailed analysis)
        demo_heatmap_analysis()
        
        print(f"\n✅ Optimal strategy analysis completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
