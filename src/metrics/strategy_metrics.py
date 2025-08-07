"""
Centralized Strategy Performance Metrics

This module provides standardized metrics calculation for strategy evaluation.
All strategy performance analysis should use these centralized functions to
ensure consistency across the system.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class StrategyMetrics:
    """
    Centralized calculation of strategy performance metrics
    """
    
    @staticmethod
    def calculate_scenario_metrics(scenario_results: List[Any]) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a list of scenario results
        
        Args:
            scenario_results: List of ScenarioResult objects
            
        Returns:
            Dictionary with all calculated metrics
        """
        if not scenario_results:
            return {}
        
        # Extract basic data
        returns = [result.total_return for result in scenario_results]
        final_values = [result.final_portfolio_value for result in scenario_results]
        trade_counts = [len(result.trades_executed) for result in scenario_results]
        
        # Calculate invested capital metrics for each scenario
        roaic_values = []
        for result in scenario_results:
            roaic = StrategyMetrics._calculate_return_on_average_invested_capital(result)
            if roaic is not None:
                roaic_values.append(roaic)
        
        # Core metrics requested by user
        metrics = {
            # Primary metrics
            'average_return': np.mean(returns),
            'return_standard_deviation': np.std(returns),
            'return_on_average_invested_capital': np.mean(roaic_values) if roaic_values else None,
            'percentile_25th_return': np.percentile(returns, 25),
            
            # Additional useful metrics
            'median_return': np.median(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'worst_case_return': np.min(returns),  # Alias for min_return
            'best_case_return': np.max(returns),   # Alias for max_return
            'percentile_5th_return': np.percentile(returns, 5),
            'percentile_75th_return': np.percentile(returns, 75),
            'percentile_95th_return': np.percentile(returns, 95),
            
            # Risk metrics
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': StrategyMetrics._calculate_max_drawdown(scenario_results),
            
            # Portfolio value metrics
            'average_final_value': np.mean(final_values),
            'final_value_std': np.std(final_values),
            
            # Trading activity metrics
            'average_trades_per_scenario': np.mean(trade_counts),
            'total_scenarios': len(scenario_results),
            
            # ROAIC-specific metrics
            'roaic_standard_deviation': np.std(roaic_values) if roaic_values else None,
            'roaic_25th_percentile': np.percentile(roaic_values, 25) if roaic_values else None,
        }
        
        return metrics
    
    @staticmethod
    def _calculate_return_on_average_invested_capital(scenario_result) -> Optional[float]:
        """
        Calculate Return on Average Invested Capital (ROAIC) for a single scenario
        
        ROAIC = (Final Portfolio Value - Initial Capital) / Average Invested Capital
        
        Where Average Invested Capital is the time-weighted average of capital deployed
        in the market (not sitting in cash).
        
        Args:
            scenario_result: ScenarioResult object with daily_values and trades
            
        Returns:
            ROAIC as a decimal (e.g., 0.15 for 15%) or None if calculation not possible
        """
        if not hasattr(scenario_result, 'daily_values') or not scenario_result.daily_values:
            return None
        
        try:
            # Get initial capital from first day
            initial_capital = scenario_result.daily_values[0]['portfolio_value']
            
            # Calculate invested capital for each day
            invested_capital_daily = []
            
            for daily_data in scenario_result.daily_values:
                cash_balance = daily_data.get('cash_balance', 0)
                portfolio_value = daily_data.get('portfolio_value', 0)
                
                # Invested capital = Portfolio Value - Cash Balance
                invested_capital = portfolio_value - cash_balance
                invested_capital_daily.append(max(0, invested_capital))  # Ensure non-negative
            
            # Calculate average invested capital over the period
            average_invested_capital = np.mean(invested_capital_daily)
            
            # If no capital was ever invested, ROAIC is undefined
            if average_invested_capital <= 0:
                return None
            
            # Calculate ROAIC
            final_value = scenario_result.final_portfolio_value
            total_return_dollars = final_value - initial_capital
            roaic = total_return_dollars / average_invested_capital
            
            return roaic
            
        except (KeyError, TypeError, ZeroDivisionError) as e:
            # If any required data is missing or invalid, return None
            return None
    
    @staticmethod
    def print_metrics_summary(metrics: Dict[str, float], strategy_name: str = "Strategy") -> None:
        """
        Print a formatted summary of strategy metrics
        
        Args:
            metrics: Dictionary of calculated metrics
            strategy_name: Name of the strategy for display
        """
        print(f"\n📊 {strategy_name} Performance Metrics")
        print("=" * 60)
        
        # Core requested metrics
        print("🎯 Core Metrics:")
        print(f"   • Average Return:                {metrics.get('average_return', 0):.2%}")
        print(f"   • Return Standard Deviation:     {metrics.get('return_standard_deviation', 0):.2%}")
        if metrics.get('return_on_average_invested_capital') is not None:
            print(f"   • Return on Avg Invested Capital: {metrics.get('return_on_average_invested_capital', 0):.2%}")
        else:
            print(f"   • Return on Avg Invested Capital: N/A (insufficient data)")
        print(f"   • 25th Percentile Return:        {metrics.get('percentile_25th_return', 0):.2%}")
        
        # Risk and distribution metrics
        print(f"\n📈 Risk & Distribution:")
        print(f"   • Win Rate:                      {metrics.get('win_rate', 0):.1%}")
        print(f"   • Sharpe Ratio:                  {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   • Best Case (95th %ile):         {metrics.get('percentile_95th_return', 0):.2%}")
        print(f"   • Worst Case (5th %ile):         {metrics.get('percentile_5th_return', 0):.2%}")
        
        # Additional details
        print(f"\n📋 Additional Details:")
        print(f"   • Scenarios Tested:              {metrics.get('total_scenarios', 0)}")
        print(f"   • Average Trades per Scenario:   {metrics.get('average_trades_per_scenario', 0):.1f}")
        print(f"   • Median Return:                 {metrics.get('median_return', 0):.2%}")
        
        if metrics.get('roaic_standard_deviation') is not None:
            print(f"   • ROAIC Standard Deviation:      {metrics.get('roaic_standard_deviation', 0):.2%}")
    
    @staticmethod
    def compare_strategies(strategy_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Compare multiple strategies side by side
        
        Args:
            strategy_metrics: Dict mapping strategy names to their metrics dictionaries
        """
        if not strategy_metrics:
            return
        
        print("\n🔍 Strategy Comparison")
        print("=" * 80)
        
        # Get all strategy names
        strategy_names = list(strategy_metrics.keys())
        
        # Core metrics comparison
        print(f"{'Metric':<30} ", end="")
        for name in strategy_names:
            print(f"{name[:15]:<15} ", end="")
        print()
        print("-" * 80)
        
        # Key metrics to compare
        comparison_metrics = [
            ('Average Return', 'average_return', '.2%'),
            ('Return Std Dev', 'return_standard_deviation', '.2%'),
            ('ROAIC', 'return_on_average_invested_capital', '.2%'),
            ('25th %ile Return', 'percentile_25th_return', '.2%'),
            ('Win Rate', 'win_rate', '.1%'),
            ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
            ('Worst Case (5th)', 'percentile_5th_return', '.2%'),
            ('Best Case (95th)', 'percentile_95th_return', '.2%'),
        ]
        
        for display_name, metric_key, format_str in comparison_metrics:
            print(f"{display_name:<30} ", end="")
            for name in strategy_names:
                value = strategy_metrics[name].get(metric_key)
                if value is not None:
                    formatted_value = f"{value:{format_str}}"
                    print(f"{formatted_value:<15} ", end="")
                else:
                    print(f"{'N/A':<15} ", end="")
            print()
    
    @staticmethod
    def save_metrics_to_file(metrics: Dict[str, float], filepath: str, 
                           strategy_name: str = "Strategy") -> None:
        """
        Save metrics to a file for later analysis
        
        Args:
            metrics: Dictionary of calculated metrics
            filepath: Path to save the metrics file
            strategy_name: Name of the strategy
        """
        import json
        
        output_data = {
            'strategy_name': strategy_name,
            'calculation_datetime': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Metrics saved to: {filepath}")
    
    @staticmethod
    def _calculate_max_drawdown(scenario_results: List[Any]) -> float:
        """
        Calculate maximum drawdown across all scenarios
        
        Args:
            scenario_results: List of ScenarioResult objects
            
        Returns:
            Maximum drawdown as negative decimal (e.g., -0.15 for 15% drawdown)
        """
        max_drawdown = 0.0
        
        for result in scenario_results:
            if hasattr(result, 'daily_values') and result.daily_values:
                # Calculate drawdown for this scenario
                peak = result.daily_values[0]['portfolio_value']
                scenario_max_drawdown = 0.0
                
                for day_data in result.daily_values:
                    current_value = day_data['portfolio_value']
                    
                    # Update peak if we have a new high
                    if current_value > peak:
                        peak = current_value
                    
                    # Calculate drawdown from peak
                    drawdown = (current_value - peak) / peak
                    
                    # Track maximum drawdown (most negative)
                    if drawdown < scenario_max_drawdown:
                        scenario_max_drawdown = drawdown
                
                # Track overall maximum drawdown
                if scenario_max_drawdown < max_drawdown:
                    max_drawdown = scenario_max_drawdown
        
        return max_drawdown


# Convenience function for backward compatibility
def analyze_scenario_results(scenario_results: List[Any], strategy_name: str = "Strategy") -> Dict[str, float]:
    """
    Convenience function that calculates and prints metrics for scenario results
    
    Args:
        scenario_results: List of ScenarioResult objects
        strategy_name: Name of the strategy for display
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = StrategyMetrics.calculate_scenario_metrics(scenario_results)
    StrategyMetrics.print_metrics_summary(metrics, strategy_name)
    return metrics
