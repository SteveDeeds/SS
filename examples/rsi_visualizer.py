#!/usr/bin/env python3
"""
RSI Strategy Visualization Demo

This demo shows how to visualize RSI trading results using the
EXACT SAME scenario data that was used in the simulation.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.grid_search import GridSearchOptimizer
from src.utils.scenario_visualizer import ScenarioVisualizer


def demo_rsi_visualization():
    """Demonstrate RSI strategy visualization using scenario data"""
    print("🎨 RSI STRATEGY VISUALIZATION DEMO")
    print("=" * 60)
    print("This demo shows how to visualize RSI trading results using the")
    print("EXACT SAME scenario data that was used in the simulation.")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run optimization to get results with scenario data
    print("🚀 Running RSI strategy optimization...")
    optimizer = GridSearchOptimizer("strategies/rsi_strategy.py")
    results = optimizer.optimize(
        symbol="SPXL",
        parameter_grid={
            'rsi_period': [14], 
            'oversold_threshold': [30.0], 
            'overbought_threshold': [70.0], 
            'cash_percentage': [0.15]
        },
        initial_capital=10000,
        test_scenarios=1,  # Just one scenario for demo
        export_csv=False
    )
    
    print(f"✅ RSI optimization completed")
    
    # Create visualizer and generate chart
    print(f"\n🎨 Creating RSI visualization...")
    visualizer = ScenarioVisualizer()
    
    try:
        fig = visualizer.visualize_scenario_result(
            results=results,
            scenario_index=0,  # First (and only) scenario
            save_path="results/rsi_strategy_visualization.png"
        )
        
        # Show the chart
        plt.show()
        
    except Exception as e:
        print(f"❌ RSI visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        demo_rsi_visualization()
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
