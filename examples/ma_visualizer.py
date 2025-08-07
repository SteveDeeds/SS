#!/usr/bin/env python3
"""
Scenario Visualization Demo

This demo shows how to visualize trading results using the
EXACT SAME scenario data that was used in the simulation.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.grid_search import GridSearchOptimizer
from src.utils.scenario_visualizer import ScenarioVisualizer


def demo_scenario_visualization():
    """Demonstrate visualization using scenario data"""
    print("🎨 SCENARIO VISUALIZATION DEMO")
    print("=" * 60)
    print("This demo shows how to visualize trading results using the")
    print("EXACT SAME scenario data that was used in the simulation.")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run optimization to get results with scenario data
    print("🚀 Running optimization to generate data...")
    optimizer = GridSearchOptimizer("strategies/adaptive_ma_crossover.py")
    results = optimizer.optimize(
        symbol="SPXL",
        parameter_grid={'slow_period': [20], 'fast_period': [10], 'cash_percentage': [0.15]},
        initial_capital=10000,
        test_scenarios=1,  # Just one scenario for demo
        export_csv=False
    )
    
    print(f"✅ Optimization completed")
    
    # Create visualizer and generate chart
    print(f"\n🎨 Creating visualization...")
    visualizer = ScenarioVisualizer()
    
    try:
        fig = visualizer.visualize_scenario_result(
            results=results,
            scenario_index=0,  # First (and only) scenario
            save_path="results/scenario_visualization.png"
        )
        
        # Show the chart
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        demo_scenario_visualization()
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
