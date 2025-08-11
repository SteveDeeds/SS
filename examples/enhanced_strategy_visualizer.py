#!/usr/bin/env python3
"""
Enhanced Strategy Visualizer

This demo shows how to visualize any strategy using the efficient SingleParameterOptimizer
and ScenarioVisualizer. It provides the same visualization capability as the original
rsi_visualizer.py but works with any strategy and uses the optimized architecture.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.single_parameter_optimizer import SingleParameterOptimizer
from src.utils.scenario_visualizer import ScenarioVisualizer
from src.data.loader import get_symbol_data


def visualize_strategy(strategy_file: str, strategy_name: str, parameters: dict, 
                      symbol: str = "SPXL", save_path: str = None):
    """
    Visualize a strategy using specific parameters on a single scenario.
    
    Args:
        strategy_file: Path to strategy file
        strategy_name: Display name for the strategy
        parameters: Dictionary of strategy parameters
        symbol: Symbol to test on
        save_path: Optional path to save the chart
    """
    print(f"🎨 {strategy_name.upper()} STRATEGY VISUALIZATION")
    print("=" * 60)
    print("This demo shows how to visualize strategy trading results using the")
    print("EXACT SAME scenario data that was used in the simulation.")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load historical data once
    print(f"📡 Loading historical data for {symbol}...")
    historical_data = get_symbol_data(symbol, period="4y", force_download=False)
    
    if not historical_data:
        print(f"❌ Failed to load data for {symbol}")
        return None
    
    print(f"✅ Loaded {len(historical_data)} days of historical data")
    
    # Create optimizer and test parameters
    print(f"🚀 Running {strategy_name} strategy optimization...")
    optimizer = SingleParameterOptimizer(strategy_file)
    
    # Test the specific parameter combination
    results = optimizer.test_single_parameters(
        historical_data=historical_data,
        parameters=parameters,
        symbol=symbol,
        initial_capital=10000,
        test_scenarios=1,  # Just one scenario for visualization
        scenario_length=352,  # ~1.4 years
        warmup_days=100
    )
    
    print(f"✅ {strategy_name} optimization completed")
    print(f"   ROAIC: {results.get('roaic', 'N/A')}")
    print(f"   Win Rate: {results.get('win_rate', 'N/A'):.1%}" if results.get('win_rate') else "   Win Rate: N/A")
    print(f"   Total Trades: {results.get('total_trades', 'N/A')}")
    
    # Create visualizer and generate chart
    print(f"\n🎨 Creating {strategy_name} visualization...")
    visualizer = ScenarioVisualizer()
    
    try:
        # Convert our results format to match what ScenarioVisualizer expects
        # ScenarioVisualizer expects:
        # - scenarios: list of scenario data (OHLCV data)
        # - scenario_results_map: dict mapping strategy name to list of ScenarioResult objects
        
        # Use the stored scenario data from the optimization results
        scenarios = results.get('scenario_data', [])
        
        # If scenario data is not available, regenerate it (fallback)
        if not scenarios:
            print("⚠️ No stored scenario data found, regenerating...")
            from src.data.augmentation import DataAugmentationEngine
            augmentation_engine = DataAugmentationEngine()
            
            scenarios = augmentation_engine.generate_augmented_data(
                historical_data,
                n_scenarios=1,
                scenario_length=352,
                market_regime='all'
            )
        else:
            print("✅ Using stored scenario data from optimization")
        
        visualization_results = {
            'scenarios': scenarios,  # List of OHLCV data lists
            'scenario_results_map': {
                results.get('strategy_name', strategy_name): results['scenario_results']  # List of ScenarioResult objects
            },
            'best_strategy': {
                'name': results.get('strategy_name', strategy_name),
                'parameters': parameters,
                'roaic': results.get('roaic', 0),
                'strategy_name': results.get('strategy_name', strategy_name)
            },
            'symbol': symbol
        }
        
        if not save_path:
            safe_strategy_name = strategy_name.lower().replace(' ', '_')
            save_path = f"results/{safe_strategy_name}_visualization_{symbol}.png"
        
        fig = visualizer.visualize_scenario_result(
            results=visualization_results,
            scenario_index=0,  # First (and only) scenario
            save_path=save_path
        )
        
        print(f"✅ Visualization saved to: {save_path}")
        
        # Show the chart
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"❌ {strategy_name} visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_rsi_visualization():
    """Demonstrate RSI strategy visualization using the new architecture"""
    parameters = {
        'rsi_period': 14,
        'oversold_threshold': 30.0,
        'overbought_threshold': 70.0,
        'cash_percentage': 0.15
    }
    
    return visualize_strategy(
        strategy_file="strategies/rsi_strategy.py",
        strategy_name="RSI",
        parameters=parameters,
        symbol="SPXL",
        save_path="results/rsi_strategy_visualization.png"
    )


def demo_bollinger_bands_visualization():
    """Demonstrate Bollinger Bands strategy visualization"""
    parameters = {
        'period': 20,
        'std_positive': 2.0,
        'std_negative': 2.0,
        'cash_percentage': 0.15
    }
    
    return visualize_strategy(
        strategy_file="strategies/bollinger_bands_strategy.py",
        strategy_name="Bollinger Bands",
        parameters=parameters,
        symbol="SPXL",
        save_path="results/bollinger_bands_visualization.png"
    )


def demo_moving_average_visualization():
    """Demonstrate Moving Average strategy visualization"""
    parameters = {
        'buy_slow_period': 30,
        'buy_fast_period': 15,
        'sell_slow_period': 30,
        'sell_fast_period': 15,
        'cash_percentage': 0.15
    }
    
    return visualize_strategy(
        strategy_file="strategies/adaptive_ma_crossover.py",
        strategy_name="Moving Average",
        parameters=parameters,
        symbol="SPXL",
        save_path="results/moving_average_visualization.png"
    )


def main():
    """Main function with strategy selection for visualization"""
    print("🎨 ENHANCED STRATEGY VISUALIZER")
    print("=" * 50)
    print("Choose a strategy to visualize:")
    print("1. RSI Strategy")
    print("2. Bollinger Bands Strategy")
    print("3. Moving Average Strategy")
    print("4. All Strategies")
    print("=" * 50)
    
    choice = input("Enter choice (1/2/3/4): ").strip()
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    if choice == "1":
        demo_rsi_visualization()
    elif choice == "2":
        demo_bollinger_bands_visualization()
    elif choice == "3":
        demo_moving_average_visualization()
    elif choice == "4":
        print("\n🎨 Visualizing all strategies...")
        demo_rsi_visualization()
        demo_bollinger_bands_visualization()
        demo_moving_average_visualization()
    else:
        print("Invalid choice. Running RSI visualization by default.")
        demo_rsi_visualization()
    
    print("\n" + "=" * 60)
    print("🎉 VISUALIZATION COMPLETED!")


if __name__ == "__main__":
    main()
