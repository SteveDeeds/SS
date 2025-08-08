#!/usr/bin/env python3
"""
Heat Map Generator Utility

Utility class for generating heat maps from optimization results data.
Extracted from the GUI visualizer for reuse in other examples.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union
import os


class HeatMapGenerator:
    """
    Utility class for generating heat maps from optimization results.
    
    Supports filtering, parameter selection, and various customization options.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the heat map generator.
        
        Args:
            data: Optional DataFrame with optimization results
        """
        self.data = data
        self.filtered_data = None
        self.available_params = []
        self.available_metrics = []
        self.available_strategies = []
        self.available_symbols = []
        
        if self.data is not None:
            self._analyze_data()
    
    def load_data(self, data_source: Union[str, pd.DataFrame]) -> bool:
        """
        Load data from CSV file or DataFrame.
        
        Args:
            data_source: Path to CSV file or pandas DataFrame
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            if isinstance(data_source, str):
                self.data = pd.read_csv(data_source)
            elif isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
            else:
                raise ValueError("Data source must be a file path or pandas DataFrame")
            
            self._analyze_data()
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _analyze_data(self):
        """Analyze the loaded data to extract available parameters and metrics."""
        if self.data is None:
            return
        
        # Extract parameter columns
        self.available_params = [col.replace('param_', '') for col in self.data.columns 
                                if col.startswith('param_')]
        
        # Extract available strategies and symbols
        if 'strategy_file' in self.data.columns:
            self.available_strategies = sorted(self.data['strategy_file'].unique().tolist())
        if 'symbol' in self.data.columns:
            self.available_symbols = sorted(self.data['symbol'].unique().tolist())
        
        # Find metric columns (exclude specific columns and param columns)
        excluded_columns = {'datetime', 'strategy_file', 'strategy_name', 'symbol', 'parameter_combination'}
        self.available_metrics = [col for col in self.data.columns 
                                 if col not in excluded_columns and not col.startswith('param_')]
    
    def get_available_parameters(self) -> List[str]:
        """Get list of available parameters for axes."""
        return self.available_params.copy()
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics for visualization."""
        return self.available_metrics.copy()
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies for filtering."""
        return self.available_strategies.copy()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols for filtering."""
        return self.available_symbols.copy()
    
    def apply_filters(self, 
                     strategy: Optional[str] = None,
                     symbol: Optional[str] = None,
                     param_filters: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
        """
        Apply filters to the data.
        
        Args:
            strategy: Strategy file to filter by (None for no filter)
            symbol: Symbol to filter by (None for no filter)
            param_filters: Dictionary of parameter filters {param_name: (min_val, max_val)}
            
        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            return pd.DataFrame()
        
        filtered = self.data.copy()
        
        # Strategy filter
        if strategy and strategy != "All":
            filtered = filtered[filtered['strategy_file'] == strategy]
        
        # Symbol filter
        if symbol and symbol != "All":
            filtered = filtered[filtered['symbol'] == symbol]
        
        # Parameter filters
        if param_filters:
            for param, (min_val, max_val) in param_filters.items():
                param_col = f'param_{param}'
                if param_col in filtered.columns:
                    try:
                        param_values = pd.to_numeric(filtered[param_col], errors='coerce')
                        filtered = filtered[(param_values >= min_val) & (param_values <= max_val)]
                    except (ValueError, TypeError):
                        continue
        
        self.filtered_data = filtered
        return filtered
    
    def generate_heatmap(self,
                        x_param: str,
                        y_param: str,
                        metric: str,
                        strategy: Optional[str] = None,
                        symbol: Optional[str] = None,
                        param_filters: Optional[Dict[str, Tuple[float, float]]] = None,
                        colormap: str = 'RdYlGn',
                        figsize: Tuple[int, int] = (10, 8),
                        title: Optional[str] = None,
                        save_path: Optional[str] = None,
                        show_values: bool = True,
                        aggfunc: str = 'mean') -> Tuple[Figure, plt.Axes]:
        """
        Generate a heat map from the optimization results.
        
        Args:
            x_param: Parameter name for X-axis
            y_param: Parameter name for Y-axis
            metric: Metric to visualize
            strategy: Strategy to filter by (optional)
            symbol: Symbol to filter by (optional)
            param_filters: Parameter range filters (optional)
            colormap: Matplotlib colormap name
            figsize: Figure size (width, height)
            title: Custom title (optional)
            save_path: Path to save the plot (optional)
            show_values: Whether to show values in cells
            aggfunc: Aggregation function ('mean', 'sum', 'median', etc.)
            
        Returns:
            Tuple of (Figure, Axes) objects
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Apply filters
        filtered_data = self.apply_filters(strategy, symbol, param_filters)
        
        if len(filtered_data) == 0:
            raise ValueError("No data available after applying filters")
        
        # Validate parameters and metric
        x_col = f'param_{x_param}'
        y_col = f'param_{y_param}'
        
        if x_col not in filtered_data.columns:
            raise ValueError(f"Parameter '{x_param}' not found in data")
        if y_col not in filtered_data.columns:
            raise ValueError(f"Parameter '{y_param}' not found in data")
        if metric not in filtered_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        # Prepare data for plotting
        plot_data = filtered_data[[x_col, y_col, metric]].copy()
        plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
        plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
        plot_data[metric] = pd.to_numeric(plot_data[metric], errors='coerce')
        plot_data = plot_data.dropna()
        
        if len(plot_data) == 0:
            raise ValueError("No valid numeric data for selected parameters")
        
        # Create pivot table
        pivot_table = plot_data.pivot_table(
            index=y_col,
            columns=x_col,
            values=metric,
            aggfunc=aggfunc
        )
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine number format based on data range
        data_min = pivot_table.min().min()
        data_max = pivot_table.max().max()
        data_range = max(abs(data_min), abs(data_max))
        
        if data_range < 10:
            fmt = '.2f'  # 2 decimal places for small numbers
        elif data_range < 100:
            fmt = '.1f'  # 1 decimal place for medium numbers
        else:
            fmt = '.0f'  # No decimal places for large numbers
        
        # Create symmetric range around zero for better color distinction
        data_abs_max = max(abs(data_min), abs(data_max))
        vmin = -data_abs_max
        vmax = data_abs_max
        center = 0
        
        # Generate heat map
        sns.heatmap(
            pivot_table,
            ax=ax,
            annot=show_values,
            fmt=fmt,
            cmap=colormap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        # Set labels and title
        ax.set_xlabel(f'{x_param.title()}')
        ax.set_ylabel(f'{y_param.title()}')
        
        if title:
            ax.set_title(title)
        else:
            filter_info = []
            if strategy and strategy != "All":
                filter_info.append(f"Strategy: {strategy}")
            if symbol and symbol != "All":
                filter_info.append(f"Symbol: {symbol}")
            
            title_text = f'{metric.replace("_", " ").title()} Heat Map'
            if filter_info:
                title_text += f'\n{" | ".join(filter_info)}'
            
            ax.set_title(title_text)
        
        # Rotate labels for better readability
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def generate_comparison_heatmaps(self,
                                   x_param: str,
                                   y_param: str,
                                   metric: str,
                                   comparison_column: str,
                                   comparison_values: List[str],
                                   param_filters: Optional[Dict[str, Tuple[float, float]]] = None,
                                   colormap: str = 'RdYlGn',
                                   figsize: Tuple[int, int] = (15, 5),
                                   save_path: Optional[str] = None) -> Tuple[Figure, List[plt.Axes]]:
        """
        Generate multiple heat maps for comparison (e.g., different symbols or strategies).
        
        Args:
            x_param: Parameter name for X-axis
            y_param: Parameter name for Y-axis
            metric: Metric to visualize
            comparison_column: Column to use for comparison ('symbol', 'strategy_file', etc.)
            comparison_values: Values to compare
            param_filters: Parameter range filters (optional)
            colormap: Matplotlib colormap name
            figsize: Figure size (width, height)
            save_path: Path to save the plot (optional)
            
        Returns:
            Tuple of (Figure, List of Axes) objects
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        n_plots = len(comparison_values)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
        
        if n_plots == 1:
            axes = [axes]
        
        # Find global min/max for consistent color scaling
        global_min = float('inf')
        global_max = float('-inf')
        
        pivot_tables = []
        
        for value in comparison_values:
            # Filter data for this comparison value
            filtered_data = self.data[self.data[comparison_column] == value].copy()
            
            # Apply additional filters
            if param_filters:
                for param, (min_val, max_val) in param_filters.items():
                    param_col = f'param_{param}'
                    if param_col in filtered_data.columns:
                        try:
                            param_values = pd.to_numeric(filtered_data[param_col], errors='coerce')
                            filtered_data = filtered_data[(param_values >= min_val) & (param_values <= max_val)]
                        except (ValueError, TypeError):
                            continue
            
            if len(filtered_data) == 0:
                pivot_tables.append(None)
                continue
            
            # Prepare data
            x_col = f'param_{x_param}'
            y_col = f'param_{y_param}'
            
            plot_data = filtered_data[[x_col, y_col, metric]].copy()
            plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
            plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
            plot_data[metric] = pd.to_numeric(plot_data[metric], errors='coerce')
            plot_data = plot_data.dropna()
            
            if len(plot_data) == 0:
                pivot_tables.append(None)
                continue
            
            # Create pivot table
            pivot_table = plot_data.pivot_table(
                index=y_col,
                columns=x_col,
                values=metric,
                aggfunc='mean'
            )
            
            pivot_tables.append(pivot_table)
            
            # Update global min/max
            if not pivot_table.empty:
                global_min = min(global_min, pivot_table.min().min())
                global_max = max(global_max, pivot_table.max().max())
        
        # Create symmetric range around zero
        data_abs_max = max(abs(global_min), abs(global_max))
        vmin = -data_abs_max
        vmax = data_abs_max
        
        # Generate heat maps
        for i, (value, pivot_table) in enumerate(zip(comparison_values, pivot_tables)):
            ax = axes[i]
            
            if pivot_table is None or pivot_table.empty:
                ax.text(0.5, 0.5, f'No data\nfor {value}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{value}')
                continue
            
            # Determine format
            data_range = max(abs(global_min), abs(global_max))
            if data_range < 10:
                fmt = '.2f'
            elif data_range < 100:
                fmt = '.1f'
            else:
                fmt = '.0f'
            
            sns.heatmap(
                pivot_table,
                ax=ax,
                annot=True,
                fmt=fmt,
                cmap=colormap,
                center=0,
                vmin=vmin,
                vmax=vmax,
                cbar=i == n_plots - 1  # Only show colorbar on last plot
            )
            
            ax.set_title(f'{value}')
            ax.set_xlabel(f'{x_param.title()}')
            
            if i == 0:
                ax.set_ylabel(f'{y_param.title()}')
            else:
                ax.set_ylabel('')
            
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes


def create_heatmap_from_csv(csv_path: str,
                           x_param: str,
                           y_param: str,
                           metric: str,
                           **kwargs) -> Tuple[Figure, plt.Axes]:
    """
    Convenience function to create a heat map directly from a CSV file.
    
    Args:
        csv_path: Path to CSV file with optimization results
        x_param: Parameter name for X-axis
        y_param: Parameter name for Y-axis
        metric: Metric to visualize
        **kwargs: Additional arguments passed to generate_heatmap()
        
    Returns:
        Tuple of (Figure, Axes) objects
    """
    generator = HeatMapGenerator()
    
    if not generator.load_data(csv_path):
        raise ValueError(f"Failed to load data from {csv_path}")
    
    return generator.generate_heatmap(x_param, y_param, metric, **kwargs)


def create_comparison_heatmaps_from_csv(csv_path: str,
                                       x_param: str,
                                       y_param: str,
                                       metric: str,
                                       comparison_column: str,
                                       comparison_values: List[str],
                                       **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    """
    Convenience function to create comparison heat maps directly from a CSV file.
    
    Args:
        csv_path: Path to CSV file with optimization results
        x_param: Parameter name for X-axis
        y_param: Parameter name for Y-axis
        metric: Metric to visualize
        comparison_column: Column to use for comparison
        comparison_values: Values to compare
        **kwargs: Additional arguments passed to generate_comparison_heatmaps()
        
    Returns:
        Tuple of (Figure, List of Axes) objects
    """
    generator = HeatMapGenerator()
    
    if not generator.load_data(csv_path):
        raise ValueError(f"Failed to load data from {csv_path}")
    
    return generator.generate_comparison_heatmaps(
        x_param, y_param, metric, comparison_column, comparison_values, **kwargs
    )


# Additional utility methods for GUI integration
class HeatMapGeneratorGUI(HeatMapGenerator):
    """Extended version of HeatMapGenerator for GUI integration"""
    
    def generate_heatmap_on_figure(self, 
                                  fig: Figure,
                                  x_param: str,
                                  y_param: str,
                                  metric: str,
                                  colormap: str = 'RdYlGn',
                                  strategy: Optional[str] = None,
                                  symbol: Optional[str] = None,
                                  param_filters: Optional[Dict[str, Tuple[float, float]]] = None) -> bool:
        """
        Generate a heat map on an existing figure (for GUI integration).
        
        Args:
            fig: Matplotlib Figure object to draw on
            x_param: Parameter name for X-axis
            y_param: Parameter name for Y-axis
            metric: Metric to visualize
            colormap: Matplotlib colormap name
            strategy: Strategy to filter by (optional)
            symbol: Symbol to filter by (optional)
            param_filters: Parameter range filters (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.data is None:
                return False
            
            # Apply filters
            filtered_data = self.apply_filters(strategy, symbol, param_filters)
            
            if len(filtered_data) == 0:
                return False
            
            # Validate parameters and metric
            x_col = f'param_{x_param}'
            y_col = f'param_{y_param}'
            
            if x_col not in filtered_data.columns or y_col not in filtered_data.columns:
                return False
            if metric not in filtered_data.columns:
                return False
            
            # Convert to numeric and drop NaN values
            plot_data = filtered_data[[x_col, y_col, metric]].copy()
            plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
            plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
            plot_data[metric] = pd.to_numeric(plot_data[metric], errors='coerce')
            plot_data = plot_data.dropna()
            
            if len(plot_data) == 0:
                return False
            
            # Create pivot table for heat map
            aggfunc = 'sum' if metric == 'test_scenarios' else 'mean'
            pivot_table = plot_data.pivot_table(
                index=y_col, 
                columns=x_col, 
                values=metric, 
                aggfunc=aggfunc
            )
            
            # Clear figure and create new axes
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Determine formatting based on data range
            data_min = pivot_table.min().min()
            data_max = pivot_table.max().max()
            data_range = max(abs(data_min), abs(data_max))
            
            if data_range < 10:
                fmt = '.2f'
            elif data_range < 100:
                fmt = '.1f'
            else:
                fmt = '.0f'
            
            # Create symmetric range for better color distinction
            data_abs_max = max(abs(data_min), abs(data_max))
            vmin = -data_abs_max
            vmax = data_abs_max
            center = 0
            
            # Create the heat map
            sns.heatmap(
                pivot_table, 
                ax=ax, 
                annot=True, 
                fmt=fmt, 
                cmap=colormap,
                center=center,
                vmin=vmin,
                vmax=vmax
            )
            
            # Set labels and title
            ax.set_xlabel(f'{x_param.title()}')
            ax.set_ylabel(f'{y_param.title()}')
            ax.set_title(f'{metric.replace("_", " ").title()} Heat Map\n'
                        f'{x_param.title()} vs {y_param.title()}')
            
            # Rotate labels for better readability
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
            
            fig.tight_layout()
            
            return True
            
        except Exception:
            return False
