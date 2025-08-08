#!/usr/bin/env python3
"""
Heat Map Visualizer for Optimization Results

A GUI application for creating interactive heat maps from optimization CSV results.
Allows filtering and visualization of parameter performance across different dimensions.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.utils.heatmap_generator import HeatMapGeneratorGUI


class HeatMapVisualizer:
    """
    GUI application for creating heat maps from optimization results.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Optimization Results Heat Map Visualizer")
        self.root.geometry("1200x800")
        
        # Heat map generator
        self.heatmap_generator = HeatMapGeneratorGUI()
        
        # Data storage
        self.df = None
        self.filtered_df = None
        self.available_params = []
        self.available_strategies = []
        self.available_symbols = []
        
        # GUI variables
        self.strategy_var = tk.StringVar()
        self.symbol_var = tk.StringVar()
        self.metric_var = tk.StringVar(value="return_on_avg_invested_capital")
        self.x_axis_var = tk.StringVar()
        self.y_axis_var = tk.StringVar()
        
        # Parameter filter variables (will be created dynamically)
        self.param_vars = {}
        self.param_min_vars = {}
        self.param_max_vars = {}
        self.param_use_x_vars = {}
        self.param_use_y_vars = {}
        
        self.setup_ui()
        self.load_default_data()
    
    def setup_ui(self):
        """Set up the user interface"""
        
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection
        file_frame = ttk.LabelFrame(control_frame, text="Data Source", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load CSV File", command=self.load_csv_file).pack(fill=tk.X)
        self.file_label = ttk.Label(file_frame, text="No file loaded", foreground="gray")
        self.file_label.pack(fill=tk.X, pady=(5, 0))
        
        # Filters
        filter_frame = ttk.LabelFrame(control_frame, text="Filters", padding=10)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Strategy filter
        ttk.Label(filter_frame, text="Strategy:").pack(anchor=tk.W)
        self.strategy_combo = ttk.Combobox(filter_frame, textvariable=self.strategy_var, 
                                          values=[], state="readonly")
        self.strategy_combo.pack(fill=tk.X, pady=(0, 5))
        self.strategy_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        
        # Symbol filter
        ttk.Label(filter_frame, text="Symbol:").pack(anchor=tk.W)
        self.symbol_combo = ttk.Combobox(filter_frame, textvariable=self.symbol_var,
                                        values=[], state="readonly")
        self.symbol_combo.pack(fill=tk.X, pady=(0, 5))
        self.symbol_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        
        # Metric selection
        ttk.Label(filter_frame, text="Metric to visualize:").pack(anchor=tk.W)
        self.metric_combo = ttk.Combobox(filter_frame, textvariable=self.metric_var,
                                        values=[], state="readonly")
        self.metric_combo.pack(fill=tk.X, pady=(0, 5))
        self.metric_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        
        # Colormap selection
        ttk.Label(filter_frame, text="Color scheme:").pack(anchor=tk.W)
        self.colormap_var = tk.StringVar(value="RdYlGn")
        colormap_options = [
            "RdYlGn",      # Red-Yellow-Green (default)
            "RdBu",        # Red-Blue (good for zero distinction)
            "RdBu_r",      # Red-Blue reversed
            "coolwarm",    # Cool-Warm (blue-red)
            "bwr",         # Blue-White-Red
            "seismic",     # Blue-White-Red (earthquake style)
            "PiYG",        # Pink-Yellow-Green
            "PRGn",        # Purple-Green
            "BrBG",        # Brown-Blue-Green
            "RdGy",        # Red-Gray
            "viridis",     # Viridis (no zero distinction)
            "plasma",      # Plasma (no zero distinction)
        ]
        self.colormap_combo = ttk.Combobox(filter_frame, textvariable=self.colormap_var,
                                          values=colormap_options, state="readonly")
        self.colormap_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Parameter controls frame (will be populated dynamically)
        self.param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding=10)
        self.param_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create scrollable area for parameters
        self.param_canvas = tk.Canvas(self.param_frame)
        self.param_scrollbar = ttk.Scrollbar(self.param_frame, orient="vertical", command=self.param_canvas.yview)
        self.param_scrollable_frame = ttk.Frame(self.param_canvas)
        
        self.param_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))
        )
        
        self.param_canvas.create_window((0, 0), window=self.param_scrollable_frame, anchor="nw")
        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)
        
        self.param_canvas.pack(side="left", fill="both", expand=True)
        self.param_scrollbar.pack(side="right", fill="y")
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="Generate Heat Map", command=self.generate_heatmap).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Export Plot", command=self.export_plot).pack(fill=tk.X)
        
        # Plot area
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_default_data(self):
        """Try to load default CSV file if it exists"""
        default_file = "results/continuous_optimization_results.csv"
        if os.path.exists(default_file):
            self.load_csv_data(default_file)
    
    def load_csv_file(self):
        """Open file dialog to select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select Optimization Results CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_csv_data(file_path)
    
    def load_csv_data(self, file_path: str):
        """Load and process CSV data"""
        try:
            # Load data using the heat map generator
            if not self.heatmap_generator.load_data(file_path):
                raise Exception("Failed to load data with heat map generator")
            
            self.df = self.heatmap_generator.data
            self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}", foreground="green")
            
            # Get available values from the generator
            self.available_strategies = ["All"] + self.heatmap_generator.get_available_strategies()
            self.available_symbols = ["All"] + self.heatmap_generator.get_available_symbols()
            self.available_params = self.heatmap_generator.get_available_parameters()
            available_metrics = self.heatmap_generator.get_available_metrics()
            
            # Update UI
            self.strategy_combo['values'] = self.available_strategies
            self.symbol_combo['values'] = self.available_symbols
            self.metric_combo['values'] = available_metrics
            
            # Set default selections
            self.strategy_var.set("All")
            self.symbol_var.set("All")
            if available_metrics:
                self.metric_var.set(available_metrics[0])  # Set first available metric as default
            
            # Create parameter controls
            self.create_parameter_controls()
            
            # Apply initial filter
            self.apply_filters()
            
            self.status_var.set(f"Loaded {len(self.df)} records with {len(self.available_params)} parameters")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
            self.status_var.set("Error loading file")
    
    def create_parameter_controls(self):
        """Create UI controls for each parameter"""
        # Clear existing controls
        for widget in self.param_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.param_vars.clear()
        self.param_min_vars.clear()
        self.param_max_vars.clear()
        self.param_use_x_vars.clear()
        self.param_use_y_vars.clear()
        
        for i, param in enumerate(self.available_params):
            # Get parameter stats
            param_col = f'param_{param}'
            if param_col in self.df.columns:
                param_values = pd.to_numeric(self.df[param_col], errors='coerce').dropna()
                min_val = param_values.min()
                max_val = param_values.max()
                
                # Parameter frame
                param_group = ttk.LabelFrame(self.param_scrollable_frame, text=param.title(), padding=5)
                param_group.pack(fill=tk.X, pady=2)
                
                # Axis selection frame
                axis_frame = ttk.Frame(param_group)
                axis_frame.pack(fill=tk.X)
                
                # X-axis checkbox
                self.param_use_x_vars[param] = tk.BooleanVar()
                x_check = ttk.Checkbutton(axis_frame, text="X-axis", variable=self.param_use_x_vars[param],
                                         command=lambda p=param: self.on_axis_selection(p, 'x'))
                x_check.pack(side=tk.LEFT)
                
                # Y-axis checkbox
                self.param_use_y_vars[param] = tk.BooleanVar()
                y_check = ttk.Checkbutton(axis_frame, text="Y-axis", variable=self.param_use_y_vars[param],
                                         command=lambda p=param: self.on_axis_selection(p, 'y'))
                y_check.pack(side=tk.LEFT, padx=(10, 0))
                
                # Filter enable checkbox - checked by default
                self.param_vars[param] = tk.BooleanVar(value=True)
                filter_check = ttk.Checkbutton(axis_frame, text="Filter", variable=self.param_vars[param],
                                              command=self.on_filter_change)
                filter_check.pack(side=tk.LEFT, padx=(10, 0))
                
                # Get unique parameter values for dropdowns
                unique_values = sorted(param_values.unique())
                
                # Range frame
                range_frame = ttk.Frame(param_group)
                range_frame.pack(fill=tk.X, pady=(5, 0))
                
                # Values info label
                ttk.Label(range_frame, text=f"Values: {len(unique_values)} unique").pack(anchor=tk.W)
                
                # Min value dropdown
                min_frame = ttk.Frame(range_frame)
                min_frame.pack(fill=tk.X)
                ttk.Label(min_frame, text="Min:").pack(side=tk.LEFT)
                self.param_min_vars[param] = tk.StringVar(value=str(min_val))
                min_combo = ttk.Combobox(min_frame, textvariable=self.param_min_vars[param], 
                                        values=[str(v) for v in unique_values], width=8, state="readonly")
                min_combo.pack(side=tk.LEFT, padx=(5, 0))
                min_combo.bind('<<ComboboxSelected>>', lambda e: self.on_filter_change())
                
                # Max value dropdown
                max_frame = ttk.Frame(range_frame)
                max_frame.pack(fill=tk.X)
                ttk.Label(max_frame, text="Max:").pack(side=tk.LEFT)
                self.param_max_vars[param] = tk.StringVar(value=str(max_val))
                max_combo = ttk.Combobox(max_frame, textvariable=self.param_max_vars[param], 
                                        values=[str(v) for v in unique_values], width=8, state="readonly")
                max_combo.pack(side=tk.LEFT, padx=(5, 0))
                max_combo.bind('<<ComboboxSelected>>', lambda e: self.on_filter_change())
    
    def on_axis_selection(self, param: str, axis: str):
        """Handle axis selection changes"""
        if axis == 'x':
            # Uncheck other X-axis selections
            for other_param, var in self.param_use_x_vars.items():
                if other_param != param:
                    var.set(False)
            self.x_axis_var.set(param if self.param_use_x_vars[param].get() else "")
        
        elif axis == 'y':
            # Uncheck other Y-axis selections
            for other_param, var in self.param_use_y_vars.items():
                if other_param != param:
                    var.set(False)
            self.y_axis_var.set(param if self.param_use_y_vars[param].get() else "")
    
    def on_filter_change(self, event=None):
        """Handle filter changes"""
        self.apply_filters()
    
    def apply_filters(self):
        """Apply current filters to the data"""
        if self.df is None:
            return
        
        filtered = self.df.copy()
        
        # Strategy filter
        if self.strategy_var.get() and self.strategy_var.get() != "All":
            filtered = filtered[filtered['strategy_file'] == self.strategy_var.get()]
        
        # Symbol filter
        if self.symbol_var.get() and self.symbol_var.get() != "All":
            filtered = filtered[filtered['symbol'] == self.symbol_var.get()]
        
        # Parameter filters
        for param in self.available_params:
            if self.param_vars[param].get():  # Filter is enabled
                param_col = f'param_{param}'
                if param_col in filtered.columns:
                    try:
                        # Convert string values from dropdown to numbers
                        min_val = float(self.param_min_vars[param].get())
                        max_val = float(self.param_max_vars[param].get())
                        
                        # Convert parameter column to numeric for comparison
                        param_values = pd.to_numeric(filtered[param_col], errors='coerce')
                        filtered = filtered[(param_values >= min_val) & (param_values <= max_val)]
                    except (ValueError, TypeError):
                        # Skip filter if conversion fails
                        continue
        
        self.filtered_df = filtered
        self.status_var.set(f"Filtered to {len(filtered)} records")
    
    def generate_heatmap(self):
        """Generate and display the heat map"""
        if self.filtered_df is None or len(self.filtered_df) == 0:
            messagebox.showwarning("Warning", "No data available for heat map")
            return
        
        x_param = self.x_axis_var.get()
        y_param = self.y_axis_var.get()
        
        if not x_param or not y_param:
            messagebox.showwarning("Warning", "Please select both X and Y axis parameters")
            return
        
        try:
            # Load the current filtered data into the generator
            self.heatmap_generator.load_data(self.filtered_df)
            
            # Use the GUI-specific method to generate the heat map on our figure
            success = self.heatmap_generator.generate_heatmap_on_figure(
                fig=self.fig,
                x_param=x_param,
                y_param=y_param,
                metric=self.metric_var.get(),
                colormap=self.colormap_var.get()
            )
            
            if success:
                # Update the axes reference after clearing/recreating
                self.ax = self.fig.axes[0] if self.fig.axes else None
                self.canvas.draw()
                self.status_var.set(f"Heat map generated successfully")
            else:
                self.status_var.set("Error generating heat map")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate heat map: {str(e)}")
            self.status_var.set("Error generating heat map")
    
    def export_plot(self):
        """Export the current plot to file"""
        if self.ax.get_title():
            file_path = filedialog.asksaveasfilename(
                title="Export Heat Map",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
            )
            
            if file_path:
                try:
                    self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Success", f"Heat map exported to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No heat map to export")


def main():
    """Run the heat map visualizer application"""
    root = tk.Tk()
    app = HeatMapVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
