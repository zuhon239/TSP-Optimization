"""
Visualization Module for TSP Optimization

Create charts, graphs, and visualizations for algorithm results

Author: Quân (Frontend Specialist)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import streamlit as st
import config

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TSPVisualizer:
    """
    Main visualization class for TSP optimization results
    """
    
    def __init__(self, width: int = 800, height: int = 600, theme: str = 'plotly_white'):
        """
        Initialize visualizer
        
        Args:
            width: Default figure width
            height: Default figure height
            theme: Plotly theme to use
        """
        self.width = width
        self.height = height
        self.theme = theme
        self.colors = {
            'ga': '#1f77b4',    # Blue
            'pso': '#ff7f0e',   # Orange
            'best': '#2ca02c',  # Green
            'route': '#d62728', # Red
            'start': '#17becf', # Cyan
            'waypoint': '#e377c2' # Pink
        }
    
    def plot_convergence(self, 
                        convergence_data: Union[Dict, List],
                        algorithm_name: str = "Algorithm") -> go.Figure:
        """
        Plot convergence for single algorithm
        
        Args:
            convergence_data: Convergence data (dict with 'best'/'average' or list)
            algorithm_name: Name of algorithm
        
        Returns:
            Plotly figure object
        """
        # Handle different data formats
        best_data = None
        avg_data = None
        
        if isinstance(convergence_data, dict):
            best_data = convergence_data.get('best_distances', convergence_data.get('best'))
            avg_data = convergence_data.get('average_distances', convergence_data.get('average'))
        elif isinstance(convergence_data, (list, np.ndarray)):
            best_data = convergence_data
        
        if not best_data or len(best_data) == 0:
            raise ValueError("No convergence data available")
        
        fig = go.Figure()
        
        # Best distance line
        fig.add_trace(go.Scatter(
            x=list(range(len(best_data))),
            y=best_data,
            mode='lines',
            name='Best Distance',
            line=dict(color=self.colors['best'], width=3),
            hovertemplate='Iteration: %{x}<br>Best Distance: %{y:.2f} km<extra></extra>'
        ))
        
        # Average distance line (if available)
        if avg_data and len(avg_data) > 0:
            fig.add_trace(go.Scatter(
                x=list(range(len(avg_data))),
                y=avg_data,
                mode='lines',
                name='Average Distance',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                hovertemplate='Iteration: %{x}<br>Average Distance: %{y:.2f} km<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=f"{algorithm_name} - Convergence", x=0.5, font=dict(size=16)),
            xaxis_title="Iteration",
            yaxis_title="Distance (km)",
            template=self.theme,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            width=self.width,
            height=self.height
        )
        
        return fig

