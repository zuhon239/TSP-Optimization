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

