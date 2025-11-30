"""
Visualization Module for TSP Optimization

Create charts, graphs, and visualizations for algorithm results
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
    
    def plot_distance_matrix(self, 
                            distance_matrix: np.ndarray,
                            location_names: List[str] = None) -> go.Figure:
        """
        Plot distance matrix as heatmap (alias for plot_distance_matrix_heatmap)
        
        Args:
            distance_matrix: Distance matrix
            location_names: Names for locations
        
        Returns:
            Plotly figure object
        """
        return self.plot_distance_matrix_heatmap(distance_matrix, location_names)
    
    def plot_distance_matrix_heatmap(self, 
                                    distance_matrix: np.ndarray,
                                    location_names: List[str] = None) -> go.Figure:
        """
        Plot distance matrix as heatmap
        
        Args:
            distance_matrix: Distance matrix
            location_names: Names for locations
        
        Returns:
            Plotly figure object
        """
        if location_names is None:
            location_names = [f"City {i+1}" for i in range(len(distance_matrix))]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=distance_matrix,
            x=location_names,
            y=location_names,
            colorscale='Viridis',
            text=np.round(distance_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Distance (km)")
        ))
        
        fig.update_layout(
            title="Distance Matrix Heatmap",
            xaxis_title="Destination",
            yaxis_title="Origin",
            width=self.width,
            height=self.height,
            template=self.theme
        )
        
        return fig
    
    def display_algorithm_params(self, results: Dict) -> None:
        """
        Display algorithm parameters using Streamlit metrics
        (Not a Plotly chart - uses st.metrics directly)
        
        Args:
            results: Results dictionary containing algorithm parameters
        """
        import streamlit as st
        
        algorithm = results.get('algorithm', 'Unknown')
        
        if 'Genetic' in algorithm:
            st.write("### ⚙️ Genetic Algorithm Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("👥 Population Size", results.get('population_size', 'N/A'))
                st.metric("🧬 Crossover Rate", f"{results.get('crossover_prob', 0):.2%}")
            
            with col2:
                st.metric("🔄 Generations", results.get('generations', 'N/A'))
                st.metric("🎲 Mutation Rate", f"{results.get('mutation_prob', 0):.2%}")
            
            with col3:
                st.metric("🏆 Elite Size", results.get('elite_size', 'N/A'))
                st.metric("🎯 Tournament Size", results.get('tournament_size', 'N/A'))
        
        elif 'PSO' in algorithm:
            st.write("### ⚙️ PSO Algorithm Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🐝 Swarm Size", results.get('swarm_size', 'N/A'))
                st.metric("⚖️ Inertia (w)", f"{results.get('w', 0):.3f}")
            
            with col2:
                st.metric("🔄 Max Iterations", results.get('max_iterations', 'N/A'))
                st.metric("🧠 Cognitive (c1)", f"{results.get('c1', 0):.3f}")
            
            with col3:
                w_min = results.get('w_min', 0)
                w_max = results.get('w_max', 0)
                st.metric("🌡️ w Range", f"{w_min:.2f} - {w_max:.2f}")
                st.metric("👥 Social (c2)", f"{results.get('c2', 0):.3f}")
    
    def plot_route_segments(self, 
                            route: List[int],
                            distance_matrix: np.ndarray,
                            locations: List[Dict] = None) -> go.Figure:
        """
        Visualize route segment lengths analysis
        
        Args:
            route: Route sequence (list of indices)
            distance_matrix: Distance matrix
            locations: Location names (optional)
        
        Returns:
            Plotly figure with segment analysis
        """
        # Calculate segment lengths
        segments = []
        segment_labels = []
        
        for i in range(len(route)):
            from_idx = route[i]
            to_idx = route[(i + 1) % len(route)]
            distance = distance_matrix[from_idx][to_idx]
            segments.append(distance)
            
            # Create labels
            if locations:
                from_name = locations[from_idx].get('name', f'Point {from_idx}')[:15]
                to_name = locations[to_idx].get('name', f'Point {to_idx}')[:15]
                segment_labels.append(f"{from_name} → {to_name}")
            else:
                segment_labels.append(f"Segment {i+1}")
        
        # Calculate statistics
        avg_segment = np.mean(segments)
        max_segment = max(segments)
        min_segment = min(segments)
        std_segment = np.std(segments)
        
        # Create bar chart
        fig = go.Figure()
        
        # Add bars
        colors = ['#1f77b4' if s < avg_segment else '#ff7f0e' for s in segments]
        fig.add_trace(go.Bar(
            x=list(range(len(segments))),
            y=segments,
            marker_color=colors,
            text=[f"{s:.2f} km" for s in segments],
            textposition='outside',
            hovertemplate='%{customdata}<br>Distance: %{y:.2f} km<extra></extra>',
            customdata=segment_labels
        ))
        
        # Add average line
        fig.add_hline(
            y=avg_segment, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Average: {avg_segment:.2f} km",
            annotation_position="right"
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Route Segment Analysis<br><sub>Std Dev: {std_segment:.2f} km | Min: {min_segment:.2f} km | Max: {max_segment:.2f} km</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Segment Number",
            yaxis_title="Distance (km)",
            showlegend=False,
            width=self.width,
            height=self.height,
            template=self.theme,
            hovermode='x'
        )
        
        return fig
    
    def plot_convergence_milestones(self, 
                                    convergence_data: Union[List, Dict],
                                    algorithm_name: str = "Algorithm") -> go.Figure:
        """
        Plot convergence with milestone annotations
        
        Args:
            convergence_data: Convergence history (list or dict)
            algorithm_name: Name of algorithm
        
        Returns:
            Plotly figure with annotated milestones
        """
        # Extract best distances
        if isinstance(convergence_data, dict):
            best_data = convergence_data.get('best_distances', convergence_data.get('best', []))
        else:
            best_data = convergence_data
        
        if not best_data or len(best_data) == 0:
            raise ValueError("No convergence data available")
        
        # Create figure
        fig = go.Figure()
        
        # Plot convergence line
        fig.add_trace(go.Scatter(
            x=list(range(len(best_data))),
            y=best_data,
            mode='lines',
            name='Best Distance',
            line=dict(color=self.colors['best'], width=3),
            hovertemplate='Iteration: %{x}<br>Best: %{y:.2f} km<extra></extra>'
        ))
        
        # Calculate milestones
        initial_distance = best_data[0]
        final_distance = best_data[-1]
        improvement = initial_distance - final_distance
        
        # Add final improvement text
        if improvement > 0:
            improvement_pct = (improvement / initial_distance) * 100
            fig.add_annotation(
                x=len(best_data) - 1,
                y=final_distance,
                text=f"Final: {final_distance:.2f} km<br>↓ {improvement_pct:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor='green',
                ax=-60,
                ay=30,
                bgcolor='rgba(144,238,144,0.3)',
                bordercolor='green',
                borderwidth=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"{algorithm_name} - Convergence",
            xaxis_title="Iteration",
            yaxis_title="Distance (km)",
            width=self.width,
            height=self.height,
            template=self.theme,
            hovermode='x'
        )
        
        return fig


if __name__ == "__main__":
    # Example usage and testing
    print("TSP Visualizer Module")
    print("Available functions:")
    print("- TSPVisualizer.plot_convergence()")
    print("- TSPVisualizer.plot_distance_matrix()")
    print("- create_distance_heatmap()")

