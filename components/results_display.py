"""
Results Display Components
Display optimization results with Visualizer integration

Author: Quân (Frontend Specialist)
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# ✅ Plotly for charts
import plotly.graph_objects as go


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula (km)"""
    R = 6371
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def get_openroute_directions(coordinates: List[List[float]], api_key: str) -> Optional[Dict]:
    """
    Get real road directions from OpenRouteService
    
    Args:
        coordinates: List of [lng, lat] pairs
        api_key: OpenRouteService API key
    
    Returns:
        Route geometry dict or None
    """
    try:
        url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
        
        headers = {
            'Accept': 'application/json, application/geo+json',
            'Authorization': api_key,
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        body = {
            "coordinates": coordinates,
            "instructions": "true",
            "geometry": "true"
        }
        
        response = requests.post(url, json=body, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception:
        return None


def calculate_eta_with_service_time(
    route: List[int],
    locations: List[Dict],
    distance_matrix: np.ndarray = None,
    avg_speed_kmh: float = 40.0,
    service_time_min: float = 10.0,
    start_time: datetime = None
) -> pd.DataFrame:
    """Calculate ETA for each stop with service time"""
    if start_time is None:
        start_time = datetime.now()
    
    current_time = start_time
    eta_data = []
    
    # Start point
    eta_data.append({
        'Stop': 0,
        'Type': '🏠 START',
        'Location': locations[route[0]]['name'],
        'Arrival': current_time.strftime('%H:%M'),
        'Departure': current_time.strftime('%H:%M'),
        'Service Time': '0 min',
        'Cumulative Time': '0 min'
    })
    
    cumulative_minutes = 0
    
    # Each delivery stop
    for i in range(len(route) - 1):
        from_idx = route[i]
        to_idx = route[i + 1]
        
        if distance_matrix is not None:
            distance_km = distance_matrix[from_idx][to_idx]
        else:
            from_loc = locations[from_idx]
            to_loc = locations[to_idx]
            distance_km = haversine_distance(
                from_loc['lat'], from_loc['lng'],
                to_loc['lat'], to_loc['lng']
            )
        
        travel_time_minutes = (distance_km / avg_speed_kmh) * 60
        current_time += timedelta(minutes=travel_time_minutes)
        arrival_time = current_time
        
        current_time += timedelta(minutes=service_time_min)
        departure_time = current_time
        
        cumulative_minutes += travel_time_minutes + service_time_min
        
        eta_data.append({
            'Stop': i + 1,
            'Type': '📦 Delivery',
            'Location': locations[to_idx]['name'],
            'Arrival': arrival_time.strftime('%H:%M'),
            'Departure': departure_time.strftime('%H:%M'),
            'Service Time': f'{service_time_min:.0f} min',
            'Cumulative Time': f'{cumulative_minutes:.0f} min'
        })
    
    # Return to depot
    from_idx = route[-1]
    to_idx = route[0]
    
    if distance_matrix is not None:
        distance_km = distance_matrix[from_idx][to_idx]
    else:
        from_loc = locations[from_idx]
        to_loc = locations[to_idx]
        distance_km = haversine_distance(
            from_loc['lat'], from_loc['lng'],
            to_loc['lat'], to_loc['lng']
        )
    
    travel_time_minutes = (distance_km / avg_speed_kmh) * 60
    current_time += timedelta(minutes=travel_time_minutes)
    cumulative_minutes += travel_time_minutes
    
    eta_data.append({
        'Stop': len(route),
        'Type': '🏠 RETURN',
        'Location': locations[route[0]]['name'],
        'Arrival': current_time.strftime('%H:%M'),
        'Departure': current_time.strftime('%H:%M'),
        'Service Time': '0 min',
        'Cumulative Time': f'{cumulative_minutes:.0f} min'
    })
    
    return pd.DataFrame(eta_data)


# =============================================================================
# MAIN DISPLAY FUNCTION - WITH VISUALIZER OPTION
# =============================================================================
def display_results(
    results: Dict[str, Any],
    locations: List[Dict],
    distance_matrix: np.ndarray = None,
    visualizer = None  # ✅ Optional TSPVisualizer instance
) -> None:
    """
    Display optimization results
    
    Args:
        results: Results dict from solver
        locations: Location data
        distance_matrix: Distance matrix
        visualizer: Optional TSPVisualizer instance for advanced charts
    """
    if not results or not results.get('success', False):
        st.error("❌ Optimization failed or no results to display")
        if results and results.get('error_message'):
            st.error(f"Error: {results['error_message']}")
        return
    
    algorithm_name = results.get('algorithm', 'Unknown')
    
    # =========================================================================
    # 1. KEY METRICS
    # =========================================================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Best Distance", f"{results.get('best_distance', 0):.2f} km")
    with col2:
        st.metric("⏱️ Runtime", f"{results.get('runtime_seconds', 0):.2f}s")
    with col3:
        st.metric("🔄 Iterations", f"{results.get('num_iterations', 0):,}")
    with col4:
        improvement = results.get('improvement_percentage', 0)
        st.metric("📈 Improvement", f"{improvement:.1f}%")
    st.write("---")

    # ✅ EXTRACT ROUTE FIRST (BEFORE USING IT)
    route = results.get('best_route', [])
    # =========================================================================
    # 2. ROUTE SEQUENCE TABLE
    # =========================================================================
    st.write("---")
    st.write("### 📋 Route Sequence")

    if route and locations:
        route_data = []
        for i, idx in enumerate(route):
            loc = locations[idx]
            route_data.append({
                'Stop': i,
                'Type': '🏠 Depot' if loc.get('isStart') else '📦 Delivery',
                'Location': loc['name'],
                'Address': loc.get('address', 'N/A'),
                'Coordinates': f"{loc['lat']:.6f}, {loc['lng']:.6f}"
            })
        
        depot = locations[route[0]]
        route_data.append({
            'Stop': len(route),
            'Type': '🏠 Return',
            'Location': depot['name'],
            'Address': depot.get('address', 'N/A'),
            'Coordinates': f"{depot['lat']:.6f}, {depot['lng']:.6f}"
        })
        
        df_route = pd.DataFrame(route_data)
        st.dataframe(df_route, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # 3. ETA CALCULATOR
    # =========================================================================
    if route and locations:
        st.write("---")
        st.write("### ⏱️ Estimated Time of Arrival (ETA)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_speed = st.slider("🚗 Average Speed (km/h)", 20, 80, 40, 5)
        with col2:
            service_time = st.slider("⏱️ Service Time (min)", 5, 30, 10, 5)
        with col3:
            start_hour = st.selectbox("🕐 Start Time", list(range(6, 20)), 2,
                                     format_func=lambda x: f"{x:02d}:00")
        with col4:
            st.metric("📏 Total Distance", f"{results.get('best_distance', 0):.2f} km")
        
        start_time = datetime.now().replace(hour=start_hour, minute=0, second=0)
        
        eta_df = calculate_eta_with_service_time(
            route, locations, distance_matrix,
            avg_speed_kmh=avg_speed,
            service_time_min=service_time,
            start_time=start_time
        )
        
        st.write("")
        total_time = float(eta_df.iloc[-1]['Cumulative Time'].split()[0])
        end_time = start_time + timedelta(minutes=total_time)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏰ Start", start_time.strftime('%H:%M'))
        with col2:
            st.metric("🏁 Finish", end_time.strftime('%H:%M'))
        with col3:
            st.metric("⏱️ Duration", f"{total_time/60:.1f} hrs")
        
        st.write("")
        st.dataframe(eta_df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # 4. CONVERGENCE PLOT - USE VISUALIZER IF AVAILABLE
    # =========================================================================
    if results.get('convergence_data'):
        st.write("---")
        st.write("### 📈 Algorithm Convergence")
        
        # ✅ TRY VISUALIZER FIRST
        plot_success = False
        
        if visualizer is not None:
            try:
                fig = visualizer.plot_convergence(
                    results['convergence_data'],
                    algorithm_name
                )
                st.plotly_chart(fig, use_container_width=True)
                plot_success = True
            except Exception as e:
                st.warning(f"⚠️ Visualizer failed: {str(e)}")        
    # =========================================================================
    # 5. ADVANCED VISUALIZATIONS (NEW - ONLY IF VISUALIZER PROVIDED)
    # =========================================================================
    if visualizer is not None:
        with st.expander("🎨 Advanced Visualizations", expanded=False):
            
            # Distance matrix heatmap
            if distance_matrix is not None:
                st.write("#### 🔥 Distance Matrix Heatmap")
                try:
                    location_names = [loc['name'] for loc in locations]
                    fig = visualizer.plot_distance_matrix(distance_matrix, location_names)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to plot distance matrix: {str(e)}")
            # Route Segment Analysis
            st.write("### 📊 Route Segment Analysis")
            if visualizer and route and distance_matrix is not None:
                try:
                    fig = visualizer.plot_route_segments(
                        route, 
                        distance_matrix,
                        locations
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to plot route segments: {str(e)}")
            else:
                st.info("📌 Route segment analysis will appear here after optimization")

            st.write("---")
            st.write("---")
    # =========================================================================
    # 6. DETAILED STATISTICS
    # =========================================================================
    with st.expander("📊 Detailed Statistics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Algorithm Parameters:**")
            st.write(f"- Algorithm: {algorithm_name}")
            st.write(f"- Best Distance: {results.get('best_distance', 0):.2f} km")
            st.write(f"- Initial Distance: {results.get('initial_distance', 0):.2f} km")
            st.write(f"- Runtime: {results.get('runtime_seconds', 0):.3f} seconds")
        
        with col2:
            st.write("**Performance Metrics:**")
            st.write(f"- Iterations: {results.get('num_iterations', 0):,}")
            st.write(f"- Locations: {len(route)}")
            st.write(f"- Improvement: {results.get('improvement_percentage', 0):.2f}%")


# =============================================================================
# COMPARISON DISPLAY
# =============================================================================
def display_comparison_results(
    ga_results: Dict[str, Any],
    pso_results: Dict[str, Any],
    locations: List[Dict] = None
) -> None:
    """Display comparison between GA and PSO"""
    st.write("### 🔄 Algorithm Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ga_dist = ga_results.get('best_distance', 0)
        pso_dist = pso_results.get('best_distance', 0)
        winner = "GA" if ga_dist < pso_dist else "PSO"
        st.metric("🎯 Best Distance", winner)
        st.caption(f"GA: {ga_dist:.2f} km")
        st.caption(f"PSO: {pso_dist:.2f} km")
    
    with col2:
        ga_time = ga_results.get('runtime_seconds', 0)
        pso_time = pso_results.get('runtime_seconds', 0)
        winner = "GA" if ga_time < pso_time else "PSO"
        st.metric("⏱️ Faster", winner)
        st.caption(f"GA: {ga_time:.2f}s")
        st.caption(f"PSO: {pso_time:.2f}s")
    
    with col3:
        ga_iter = ga_results.get('num_iterations', 0)
        pso_iter = pso_results.get('num_iterations', 0)
        st.metric("🔄 Iterations", "Comparison")
        st.caption(f"GA: {ga_iter:,}")
        st.caption(f"PSO: {pso_iter:,}")
    
    with col4:
        ga_imp = ga_results.get('improvement_percentage', 0)
        pso_imp = pso_results.get('improvement_percentage', 0)
        winner = "GA" if ga_imp > pso_imp else "PSO"
        st.metric("📈 Better Improvement", winner)
        st.caption(f"GA: {ga_imp:.1f}%")
        st.caption(f"PSO: {pso_imp:.1f}%")
    
    st.write("---")
    
    if ga_results.get('convergence_data') and pso_results.get('convergence_data'):
        fig = go.Figure()
        
        ga_conv = ga_results['convergence_data'].get('best', [])
        if ga_conv:
            fig.add_trace(go.Scatter(
                x=list(range(len(ga_conv))),
                y=ga_conv,
                mode='lines',
                name='GA',
                line=dict(color='#1976d2', width=3)
            ))
        
        pso_conv = pso_results['convergence_data'].get('best', [])
        if pso_conv:
            fig.add_trace(go.Scatter(
                x=list(range(len(pso_conv))),
                y=pso_conv,
                mode='lines',
                name='PSO',
                line=dict(color='#ff6b6b', width=3)
            ))
        
        fig.update_layout(
            title="Algorithm Convergence Comparison",
            xaxis_title="Iteration",
            yaxis_title="Distance (km)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)