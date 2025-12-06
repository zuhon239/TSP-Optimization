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
        initial_dist = results.get('initial_distance', 0)
        best_dist = results.get('best_distance', 0)
        if initial_dist > 0:  
            improvement = ((initial_dist - best_dist) / initial_dist) * 100
        else:
            improvement = 0.0
        
        st.metric(
            "📈 Improvement",
            f"{improvement:.1f}%",
            help=(
                "**📍 Initial Distance:**\n"
                "Total distance of the FIRST random route tried by the algorithm. "
                "This is the baseline starting point before any optimization.\n\n"
                "**🎯 Best Distance Found:**\n"
                "Total distance of the BEST route discovered after running the optimization algorithm. "
                "This is the optimized result after all iterations/generations.\n\n"
                "**📊 Calculation Formula:**\n"
                "((Initial - Best) / Initial) × 100\n\n"
                f"**💾 Results:**\n"
                f"• Initial: {initial_dist:.2f} km\n"
                f"• Best: {best_dist:.2f} km\n"
                f"• Saved: {initial_dist - best_dist:.2f} km\n"
                f"• Improvement: {improvement:.1f}%"
            )
        )
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
        st.dataframe(df_route, width='stretch', hide_index=True)
    
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
        st.dataframe(eta_df, width='stretch', hide_index=True)
    
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
                st.plotly_chart(fig, width='stretch')
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
                    st.plotly_chart(fig, width='stretch')
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
                    st.plotly_chart(fig, width='stretch')
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
            initial_dist = results.get('initial_distance', 0)
            best_dist = results.get('best_distance', 0)
            if initial_dist > 0:  
                improvement = ((initial_dist - best_dist) / initial_dist) * 100
            else:
                improvement = 0.0       
            st.write("📈 Improvement", f"{improvement:.1f}%")


# =============================================================================
# COMPARISON DISPLAY
# =============================================================================
def display_comparison_results(
    comparison_results: Dict[str, Any],
    locations: List[Dict] = None,
    distance_matrix: np.ndarray = None
) -> None:
    """
    Display comprehensive comparison between GA and PSO algorithms
    
    Args:
        comparison_results: Results dict from AlgorithmComparison class
        locations: Location data with coordinates and names
        distance_matrix: Optional distance matrix for visualization
    """
    import streamlit as st
    
    # Extract results
    ga_results = comparison_results.get('ga_results')
    pso_results = comparison_results.get('pso_results')
    comparison = comparison_results.get('comparison')
    
    # Validate both algorithms completed
    if not ga_results or not pso_results:
        st.error("❌ Invalid comparison data")
        
        if ga_results:
            st.write(f"GA Status: {ga_results.get('success', 'Unknown')}")
            if ga_results.get('error_message'):
                st.error(f"GA Error: {ga_results['error_message']}")
        else:
            st.write("GA Results: None")
            
        if pso_results:
            st.write(f"PSO Status: {pso_results.get('success', 'Unknown')}")
            if pso_results.get('error_message'):
                st.error(f"PSO Error: {pso_results['error_message']}")
        else:
            st.write("PSO Results: None")
        
        return
    
    # =========================================================================
    # SECTION A: Algorithm Configuration Comparison
    # =========================================================================
    st.subheader("⚙️ Algorithm Configuration")
    
    # Get parameters from top-level or from results
    ga_params = comparison_results.get('ga_params') or ga_results.get('parameters', {})
    pso_params = comparison_results.get('pso_params') or pso_results.get('parameters', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🧬 Genetic Algorithm Parameters**")
        ga_config_data_items = [
            ('Population Size', str(ga_params.get('population_size', 'N/A'))),
            ('Generations', str(ga_params.get('generations', 'N/A'))),
            ('Crossover Rate', str(f"{float(ga_params.get('crossover_probability', 0)):.2f}")),
            ('Mutation Rate', str(f"{float(ga_params.get('mutation_probability', 0)):.2f}")),
            ('Tournament Size', str(ga_params.get('tournament_size', 'N/A'))),
            ('Elite Size', str(ga_params.get('elite_size', 'N/A')))
        ]
        ga_config_df = pd.DataFrame(ga_config_data_items, columns=['Parameter', 'Value'])
        ga_config_df['Value'] = ga_config_df['Value'].astype(str)
        st.dataframe(ga_config_df, width='stretch')
    
    with col2:
        st.write("**🐝 Particle Swarm Optimization Parameters**")
        pso_config_data_items = [
            ('Swarm Size', str(pso_params.get('swarm_size', 'N/A'))),
            ('Max Iterations', str(pso_params.get('max_iterations', 'N/A'))),
            ('Inertia Weight (w)', str(f"{float(pso_params.get('w', 0)):.3f}")),
            ('Cognitive (c1)', str(f"{float(pso_params.get('c1', 0)):.3f}")),
            ('Social (c2)', str(f"{float(pso_params.get('c2', 0)):.3f}")),
            ('w_min', str(f"{float(pso_params.get('w_min', 0)):.3f}"))
        ]
        pso_config_df = pd.DataFrame(pso_config_data_items, columns=['Parameter', 'Value'])
        pso_config_df['Value'] = pso_config_df['Value'].astype(str)
        st.dataframe(pso_config_df, width='stretch')
    
    st.write("---")
    
    # =========================================================================
    # SECTION B: Performance Metrics Comparison
    # =========================================================================
    st.subheader("📊 Performance Metrics")
    
    # Always display metrics - no conditional
    # Metric cards with winner indicators
    col1, col2, col3, col4 = st.columns(4)
    
    # Best Distance
    with col1:
        ga_dist = ga_results.get('best_distance', float('inf'))
        pso_dist = pso_results.get('best_distance', float('inf'))
        winner = "🏆 GA" if ga_dist < pso_dist else "🏆 PSO"
        st.metric(
            "Best Distance (km)",
            f"{min(ga_dist, pso_dist):.2f}",
            delta=f"{winner}"
        )
        st.caption(f"GA: {ga_dist:.2f}km | PSO: {pso_dist:.2f}km")
    
    # Runtime
    with col2:
        ga_time = ga_results.get('runtime_seconds', float('inf'))
        pso_time = pso_results.get('runtime_seconds', float('inf'))
        winner = "🏆 GA" if ga_time < pso_time else "🏆 PSO"
        st.metric(
            "Runtime (seconds)",
            f"{min(ga_time, pso_time):.4f}",
            delta=f"{winner}"
        )
        st.caption(f"GA: {ga_time:.4f}s | PSO: {pso_time:.4f}s")
    
    # Improvement Percentage
    with col3:
        # Calculate improvement from initial and best distance
        ga_initial = ga_results.get('initial_distance', 0)
        ga_best = ga_results.get('best_distance', 0)
        ga_imp = ((ga_initial - ga_best) / ga_initial * 100) if ga_initial > 0 else 0
        
        pso_initial = pso_results.get('initial_distance', 0)
        pso_best = pso_results.get('best_distance', 0)
        pso_imp = ((pso_initial - pso_best) / pso_initial * 100) if pso_initial > 0 else 0
        
        winner = "🏆 GA" if ga_imp > pso_imp else "🏆 PSO"
        st.metric(
            "Improvement (%)",            
            f"{max(ga_imp, pso_imp):.1f}%",
            delta=f"{winner}"
        )
        st.caption(f"GA: {ga_imp:.1f}% | PSO: {pso_imp:.1f}%")
    
    # Iterations/Convergence
    with col4:
        ga_iter = ga_results.get('num_iterations', 0)
        pso_iter = pso_results.get('num_iterations', 0)
        st.metric(
            "Total Iterations",
            f"GA: {ga_iter} | PSO: {pso_iter}",
            delta=f"Difference: {abs(ga_iter - pso_iter)}"
        )
    
    st.write("---")
    
    # =========================================================================
    # SECTION C: Convergence Chart
    # =========================================================================
    st.subheader("📈 Convergence Comparison")
    
    ga_convergence = ga_results.get('convergence_data', {})
    pso_convergence = pso_results.get('convergence_data', {})
    ga_success = ga_results.get('success', False)
    pso_success = pso_results.get('success', False)
    
    if ga_convergence.get('best_distances') and pso_convergence.get('best_distances'):
        fig = go.Figure()
        
        # GA convergence
        ga_dists = ga_convergence['best_distances']
        fig.add_trace(go.Scatter(
            x=list(range(len(ga_dists))),
            y=ga_dists,
            mode='lines',
            name='GA',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>GA</b><br>Iteration: %{x}<br>Distance: %{y:.2f}km<extra></extra>'
        ))
        
        # PSO convergence
        pso_dists = pso_convergence['best_distances']
        fig.add_trace(go.Scatter(
            x=list(range(len(pso_dists))),
            y=pso_dists,
            mode='lines',
            name='PSO',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>PSO</b><br>Iteration: %{x}<br>Distance: %{y:.2f}km<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="Algorithm Convergence Comparison",
            xaxis_title="Iteration / Generation",
            yaxis_title="Best Distance Found (km)",
            hovermode='x unified',
            template='plotly_white',
            height=450,
            showlegend=True,
            legend=dict(x=0.7, y=0.95)
        )
        
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Convergence data not available for both algorithms")
    
    st.write("---")
    
    # =========================================================================
    # SECTION D: Route Quality Comparison
    # =========================================================================
    st.subheader("🛣️ Route Quality Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🧬 Genetic Algorithm Route**")
        ga_route = ga_results.get('best_route', [])
        if locations and ga_route:
            ga_route_names = [locations[i].get('name', f'City {i}') for i in ga_route]
            st.caption(" → ".join(ga_route_names) + " → " + ga_route_names[0])
            st.caption(f"Route indices: {ga_route}")
        st.metric("Distance", f"{ga_results.get('best_distance', 0):.2f} km")
    
    with col2:
        st.write("**🐝 PSO Route**")
        pso_route = pso_results.get('best_route', [])
        if locations and pso_route:
            pso_route_names = [locations[i].get('name', f'City {i}') for i in pso_route]
            st.caption(" → ".join(pso_route_names) + " → " + pso_route_names[0])
            st.caption(f"Route indices: {pso_route}")
        st.metric("Distance", f"{pso_results.get('best_distance', 0):.2f} km")
    
    st.write("---")
    # =========================================================================
    # SECTION D.2: SIDE-BY-SIDE MAP COMPARISON - WITH ANIMATION & NUMBERING
    # =========================================================================
    st.subheader("🗺️ Visual Route Comparison")

    if locations and ga_route and pso_route:
        try:
            import folium
            from folium.plugins import AntPath
            from streamlit_folium import st_folium
            
            # Get OpenRouteService API key
            openroute_key = os.getenv('OPENROUTE_API_KEY')
            
            # =====================================================================
            # CACHE KEY - Unique per route comparison
            # =====================================================================
            ga_route_key = "_".join(map(str, ga_route))
            pso_route_key = "_".join(map(str, pso_route))
            cache_key_ga = f"comparison_ga_{ga_route_key}"
            cache_key_pso = f"comparison_pso_{pso_route_key}"
            
            # =====================================================================
            # FETCH REAL ROAD COORDINATES (WITH CACHING)
            # =====================================================================
            ga_real_coords = None
            pso_real_coords = None
            
            # Check if already cached
            if cache_key_ga in st.session_state:
                ga_real_coords = st.session_state[cache_key_ga]
            
            if cache_key_pso in st.session_state:
                pso_real_coords = st.session_state[cache_key_pso]
            
            # Only fetch if not cached
            if openroute_key and (ga_real_coords is None or pso_real_coords is None):
                with st.spinner("🛣️ Fetching real road routes..."):
                    # GA route (if not cached)
                    if ga_real_coords is None:
                        try:
                            ga_coords_api = [[locations[i]['lng'], locations[i]['lat']] for i in ga_route]
                            ga_coords_api.append(ga_coords_api[0])
                            ga_directions = get_openroute_directions(ga_coords_api, openroute_key)
                            
                            if ga_directions and ga_directions.get('features'):
                                geometry = ga_directions['features'][0]['geometry']
                                if geometry['type'] == 'LineString':
                                    ga_real_coords = [[c[1], c[0]] for c in geometry['coordinates']]
                                    st.session_state[cache_key_ga] = ga_real_coords  # ✅ CACHE IT
                        except Exception as e:
                            st.session_state[cache_key_ga] = None  # Cache failure
                    
                    # PSO route (if not cached)
                    if pso_real_coords is None:
                        try:
                            pso_coords_api = [[locations[i]['lng'], locations[i]['lat']] for i in pso_route]
                            pso_coords_api.append(pso_coords_api[0])
                            pso_directions = get_openroute_directions(pso_coords_api, openroute_key)
                            
                            if pso_directions and pso_directions.get('features'):
                                geometry = pso_directions['features'][0]['geometry']
                                if geometry['type'] == 'LineString':
                                    pso_real_coords = [[c[1], c[0]] for c in geometry['coordinates']]
                                    st.session_state[cache_key_pso] = pso_real_coords  
                        except Exception as e:
                            st.session_state[cache_key_pso] = None  # Cache failure
            
            # Calculate center
            lats = [locations[i]['lat'] for i in ga_route]
            lngs = [locations[i]['lng'] for i in ga_route]
            center_lat = sum(lats) / len(lats)
            center_lng = sum(lngs) / len(lngs)
            
            col1, col2 = st.columns(2)
            
            # =====================================================================
            # GA MAP 
            # =====================================================================
            with col1:
                st.write("**🧬 Genetic Algorithm Route**")
                
                ga_map = folium.Map(
                    location=[center_lat, center_lng],
                    zoom_start=12,
                    tiles='OpenStreetMap',
                    control_scale=True
                )
                
                # DRAW ANIMATED ROUTE
                if ga_real_coords:
                    AntPath(
                        locations=ga_real_coords,
                        color='#1976d2',
                        weight=6,
                        opacity=0.8,
                        delay=800,
                        dash_array=[10, 20],
                        pulse_color='#ff6b6b'
                    ).add_to(ga_map)
                else:
                    # Fallback: straight lines with animation
                    route_coords = [[locations[i]['lat'], locations[i]['lng']] for i in ga_route]
                    route_coords.append(route_coords[0])
                    AntPath(
                        locations=route_coords,
                        color='#1976d2',
                        weight=5,
                        opacity=0.7,
                        delay=1000,
                        dash_array=[15, 25],
                        pulse_color='#ff6b6b'
                    ).add_to(ga_map)
                
                # ✅ ADD NUMBERED MARKERS
                for route_order, loc_idx in enumerate(ga_route):
                    loc = locations[loc_idx]
                    is_start = route_order == 0
                    
                    if is_start:
                        icon_html = f'''
                            <div style="
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                width: 35px;
                                height: 35px;
                                background-color: #ef5350;
                                border: 3px solid white;
                                border-radius: 50%;
                                font-weight: bold;
                                font-size: 16px;
                                color: white;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                            ">🏠</div>
                        '''
                    else:
                        icon_html = f'''
                            <div style="
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                width: 30px;
                                height: 30px;
                                background-color: #1976d2;
                                border: 3px solid white;
                                border-radius: 50%;
                                font-weight: bold;
                                font-size: 14px;
                                color: white;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                            ">{route_order}</div>
                        '''
                    
                    popup_html = f"""
                    <div style="min-width: 200px; font-family: Arial;">
                        <h4 style="margin: 0 0 10px 0; color: #1976d2;">
                            {'🏠 Depot' if is_start else f'📦 Stop {route_order}'}
                        </h4>
                        <p><b>📍</b> {loc['name']}</p>
                        <p><b>🔢 Order:</b> {route_order} / {len(ga_route)-1}</p>
                    </div>
                    """
                    
                    folium.Marker(
                        location=[loc['lat'], loc['lng']],
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=f"{'🏠 ' if is_start else ''}{loc['name']} ({route_order})",
                        icon=folium.DivIcon(html=icon_html)
                    ).add_to(ga_map)
                
                st_folium(ga_map, width=400, height=450, key="ga_comp_map")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("📏 Distance", f"{ga_results.get('best_distance', 0):.2f} km")
                with col_b:
                    st.metric("📦 Stops", len(ga_route) - 1)
            
            # =====================================================================
            # PSO MAP 
            # =====================================================================
            with col2:
                st.write("**🐝 Particle Swarm Optimization Route**")
                
                pso_map = folium.Map(
                    location=[center_lat, center_lng],
                    zoom_start=12,
                    tiles='OpenStreetMap',
                    control_scale=True
                )
                
                # ✅ DRAW ANIMATED ROUTE
                if pso_real_coords:
                    AntPath(
                        locations=pso_real_coords,
                        color='#ff7f0e',
                        weight=6,
                        opacity=0.8,
                        delay=800,
                        dash_array=[10, 20],
                        pulse_color='#66ff66'
                    ).add_to(pso_map)
                else:
                    # Fallback: straight lines with animation
                    route_coords = [[locations[i]['lat'], locations[i]['lng']] for i in pso_route]
                    route_coords.append(route_coords[0])
                    AntPath(
                        locations=route_coords,
                        color='#ff7f0e',
                        weight=5,
                        opacity=0.7,
                        delay=1000,
                        dash_array=[15, 25],
                        pulse_color='#66ff66'
                    ).add_to(pso_map)
                
                # ✅ ADD NUMBERED MARKERS
                for route_order, loc_idx in enumerate(pso_route):
                    loc = locations[loc_idx]
                    is_start = route_order == 0
                    
                    if is_start:
                        icon_html = f'''
                            <div style="
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                width: 35px;
                                height: 35px;
                                background-color: #ef5350;
                                border: 3px solid white;
                                border-radius: 50%;
                                font-weight: bold;
                                font-size: 16px;
                                color: white;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                            ">🏠</div>
                        '''
                    else:
                        icon_html = f'''
                            <div style="
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                width: 30px;
                                height: 30px;
                                background-color: #ff7f0e;
                                border: 3px solid white;
                                border-radius: 50%;
                                font-weight: bold;
                                font-size: 14px;
                                color: white;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                            ">{route_order}</div>
                        '''
                    
                    popup_html = f"""
                    <div style="min-width: 200px; font-family: Arial;">
                        <h4 style="margin: 0 0 10px 0; color: #ff7f0e;">
                            {'🏠 Depot' if is_start else f'📦 Stop {route_order}'}
                        </h4>
                        <p><b>📍</b> {loc['name']}</p>
                        <p><b>🔢 Order:</b> {route_order} / {len(pso_route)-1}</p>
                    </div>
                    """
                    
                    folium.Marker(
                        location=[loc['lat'], loc['lng']],
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=f"{'🏠 ' if is_start else ''}{loc['name']} ({route_order})",
                        icon=folium.DivIcon(html=icon_html)
                    ).add_to(pso_map)
                
                st_folium(pso_map, width=400, height=450, key="pso_comp_map")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("📏 Distance", f"{pso_results.get('best_distance', 0):.2f} km")
                with col_b:
                    st.metric("📦 Stops", len(pso_route) - 1)
            
            # =====================================================================
            # ROUTE DIFFERENCES ANALYSIS 
            # =====================================================================
            st.write("")
            st.write("**📊 Route Comparison Analysis:**")
            
            if ga_route == pso_route:
                st.success("✅ Both algorithms found the **exact same route**!")
            else:
                # Compare routes by checking common edges (segments)
                # Use sets to handle routes that visit same cities but in different order
                
                def get_edges(route):
                    """Convert route to set of undirected edges (segments)"""
                    edges = set()
                    for i in range(len(route) - 1):
                        # Store as sorted tuple to ignore direction
                        edge = tuple(sorted([route[i], route[i+1]]))
                        edges.add(edge)
                    return edges
                
                ga_edges = get_edges(ga_route)
                pso_edges = get_edges(pso_route)
                
                # Common edges are those that appear in both routes
                common_edges = ga_edges.intersection(pso_edges)
                same_segments = len(common_edges)
                
                # Total possible edges (using GA route length as reference)
                total_segments = len(ga_route) - 1
                different_segments = total_segments - same_segments
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🔄 Different Segments", f"{different_segments}/{total_segments}")
                with col2:
                    similarity = (same_segments / total_segments * 100) if total_segments > 0 else 0
                    st.metric(
                        "📊 Route Similarity",
                        f"{similarity:.1f}%",
                        help=(
                            "**🔍 What is measured:**\n"
                            "Compares the delivery segments (edges) between GA and PSO routes. "
                            "A segment is a direct trip from one location to another.\n\n"
                            "**📍 How it's calculated:**\n"
                            "1. Extract all delivery segments from GA route\n"
                            "2. Extract all delivery segments from PSO route\n"
                            "3. Count how many segments are identical (regardless of direction)\n"
                            "4. Calculate percentage: (Common Segments / Total Segments) × 100\n\n"
                            "**💡 What it means:**\n"
                            "• **100%:** Both algorithms chose identical delivery segments → Same route!\n"
                            "• **50%:** Half the segments match → Different route strategies\n"
                            "• **0%:** No common segments → Completely different routes\n\n"
                            "**📊 Example:**\n"
                            f"• Common segments: {same_segments}/{total_segments}\n"
                            f"• Different segments: {different_segments}/{total_segments}\n"
                            f"• Similarity: {similarity:.1f}%\n\n"
                            "**🎯 Interpretation:**\n"
                            "High similarity suggests algorithms converged to similar solutions. "
                            "Low similarity shows different optimization strategies but both may be optimal."
                        )
                    )
                with col3:
                    distance_diff = abs(ga_results.get('best_distance', 0) - pso_results.get('best_distance', 0))
                    st.metric("📏 Distance Gap", f"{distance_diff:.2f} km")
            
        except ImportError as e:
            st.error(f"❌ Required library not installed: {str(e)}")
            st.info("Run: `pip install folium streamlit-folium`")
        except Exception as e:
            st.error(f"❌ Failed to render comparison maps: {str(e)}")
            with st.expander("🐛 Debug Info"):
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("📍 Location data or routes not available for map comparison")

    st.write("---")

    
    # =========================================================================
    # SECTION E: Statistical Summary Table
    # =========================================================================
    st.subheader("📋 Detailed Statistical Summary")
    
    # Build summary data
    summary_data = {
        'Metric': [
            'Algorithm',
            'Best Distance (km)',
            'Initial Distance (km)',
            'Worst Distance (km)',
            'Mean Distance (km)',
            'Std. Deviation (km)',
            'Runtime (seconds)',
            'Iterations',
            'Improvement (%)',
            'Success'
        ],
        'Genetic Algorithm': [
            str('GA'),
            str(f"{ga_results.get('best_distance', 0):.2f}"),
            str(f"{ga_convergence.get('best_distances', [0])[0]:.2f}" if ga_convergence.get('best_distances') else 'N/A'),
            str(f"{max(ga_convergence.get('best_distances', [0])):.2f}" if ga_convergence.get('best_distances') else 'N/A'),
            str(f"{np.mean(ga_convergence.get('best_distances', [0])):.2f}" if ga_convergence.get('best_distances') else 'N/A'),
            str(f"{np.std(ga_convergence.get('best_distances', [0])):.2f}" if ga_convergence.get('best_distances') and len(ga_convergence.get('best_distances', [])) > 1 else 'N/A'),
            str(f"{ga_results.get('runtime_seconds', 0):.4f}"),
            str(f"{ga_results.get('num_iterations', 0)}"),
            str(f"{comparison.get('ga_improvement_pct', 0):.1f}%" if comparison else 'N/A'),
            str("✅ Yes" if ga_success else "❌ No")
        ],
        'Particle Swarm Optimization': [
            str('PSO'),
            str(f"{pso_results.get('best_distance', 0):.2f}"),
            str(f"{pso_convergence.get('best_distances', [0])[0]:.2f}" if pso_convergence.get('best_distances') else 'N/A'),
            str(f"{max(pso_convergence.get('best_distances', [0])):.2f}" if pso_convergence.get('best_distances') else 'N/A'),
            str(f"{np.mean(pso_convergence.get('best_distances', [0])):.2f}" if pso_convergence.get('best_distances') else 'N/A'),
            str(f"{np.std(pso_convergence.get('best_distances', [0])):.2f}" if pso_convergence.get('best_distances') and len(pso_convergence.get('best_distances', [])) > 1 else 'N/A'),
            str(f"{pso_results.get('runtime_seconds', 0):.4f}"),
            str(f"{pso_results.get('num_iterations', 0)}"),
            str(f"{comparison.get('pso_improvement_pct', 0):.1f}%" if comparison else 'N/A'),
            str("✅ Yes" if pso_success else "❌ No")
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width='stretch', use_container_width=True)
    
    st.write("---")
    

