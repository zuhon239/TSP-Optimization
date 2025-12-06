"""
TSP Optimization Web Application
Correct architecture: src (algorithms) → components (UI) → app.py (integration)

Authors: All team members
Final integration by: Hoàng (Team Leader)
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import logging
import traceback
import time
from typing import List, Dict, Any, Optional
from math import radians, cos, sin, asin, sqrt

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🚛 TSP Optimization",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
components_path = os.path.join(current_dir, 'components')

# Add paths to sys.path
for path in [current_dir, src_path, components_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Global status tracking
MODULES_STATUS = {
    # Source modules (algorithms)
    'config': False,
    'TSPSolver': False,
    'GASolver': False,
    'PSOSolver': False,
    'GeoUtils': False,
    'AlgorithmComparison': False,
    'Visualizer': False,
    # Component modules (UI)
    'google_maps_ui': False,
    'sidebar': False,
    'results_display': False
}

# Application configuration
APP_CONFIG = {
    'title': '🚛 TSP Optimization System',
    'version': '1.0',
    'description': 'Giải bài toán Travelling Salesman Problem (TSP) bằng Genetic Algorithm và Particle Swarm Optimization'
}

# Default algorithm parameters
GA_CONFIG = {
    'population_size': 50,
    'generations': 100,
    'crossover_probability': 0.8,
    'mutation_probability': 0.05,
    'tournament_size': 3,
    'elite_size': 2
}

PSO_CONFIG = {
    'swarm_size': 30,
    'max_iterations': 100,
    'w': 0.729,
    'c1': 1.494,
    'c2': 1.494
}

# Initialize import logs storage
if 'import_logs' not in st.session_state:
    st.session_state.import_logs = []

# =============================================================================
# 1. IMPORT SOURCE MODULES (ALGORITHMS) - SILENT MODE
# =============================================================================
def import_src_modules():
    """Import all source modules (algorithms and core logic)"""
    import_logs = []
    
    # Import config
    try:
        import config
        MODULES_STATUS['config'] = True
        import_logs.append(("✅", "Config loaded"))
    except ImportError as e:
        config = None
        MODULES_STATUS['config'] = False
        import_logs.append(("⚠️", f"Config not available: {e}"))
    
    # Import TSP Solver base class
    try:
        from src.tsp_solver import TSPSolver
        MODULES_STATUS['TSPSolver'] = True
        import_logs.append(("✅", "TSPSolver (base class) loaded"))
    except ImportError as e:
        TSPSolver = None
        MODULES_STATUS['TSPSolver'] = False
        import_logs.append(("⚠️", f"TSPSolver not available: {e}"))
    
    # Import Genetic Algorithm
    try:
        from src.ga_solver import GASolver
        MODULES_STATUS['GASolver'] = True
        import_logs.append(("✅", "GASolver (Genetic Algorithm) loaded"))
    except ImportError as e:
        GASolver = None
        MODULES_STATUS['GASolver'] = False
        import_logs.append(("⚠️", f"GASolver not available: {e}"))
    
    # Import Particle Swarm Optimization
    try:
        from src.pso_solver import PSOSolver
        MODULES_STATUS['PSOSolver'] = True
        import_logs.append(("✅", "PSOSolver (Particle Swarm Optimization) loaded"))
    except ImportError as e:
        PSOSolver = None
        MODULES_STATUS['PSOSolver'] = False
        import_logs.append(("⚠️", f"PSOSolver not available: {e}"))
    
    # Import Google Maps utilities
    try:
        from src.geo_utils import GeoUtils
        MODULES_STATUS['GeoUtils'] = True
        import_logs.append(("✅", "GeoUtils (Google Maps API) loaded"))
    except ImportError as e:
        GeoUtils = None
        MODULES_STATUS['GeoUtils'] = False
        import_logs.append(("⚠️", f"GeoUtils not available: {e}"))
    
    # Import Algorithm Comparison
    try:
        from src.comparison import AlgorithmComparison
        MODULES_STATUS['AlgorithmComparison'] = True
        import_logs.append(("✅", "AlgorithmComparison loaded"))
    except ImportError as e:
        AlgorithmComparison = None
        MODULES_STATUS['AlgorithmComparison'] = False
        import_logs.append(("⚠️", f"AlgorithmComparison not available: {e}"))
    
    # Import Visualizer
    try:
        from src.visualizer import TSPVisualizer
        MODULES_STATUS['Visualizer'] = True
        import_logs.append(("✅", "TSPVisualizer loaded"))
    except ImportError as e:
        TSPVisualizer = None
        MODULES_STATUS['Visualizer'] = False
        import_logs.append(("⚠️", f"TSPVisualizer not available: {e}"))
    
    # Store logs for debug view
    st.session_state.import_logs.extend(import_logs)
    
    return {
        'config': config,
        'TSPSolver': TSPSolver,
        'GASolver': GASolver,
        'PSOSolver': PSOSolver,
        'GeoUtils': GeoUtils,
        'AlgorithmComparison': AlgorithmComparison,
        'TSPVisualizer': TSPVisualizer
    }

# =============================================================================
# 2. IMPORT COMPONENT MODULES (UI) - SILENT MODE
# =============================================================================
def import_component_modules():
    """Import all UI component modules"""
    import_logs = []
    
    # Import Google Maps UI
    try:
        from components.google_maps_ui import render_integrated_map, validate_locations
        MODULES_STATUS['google_maps_ui'] = True
        import_logs.append(("✅", "Google Maps UI component loaded"))
    except ImportError as e:
        render_integrated_map = None
        validate_locations = None
        MODULES_STATUS['google_maps_ui'] = False
        import_logs.append(("⚠️", f"Google Maps UI not available: {e}"))
    
    # Import Sidebar UI
    try:
        from components.sidebar import render_sidebar
        MODULES_STATUS['sidebar'] = True
        import_logs.append(("✅", "Sidebar component loaded"))
    except ImportError as e:
        render_sidebar = None
        MODULES_STATUS['sidebar'] = False
        import_logs.append(("⚠️", f"Sidebar component not available: {e}"))
    
    # Import Results Display UI
    try:
        from components.results_display import display_results, display_comparison_results
        MODULES_STATUS['results_display'] = True
        import_logs.append(("✅", "Results Display component loaded"))
    except ImportError as e:
        display_results, display_comparison_results = None, None
        MODULES_STATUS['results_display'] = False
        import_logs.append(("⚠️", f"Results Display component not available: {e}"))
    
    st.session_state.import_logs.extend(import_logs)
    
    return {
        'render_integrated_map': render_integrated_map,
        'validate_locations': validate_locations,
        'render_sidebar': render_sidebar,
        'display_results': display_results,
        'display_comparison_results': display_comparison_results
    }

# =============================================================================
# 3. INITIALIZE APPLICATION
# =============================================================================
# Import all modules
src_modules = import_src_modules()
component_modules = import_component_modules()

# Extract modules for easy access
config = src_modules['config']
TSPSolver = src_modules['TSPSolver']
GASolver = src_modules['GASolver']
PSOSolver = src_modules['PSOSolver']
GeoUtils = src_modules['GeoUtils']
AlgorithmComparison = src_modules['AlgorithmComparison']
TSPVisualizer = src_modules['TSPVisualizer']
render_integrated_map = component_modules['render_integrated_map']
validate_locations = component_modules['validate_locations']
render_sidebar = component_modules['render_sidebar']
display_results = component_modules['display_results']
display_comparison_results = component_modules['display_comparison_results']

# Google Maps API configuration
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# =============================================================================
# 4. SESSION STATE INITIALIZATION
# =============================================================================
if 'clicked_locations' not in st.session_state:
    st.session_state.clicked_locations = []

if 'locations' not in st.session_state:
    st.session_state.locations = []

if 'location_counter' not in st.session_state:
    st.session_state.location_counter = 1

if 'distance_matrix' not in st.session_state:
    st.session_state.distance_matrix = None

if 'optimized_route' not in st.session_state:
    st.session_state.optimized_route = None

if 'route_coords' not in st.session_state:
    st.session_state.route_coords = None

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

if 'geo_utils' not in st.session_state:
    if GeoUtils and MODULES_STATUS.get('GeoUtils', False):
        try:
            st.session_state.geo_utils = GeoUtils()
        except Exception:
            st.session_state.geo_utils = None
    else:
        st.session_state.geo_utils = None

if 'last_map_center' not in st.session_state:
    st.session_state.last_map_center = None

if 'last_map_zoom' not in st.session_state:
    st.session_state.last_map_zoom = None

# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================
def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def get_map_center_and_zoom():
    """Intelligently determine map center based on clicked locations"""
    # Priority 1: Use last clicked position
    if st.session_state.last_map_center:
        return st.session_state.last_map_center, st.session_state.get('last_map_zoom', 15)
    
    # Priority 2: Center around existing locations
    if st.session_state.clicked_locations:
        lats = [loc['lat'] for loc in st.session_state.clicked_locations]
        lngs = [loc['lng'] for loc in st.session_state.clicked_locations]
        
        center_lat = sum(lats) / len(lats)
        center_lng = sum(lngs) / len(lngs)
        
        # Calculate zoom based on spread
        lat_range = max(lats) - min(lats) if len(lats) > 1 else 0
        lng_range = max(lngs) - min(lngs) if len(lngs) > 1 else 0
        max_range = max(lat_range, lng_range)
        
        if max_range < 0.01:
            zoom = 15
        elif max_range < 0.05:
            zoom = 13
        elif max_range < 0.1:
            zoom = 12
        else:
            zoom = 11
        
        return {'lat': center_lat, 'lng': center_lng}, zoom
    
    # Priority 3: Default center (Ho Chi Minh City)
    return {'lat': 10.762622, 'lng': 106.660172}, 13

def run_genetic_algorithm(distance_matrix, config_params):
    """Run Genetic Algorithm with enhanced error handling"""
    if not GASolver or not MODULES_STATUS.get('GASolver', False):
        raise Exception("Genetic Algorithm not available")
    
    try:
        ga_params = config_params.get('ga_params', GA_CONFIG)
        
        # Create solver with locations
        solver = GASolver(
            distance_matrix=distance_matrix,
            locations=st.session_state.locations,
            population_size=ga_params.get('population_size', 50),
            generations=ga_params.get('generations', 100),
            crossover_prob=ga_params.get('crossover_probability', 0.8),
            mutation_prob=ga_params.get('mutation_probability', 0.05),
            tournament_size=ga_params.get('tournament_size', 3),
            elite_size=ga_params.get('elite_size', 2)
        )
        
        result = solver.solve_with_stats()
        
        return result
        
    except Exception as e:
        raise e

def run_pso_algorithm(distance_matrix, config_params):
    """Run PSO Algorithm with enhanced error handling"""
    if not PSOSolver or not MODULES_STATUS.get('PSOSolver', False):
        raise Exception("PSO Algorithm not available")
    
    try:
        pso_params = config_params.get('pso_params', PSO_CONFIG)
        
        # Create solver with locations
        solver = PSOSolver(
            distance_matrix=distance_matrix,
            locations=st.session_state.locations,
            swarm_size=pso_params.get('swarm_size', 30),
            max_iterations=pso_params.get('max_iterations', 100),
            w=pso_params.get('w', 0.729),
            c1=pso_params.get('c1', 1.494),
            c2=pso_params.get('c2', 1.494)
        )
        
        result = solver.solve_with_stats()
        
        return result
        
    except Exception as e:
        raise e

# =============================================================================
# 6. UI COMPONENTS
# =============================================================================
def render_header():
    """Render application header with collapsible debug info"""
    st.title(APP_CONFIG['title'])
    st.markdown("""
    **Giải bài toán Travelling Salesman Problem (TSP) bằng Genetic Algorithm (GA) và Particle Swarm Optimization (PSO)**
    
    Ứng dụng tối ưu hóa lộ trình giao hàng sử dụng thuật toán tiến hóa và bầy đàn.
    """)
    
    # ✅ HIDE DEBUG IN COLLAPSIBLE EXPANDER
    with st.expander("🔧 Developer Debug Panel", expanded=False):
        st.write("### 📊 Module Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🧠 Algorithm Modules:**")
            algorithm_modules = ['TSPSolver', 'GASolver', 'PSOSolver', 'AlgorithmComparison']
            for module in algorithm_modules:
                status = MODULES_STATUS.get(module, False)
                icon = "✅" if status else "❌"
                st.write(f"{icon} {module}")
        
        with col2:
            st.write("**🎨 UI Components:**")
            ui_modules = ['google_maps_ui', 'sidebar', 'results_display']
            for module in ui_modules:
                status = MODULES_STATUS.get(module, False)
                icon = "✅" if status else "❌"
                st.write(f"{icon} {module}")
        
        with col3:
            st.write("**🔗 External Services:**")
            api_status = "✅ Connected" if GOOGLE_MAPS_API_KEY else "❌ Not configured"
            st.write(f"Google Maps API: {api_status}")
            
            geo_utils_status = "✅ Available" if st.session_state.geo_utils else "❌ Not available"
            st.write(f"GeoUtils: {geo_utils_status}")
            
            visualizer_status = "✅ Available" if MODULES_STATUS.get('Visualizer', False) else "❌ Not available"
            st.write(f"Visualizer: {visualizer_status}")
        
        # Import logs
        st.write("---")
        st.write("### 📝 Import Logs")
        if hasattr(st.session_state, 'import_logs'):
            for icon, msg in st.session_state.import_logs:
                st.write(f"{icon} {msg}")

# =============================================================================
# 7. MAIN APPLICATION
# =============================================================================
def main():
    """Main application logic"""
    
    # Render header
    render_header()
    
    # Check critical modules
    if not render_integrated_map or not validate_locations:
        st.error("⚠️ Critical UI components are missing. Please check your installation.")
        st.stop()
    
    # Render sidebar (configuration)
    if render_sidebar:
        config_params = render_sidebar()
    else:
        config_params = {'ga_params': GA_CONFIG, 'pso_params': PSO_CONFIG, 'algorithm_choice': 'Genetic Algorithm (GA)'}
    
    # Get dynamic map center
    map_center, map_zoom = get_map_center_and_zoom()
    
    # =========================================================================
    # SECTION 1: MAP PICKER + ROUTE DISPLAY
    # =========================================================================
    st.write("---")
    st.subheader("🗺️ Step 1: Pick Locations on Map")
    
    # Instructions
    if not st.session_state.get('optimized_route'):
        st.info("""
        **📍 How to add locations:**
        1. 🖱️ **Click directly on the map** to add locations
        2. ✏️ Edit location names in the table below
        3. 🏠 Mark one location as depot (start/end point)
        
        **💡 Tips:**
        - Need at least **3 locations** to optimize
        - Map will automatically center on your last clicked position
        - Use the search box in the map to find specific addresses
        """)
    else:
        st.success("✅ **Route optimized!** The map shows your optimized delivery route.")
    
    # Map controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("🔄 Reset View", use_container_width=True):
            st.session_state.last_map_center = None
            st.session_state.last_map_zoom = None
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.clicked_locations = []
            st.session_state.locations = []
            st.session_state.location_counter = 1
            st.session_state.distance_matrix = None
            st.session_state.optimized_route = None
            st.session_state.route_coords = None
            st.session_state.optimization_results = None
            st.session_state.comparison_results = None
            st.session_state.last_map_center = None
            st.session_state.last_map_zoom = None
            st.success("✅ Cleared all data!")
            st.rerun()

    # Render integrated map
    current_locations = render_integrated_map(
        initial_locations=st.session_state.clicked_locations,
        optimized_route=st.session_state.optimized_route,
        route_coordinates=st.session_state.route_coords,
        center=map_center,
        zoom=map_zoom,
        height=600
    )

    # Update session state
    st.session_state.clicked_locations = current_locations
    st.session_state.locations = current_locations
    
    # =========================================================================
    # SECTION 2: LOCATION TABLE WITH SELECTION
    # =========================================================================
    if st.session_state.clicked_locations:
        st.write("---")
        st.subheader("📋 Step 2: Review & Edit Locations")
        
        # Validation
        validation = validate_locations(st.session_state.clicked_locations)
        
        if not validation['valid']:
            for error in validation['errors']:
                st.error(error)
        
        if validation['warnings']:
            for warning in validation['warnings']:
                st.warning(warning)
        
        # Create DataFrame
        df_data = []
        for i, loc in enumerate(st.session_state.clicked_locations):
            df_data.append({
                'No.': i + 1,
                'Name': loc['name'],
                'Address': loc.get('address', ''),
                'Lat': loc['lat'],
                'Lng': loc['lng'],
                '🏠': loc.get('isStart', False),
                '☑️': False
            })
        
        df = pd.DataFrame(df_data)
        
        st.write("**💡 Tip:** Check ☑️ to select rows, edit Name column directly, then use buttons below.")
        
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                'No.': st.column_config.NumberColumn('No.', disabled=True, width='small'),
                'Name': st.column_config.TextColumn('Location Name', width='medium', required=True),
                'Address': st.column_config.TextColumn('Address', disabled=True, width='large'),
                'Lat': st.column_config.NumberColumn('Latitude', format="%.6f", disabled=True, width='small'),
                'Lng': st.column_config.NumberColumn('Longitude', format="%.6f", disabled=True, width='small'),
                '🏠': st.column_config.CheckboxColumn('Depot', disabled=True, width='small', help='✓ = Current depot'),
                '☑️': st.column_config.CheckboxColumn('Select', width='small', help='Check to select for actions')
            },
            key="location_editor"
        )
        
        # Sync name changes
        if not edited_df['Name'].equals(df['Name']):
            for idx, row in edited_df.iterrows():
                if not row['☑️']:
                    st.session_state.clicked_locations[idx]['name'] = str(row['Name'])
                    st.session_state.locations[idx]['name'] = str(row['Name'])
        
        # Get selected rows
        selected_rows = edited_df[edited_df['☑️'] == True]
        num_selected = len(selected_rows)
        
        # Action buttons (only when selections exist)
        if num_selected > 0:
            st.write("")
            col1, col2, col3, col4, col5 = st.columns([2, 1, 0.3, 1, 2])
            
            # SET DEPOT BUTTON
            with col2:
                depot_disabled = num_selected != 1
                
                if st.button("🏠 Set as Depot", type="primary", use_container_width=True, 
                           disabled=depot_disabled, help="Select exactly 1 row" if depot_disabled else "Set as depot"):
                    depot_idx = selected_rows.index[0]
                    
                    # Clear all depots
                    for loc in st.session_state.clicked_locations:
                        loc['isStart'] = False
                        loc['type'] = 'customer'
                    for loc in st.session_state.locations:
                        loc['isStart'] = False
                        loc['type'] = 'customer'
                    
                    # Set new depot
                    st.session_state.clicked_locations[depot_idx]['isStart'] = True
                    st.session_state.clicked_locations[depot_idx]['type'] = 'depot'
                    st.session_state.locations[depot_idx]['isStart'] = True
                    st.session_state.locations[depot_idx]['type'] = 'depot'
                    
                    # Reset optimization
                    st.session_state.distance_matrix = None
                    st.session_state.optimized_route = None
                    st.session_state.optimization_results = None
                    
                    depot_name = st.session_state.clicked_locations[depot_idx]['name']
                    st.success(f"✅ Set depot: {depot_name}")
                    st.rerun()
            
            with col3:
                st.write("")
            
            # DELETE BUTTON
            with col4:
                if st.button(f"🗑️ Delete ({num_selected})", type="secondary", use_container_width=True,
                           help=f"Delete {num_selected} location(s)"):
                    # Keep only unselected rows
                    indices_to_keep = edited_df[edited_df['☑️'] == False].index.tolist()
                    
                    st.session_state.clicked_locations = [st.session_state.clicked_locations[i] for i in indices_to_keep]
                    st.session_state.locations = [st.session_state.locations[i] for i in indices_to_keep]
                    
                    # Reset optimization
                    st.session_state.distance_matrix = None
                    st.session_state.optimized_route = None
                    st.session_state.optimization_results = None
                    
                    st.success(f"✅ Deleted {num_selected} location(s)")
                    st.rerun()
        
        st.write("")
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📍 Total", len(st.session_state.clicked_locations))
        with col2:
            current_depot = next((loc['name'] for loc in st.session_state.clicked_locations if loc.get('isStart')), 'None')
            st.metric("🏠 Depot", current_depot)
        with col3:
            depot_count = sum(1 for loc in st.session_state.clicked_locations if loc.get('isStart'))
            st.metric("📦 Deliveries", len(st.session_state.clicked_locations) - depot_count)
        with col4:
            validation_status = "✅ Valid" if validation['valid'] else "❌ Invalid"
            st.metric("Status", validation_status)
    
    # =========================================================================
    # SECTION 3: DISTANCE CALCULATION
    # =========================================================================
    if len(st.session_state.clicked_locations) >= 3:
        st.write("---")
        st.subheader("📏 Step 3: Calculate Distances")
        
        validation = validate_locations(st.session_state.clicked_locations)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🔄 Calculate Distances", type="primary", use_container_width=True,
                        disabled=not validation['valid']):
                try:
                    with st.spinner("🛣️ Calculating road distances..."):
                        # ✅ FIND DEPOT INDEX
                        depot_idx = next((i for i, loc in enumerate(st.session_state.clicked_locations) 
                                        if loc.get('isStart')), 0)
                        
                        # ✅ REORDER LOCATIONS - DEPOT FIRST
                        locations = st.session_state.clicked_locations.copy()
                        if depot_idx != 0:
                            depot_loc = locations.pop(depot_idx)
                            locations.insert(0, depot_loc)
                        
                        # ✅ UPDATE SESSION STATE WITH REORDERED LOCATIONS
                        st.session_state.locations = locations
                        st.session_state.clicked_locations = locations  # Keep sync
                        
                        # Calculate matrix with depot at index 0
                        if st.session_state.geo_utils:
                            coordinates = [(loc['lat'], loc['lng']) for loc in locations]
                            distance_matrix = st.session_state.geo_utils.get_distance_matrix(coordinates)
                            st.session_state.distance_matrix = distance_matrix
                            st.success("✅ Distances calculated with depot at position 0!")
                        else:
                            # Fallback haversine
                            n = len(locations)
                            distance_matrix = np.zeros((n, n))
                            for i in range(n):
                                for j in range(n):
                                    if i != j:
                                        distance_matrix[i][j] = haversine_distance(
                                            locations[i]['lat'],
                                            locations[i]['lng'],
                                            locations[j]['lat'],
                                            locations[j]['lng']
                                        )
                            st.session_state.distance_matrix = distance_matrix
                            st.info("📐 Using straight-line distances (Haversine)")
                            
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")
        
        if not validation['valid']:
            st.info("💡 Fix validation errors above to calculate distances")
        
        # Display distances
        if st.session_state.distance_matrix is not None:
            st.write("")
            
            non_zero_distances = st.session_state.distance_matrix[st.session_state.distance_matrix > 0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Routes", f"{len(non_zero_distances)}")
            with col2:
                st.metric("📏 Shortest", f"{np.min(non_zero_distances):.2f} km")
            with col3:
                st.metric("📏 Longest", f"{np.max(non_zero_distances):.2f} km")
            with col4:
                st.metric("📏 Average", f"{np.mean(non_zero_distances):.2f} km")
    
    # =========================================================================
    # SECTION 4: OPTIMIZATION
    # =========================================================================
    if st.session_state.distance_matrix is not None:
        st.write("---")
        st.subheader("🚀 Step 4: Run Optimization")
        
        validation = validate_locations(st.session_state.clicked_locations)
        
        if not validation['valid']:
            st.error("❌ Cannot optimize: Please fix validation errors above")
            for error in validation['errors']:
                st.error(error)
        else:
            # ========================
            # ALGORITHM SELECTION
            # ========================
            st.subheader("🎯 Select Optimization Mode")
            
            algorithm_mode = st.radio(
                "Choose how to run the optimization:",
                ["🧬 Genetic Algorithm (GA)", "🐝 Particle Swarm Optimization (PSO)", "🔬 Compare Both"],
                horizontal=True,
                help="GA = Evolution-based | PSO = Swarm-based | Compare = Run both algorithms"
            )
            
            st.write("")
            
            # ========================
            # PARAMETERS DISPLAY
            # ========================
            if algorithm_mode == "🧬 Genetic Algorithm (GA)":
                with st.expander("⚙️ GA Parameters", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    ga_params = config_params.get('ga_params', GA_CONFIG)
                    
                    with col1:
                        st.metric("Population Size", ga_params.get('population_size', 50))
                        st.metric("Generations", ga_params.get('generations', 100))
                    
                    with col2:
                        st.metric("Crossover Rate", f"{ga_params.get('crossover_probability', 0.8):.2f}")
                        st.metric("Mutation Rate", f"{ga_params.get('mutation_probability', 0.05):.3f}")
                    
                    with col3:
                        st.metric("Tournament Size", ga_params.get('tournament_size', 3))
                        st.metric("Elite Size", ga_params.get('elite_size', 2))
                    
                    st.caption("💡 Adjust parameters in sidebar → 🔧 Algorithm Configuration")
            
            elif algorithm_mode == "🐝 Particle Swarm Optimization (PSO)":
                with st.expander("⚙️ PSO Parameters", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    pso_params = config_params.get('pso_params', PSO_CONFIG)
                    
                    with col1:
                        st.metric("Swarm Size", pso_params.get('swarm_size', 30))
                        st.metric("Max Iterations", pso_params.get('max_iterations', 100))
                    
                    with col2:
                        st.metric("Inertia (w)", f"{pso_params.get('w', 0.729):.3f}")
                        st.metric("Cognitive (c1)", f"{pso_params.get('c1', 1.494):.2f}")
                    
                    with col3:
                        st.metric("Social (c2)", f"{pso_params.get('c2', 1.494):.2f}")
                        st.metric("Min Inertia (w_min)", f"{pso_params.get('w_min', 0.4):.3f}")
                    
                    st.caption("💡 Adjust parameters in sidebar → 🔧 Algorithm Configuration")
            
            else:  # Compare Both
                st.info("**🔬 Running Both Algorithms for Comparison**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🧬 GA Parameters:**")
                    ga_params = config_params.get('ga_params', GA_CONFIG)
                    ga_config_text = f"""
                    - Population: `{ga_params.get('population_size', 50)}`
                    - Generations: `{ga_params.get('generations', 100)}`
                    - Crossover: `{ga_params.get('crossover_probability', 0.8):.2f}`
                    - Mutation: `{ga_params.get('mutation_probability', 0.05):.3f}`
                    - Tournament: `{ga_params.get('tournament_size', 3)}`
                    - Elite: `{ga_params.get('elite_size', 2)}`
                    """
                    st.markdown(ga_config_text)
                
                with col2:
                    st.write("**🐝 PSO Parameters:**")
                    pso_params = config_params.get('pso_params', PSO_CONFIG)
                    pso_config_text = f"""
                    - Swarm Size: `{pso_params.get('swarm_size', 30)}`
                    - Iterations: `{pso_params.get('max_iterations', 100)}`
                    - Inertia (w): `{pso_params.get('w', 0.729):.3f}`
                    - c1: `{pso_params.get('c1', 1.494):.2f}`
                    - c2: `{pso_params.get('c2', 1.494):.2f}`
                    - w_min: `{pso_params.get('w_min', 0.4):.3f}`
                    """
                    st.markdown(pso_config_text)
            
            # Problem Summary
            st.write("---")
            st.write("**📊 Problem Summary:**")
            col1, col2, col3, col4 = st.columns(4)
            
            num_locs = len(st.session_state.clicked_locations)
            num_routes = num_locs * (num_locs - 1)
            
            with col1:
                st.metric("Locations", num_locs)
            
            with col2:
                st.metric("Possible Routes", f"{num_routes:,}")
            
            with col3:
                if num_locs > 0:
                    total_nodes = num_locs
                    st.metric("Search Space", f"{total_nodes}!")
            
            with col4:
                if num_locs > 0:
                    st.metric("Problem Type", "TSP")
            
            st.write("")
            
            # ========================
            # BUTTONS - BASED ON MODE
            # ========================
            if algorithm_mode == "🧬 Genetic Algorithm (GA)":
                optimize_clicked = st.button(
                    "🧬 Optimize with GA",
                    type="primary",
                    use_container_width=True,
                    help="Run Genetic Algorithm for TSP optimization"
                )
            elif algorithm_mode == "🐝 Particle Swarm Optimization (PSO)":
                optimize_clicked = st.button(
                    "🐝 Optimize with PSO",
                    type="primary",
                    use_container_width=True,
                    help="Run Particle Swarm Optimization for TSP optimization"
                )
            else:  # Compare Both
                optimize_clicked = st.button(
                    "🔬 Compare GA vs PSO",
                    type="primary",
                    use_container_width=True,
                    help="Run both GA and PSO for comprehensive comparison"
                )
            
            # Optimization process
            if optimize_clicked:
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔄 Initializing algorithm...")
                    progress_bar.progress(0.2)
                    
                    # Determine which algorithm(s) to run
                    if algorithm_mode == "🔬 Compare Both":
                        # Run comparison
                        with st.spinner("🔬 Running comparison analysis..."):
                            status_text.text("🔄 Starting comparison...")
                            progress_bar.progress(0.3)
                            
                            # Get parameters
                            ga_params = config_params.get('ga_params', GA_CONFIG)
                            pso_params = config_params.get('pso_params', PSO_CONFIG)
                            
                            # Run both algorithms
                            from src.comparison import compare_algorithms
                            
                            status_text.text("🧬 Running Genetic Algorithm...")
                            progress_bar.progress(0.4)
                            ga_results = run_genetic_algorithm(st.session_state.distance_matrix, {'ga_params': ga_params})
                            
                            status_text.text("🐝 Running Particle Swarm Optimization...")
                            progress_bar.progress(0.6)
                            pso_results = run_pso_algorithm(st.session_state.distance_matrix, {'pso_params': pso_params})
                            
                            status_text.text("📊 Generating comparison analysis...")
                            progress_bar.progress(0.8)
                            
                            if ga_results and pso_results and ga_results.get('success') and pso_results.get('success'):
                                # Generate comparison data
                                comparison_data = {
                                    'valid': True,
                                    'ga_distance': ga_results.get('best_distance', 0),
                                    'pso_distance': pso_results.get('best_distance', 0),
                                    'ga_runtime': ga_results.get('runtime_seconds', 0),
                                    'pso_runtime': pso_results.get('runtime_seconds', 0),
                                    'ga_improvement_pct': ((ga_results.get('initial_distance', 0) - ga_results.get('best_distance', 0)) / max(ga_results.get('initial_distance', 1), 0.0001)) * 100,
                                    'pso_improvement_pct': ((pso_results.get('initial_distance', 0) - pso_results.get('best_distance', 0)) / max(pso_results.get('initial_distance', 1), 0.0001)) * 100,
                                }
                                
                                # Determine winners
                                if comparison_data['ga_distance'] < comparison_data['pso_distance']:
                                    comparison_data['winner_distance'] = 'GA'
                                    comparison_data['distance_difference'] = comparison_data['pso_distance'] - comparison_data['ga_distance']
                                    comparison_data['distance_difference_pct'] = (comparison_data['distance_difference'] / comparison_data['pso_distance']) * 100 if comparison_data['pso_distance'] > 0 else 0
                                elif comparison_data['pso_distance'] < comparison_data['ga_distance']:
                                    comparison_data['winner_distance'] = 'PSO'
                                    comparison_data['distance_difference'] = comparison_data['ga_distance'] - comparison_data['pso_distance']
                                    comparison_data['distance_difference_pct'] = (comparison_data['distance_difference'] / comparison_data['ga_distance']) * 100 if comparison_data['ga_distance'] > 0 else 0
                                else:
                                    comparison_data['winner_distance'] = 'Tie'
                                    comparison_data['distance_difference'] = 0
                                    comparison_data['distance_difference_pct'] = 0
                                
                                if comparison_data['ga_runtime'] < comparison_data['pso_runtime']:
                                    comparison_data['winner_runtime'] = 'GA'
                                    comparison_data['runtime_difference'] = comparison_data['pso_runtime'] - comparison_data['ga_runtime']
                                    comparison_data['runtime_difference_pct'] = (comparison_data['runtime_difference'] / comparison_data['pso_runtime']) * 100 if comparison_data['pso_runtime'] > 0 else 0
                                elif comparison_data['pso_runtime'] < comparison_data['ga_runtime']:
                                    comparison_data['winner_runtime'] = 'PSO'
                                    comparison_data['runtime_difference'] = comparison_data['ga_runtime'] - comparison_data['pso_runtime']
                                    comparison_data['runtime_difference_pct'] = (comparison_data['runtime_difference'] / comparison_data['ga_runtime']) * 100 if comparison_data['ga_runtime'] > 0 else 0
                                else:
                                    comparison_data['winner_runtime'] = 'Tie'
                                    comparison_data['runtime_difference'] = 0
                                    comparison_data['runtime_difference_pct'] = 0
                                
                                if comparison_data['ga_improvement_pct'] > comparison_data['pso_improvement_pct']:
                                    comparison_data['winner_improvement'] = 'GA'
                                elif comparison_data['pso_improvement_pct'] > comparison_data['ga_improvement_pct']:
                                    comparison_data['winner_improvement'] = 'PSO'
                                else:
                                    comparison_data['winner_improvement'] = 'Tie'
                                
                                # Store comparison results
                                st.session_state.comparison_results = {
                                    'ga_results': ga_results,
                                    'pso_results': pso_results,
                                    'ga_params': ga_params,
                                    'pso_params': pso_params,
                                    'comparison': comparison_data
                                }
                                
                                progress_bar.progress(1.0)
                                status_text.text("✅ Comparison completed!")
                                st.balloons()
                                st.success("✅ Comparison Analysis Complete!")
                            else:
                                status_text.text("❌ One or both algorithms failed")
                                st.error("❌ Comparison failed. Please check your parameters.")
                    
                    else:
                        # Single algorithm optimization
                        # Clear previous comparison results when not comparing both
                        st.session_state.comparison_results = None

                        if "GA" in algorithm_mode:
                            algorithm_name = "Genetic Algorithm"
                            with st.spinner(f"🧬 Running {algorithm_name}..."):
                                status_text.text("🧬 Evolving population...")
                                progress_bar.progress(0.4)
                                results = run_genetic_algorithm(st.session_state.distance_matrix, config_params)
                        else:
                            algorithm_name = "Particle Swarm Optimization"
                            with st.spinner(f"🐝 Running {algorithm_name}..."):
                                status_text.text("🐝 Optimizing swarm...")
                                progress_bar.progress(0.4)
                                results = run_pso_algorithm(st.session_state.distance_matrix, config_params)
                        
                        status_text.text("📊 Processing results...")
                        progress_bar.progress(0.7)
                        
                        if results and results.get('success'):
                            st.session_state.optimized_route = results['best_route']
                            st.session_state.optimization_results = results
                            
                            # Get real road coordinates
                            status_text.text("🗺️ Generating route path...")
                            progress_bar.progress(0.85)
                            
                            if st.session_state.get('geo_utils'):
                                try:
                                    route_locs = [st.session_state.clicked_locations[i] for i in results['best_route']]
                                    route_locs.append(route_locs[0])
                                    coordinates = [[loc['lng'], loc['lat']] for loc in route_locs]
                                    
                                    from components.results_display import get_openroute_directions
                                    openroute_key = os.getenv('OPENROUTE_API_KEY')
                                    if openroute_key:
                                        directions = get_openroute_directions(coordinates, openroute_key)
                                        if directions and directions.get('features'):
                                            geometry = directions['features'][0]['geometry']
                                            if geometry['type'] == 'LineString':
                                                st.session_state.route_coords = [[c[1], c[0]] for c in geometry['coordinates']]
                                except Exception:
                                    pass
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Optimization completed!")
                            
                            st.success("✅ Optimization completed successfully!")
                                                   
                            st.balloons()
                            
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.info("💡 **Scroll up** to see the optimized route on the map!")
                            st.rerun()
                        else:
                            progress_bar.empty()
                            status_text.empty()
                            st.error("❌ Optimization failed: No valid solution found")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error during optimization: {str(e)}")
                    
                    with st.expander("🔍 Error Details", expanded=False):
                        st.code(str(e))
                        st.write("**Possible causes:**")
                        st.write("- Algorithm parameters are invalid")
                        st.write("- Distance matrix has issues")
                        st.write("- Insufficient locations for optimization")    
    # =========================================================================
    # SECTION 5: RESULTS DISPLAY
    # =========================================================================
    if st.session_state.optimization_results and display_results:
        st.write("---")
        st.subheader("📊 Optimization Results")
        
        try:
            # ✅ Initialize visualizer if available
            visualizer = None
            if TSPVisualizer and MODULES_STATUS.get('Visualizer', False):
                try:
                    visualizer = TSPVisualizer(width=900, height=600)
                except Exception:
                    visualizer = None
            
            # ✅ Call display_results with visualizer
            display_results(
                st.session_state.optimization_results,
                st.session_state.locations,
                st.session_state.distance_matrix,
                visualizer=visualizer  # Pass visualizer here
            )
        except Exception as e:
            st.error(f"❌ Failed to display results: {str(e)}")
            
            with st.expander("🔍 Error Details", expanded=False):
                st.code(traceback.format_exc())
    
    # =========================================================================
    # SECTION 6: ALGORITHM COMPARISON RESULTS
    # =========================================================================
    if st.session_state.comparison_results:
        st.write("---")
        st.subheader("🔬 Algorithm Comparison Results")
        
        try:
            display_comparison_results(
                st.session_state.comparison_results,
                st.session_state.clicked_locations,
                st.session_state.distance_matrix
            )
        except Exception as e:
            st.error(f"❌ Failed to display comparison: {str(e)}")
            
            with st.expander("🔍 Error Details", expanded=False):
                st.code(traceback.format_exc())


# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()
