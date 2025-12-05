"""
Google Maps Interactive UI Component - INTEGRATED VERSION
Single map for picking AND displaying optimized route
Author: Quân (Frontend Specialist)
Enhanced with validation, delete buttons, and constraints
"""
import streamlit as st
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# ✅ Import Folium and plugins
try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import (
        Geocoder, Fullscreen, MousePosition, MiniMap, AntPath
    )
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ✅ Import Geopy for reverse geocoding
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False


def get_address_from_coordinates(lat: float, lng: float) -> str:
    """Reverse geocoding: Convert coordinates to address"""
    if not HAS_GEOPY:
        return f"{lat:.6f}, {lng:.6f}"
    
    try:
        geolocator = Nominatim(user_agent="tsp_optimizer_app")
        location = geolocator.reverse(f"{lat}, {lng}", timeout=5, language='vi')
        
        if location:
            raw_address = location.raw.get('address', {})
            parts = []
            if raw_address.get('road'):
                parts.append(raw_address['road'])
            if raw_address.get('suburb') or raw_address.get('quarter'):
                parts.append(raw_address.get('suburb') or raw_address.get('quarter'))
            if raw_address.get('city') or raw_address.get('town'):
                parts.append(raw_address.get('city') or raw_address.get('town'))
            return ', '.join(parts) if parts else location.address
        else:
            return f"{lat:.6f}, {lng:.6f}"
    except:
        return f"{lat:.6f}, {lng:.6f}"


def validate_locations(locations: List[Dict]) -> Dict[str, any]:
    """
    Validate location constraints
    
    Returns:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str]
        }
    """
    errors = []
    warnings = []
    
    # ✅ CONSTRAINT 1: Minimum 3 locations
    if len(locations) < 3:
        errors.append(f"❌ Need at least 3 locations (current: {len(locations)})")
    
    # ✅ CONSTRAINT 2: Exactly 1 depot
    depot_count = sum(1 for loc in locations if loc.get('isStart', False))
    
    if depot_count == 0:
        errors.append("❌ Must have exactly 1 depot. Please mark one location as depot.")
    elif depot_count > 1:
        errors.append(f"❌ Can only have 1 depot (current: {depot_count}). Please uncheck extra depots.")
    
    # ✅ Check for duplicate coordinates
    coords_set = set()
    for loc in locations:
        coord = (round(loc['lat'], 6), round(loc['lng'], 6))
        if coord in coords_set:
            warnings.append(f"⚠️ Duplicate location detected: {loc['name']}")
        coords_set.add(coord)
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def render_integrated_map(
    initial_locations: List[Dict] = None,
    optimized_route: List[int] = None,
    route_coordinates: List[List[float]] = None,
    center: Dict = None,
    zoom: int = 13,
    height: int = 600
) -> List[Dict]:
    """
    Integrated map: picking + route visualization on SAME map
    """
    if center is None:
        center = {'lat': 10.762622, 'lng': 106.660172}
    
    if initial_locations is None:
        initial_locations = []
    
    # ✅ Initialize session_state
    if 'clicked_locations' not in st.session_state:
        st.session_state.clicked_locations = []
    
    if 'location_counter' not in st.session_state:
        st.session_state.location_counter = 1
    
    if 'location_to_delete' not in st.session_state:
        st.session_state.location_to_delete = None
    
    if not HAS_FOLIUM:
        st.error("❌ streamlit-folium not installed")
        return st.session_state.clicked_locations
    
    # ✅ Create Folium map
    m = folium.Map(
        location=[center['lat'], center['lng']],
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # ✅ ADD PLUGINS
    Geocoder(
        collapsed=False,
        position='topleft',
        placeholder='🔍 Search address...',
        add_marker=False
    ).add_to(m)
    
    Fullscreen(
        position='topright',
        title='📐 Fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    MousePosition(
        position='bottomright',
        separator=' | ',
        prefix='📍 ',
        lat_formatter="function(num) {return 'Lat: ' + L.Util.formatNum(num, 6);}",
        lng_formatter="function(num) {return 'Lng: ' + L.Util.formatNum(num, 6);}"
    ).add_to(m)
    
    MiniMap(
        toggle_display=True,
        position='bottomleft',
        width=150,
        height=150,
        zoom_level_offset=-5
    ).add_to(m)
    
    # ✅ DRAW ANIMATED ROUTE (if optimized)
    if optimized_route and len(optimized_route) > 0:
        if route_coordinates:
            AntPath(
                locations=route_coordinates,
                color='#1976d2',
                weight=6,
                opacity=0.8,
                delay=800,
                dash_array=[10, 20],
                pulse_color='#ff6b6b'
            ).add_to(m)
        else:
            route_locs = [st.session_state.clicked_locations[i] for i in optimized_route]
            route_coords = [[loc['lat'], loc['lng']] for loc in route_locs]
            route_coords.append([route_locs[0]['lat'], route_locs[0]['lng']])
            
            AntPath(
                locations=route_coords,
                color='#1976d2',
                weight=5,
                opacity=0.7,
                delay=1000,
                dash_array=[15, 25],
                pulse_color='#ff6b6b'
            ).add_to(m)
    
    # ✅ ADD MARKERS with DELETE button in popup
    locations_to_display = st.session_state.clicked_locations
    
    if optimized_route and len(optimized_route) > 0:
        # Show route order
        for route_order, loc_idx in enumerate(optimized_route):
            loc = locations_to_display[loc_idx]
            is_start = route_order == 0
            
            if is_start:
                icon_html = f"""
                <div style="display: flex; align-items: center; justify-content: center;
                            width: 35px; height: 35px; background-color: #4caf50;
                            border: 3px solid white; border-radius: 50%;
                            font-size: 18px; font-weight: bold; color: white;
                            box-shadow: 0 3px 10px rgba(0,0,0,0.4);">🏠</div>
                """
            else:
                icon_html = f"""
                <div style="display: flex; align-items: center; justify-content: center;
                            width: 32px; height: 32px; background-color: #f44336;
                            border: 3px solid white; border-radius: 50%;
                            font-size: 16px; font-weight: bold; color: white;
                            box-shadow: 0 3px 10px rgba(0,0,0,0.4);
                            font-family: Arial Black, sans-serif;">{route_order}</div>
                """
            
            icon = folium.DivIcon(html=icon_html)
            
            popup_html = f"""
            <div style='font-family: Arial; min-width: 250px; padding: 10px;'>
                <h4 style='margin: 0 0 10px 0; color: #1976d2;'>
                    {'🏠 START/END' if is_start else f'📦 STOP {route_order}'}: {loc['name']}
                </h4>
                <p><b>📍 Address:</b><br>{loc.get('address', 'N/A')}</p>
                <p><b>🌍 Coordinates:</b><br>{loc['lat']:.6f}, {loc['lng']:.6f}</p>
                <p><b>🔢 Route Order:</b> Stop {route_order} of {len(optimized_route)-1}</p>
            </div>
            """
            
            folium.Marker(
                location=[loc['lat'], loc['lng']],
                icon=icon,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Stop {route_order}: {loc['name']}"
            ).add_to(m)
    
    else:
        # Pick mode: show pick order with DELETE button
        for i, loc in enumerate(locations_to_display):
            is_depot = loc.get('isStart', False)
            
            if is_depot:
                icon_html = f"""
                <div style="display: flex; align-items: center; justify-content: center;
                            width: 35px; height: 35px; background-color: #4caf50;
                            border: 3px solid white; border-radius: 50%;
                            font-size: 18px; color: white;
                            box-shadow: 0 3px 10px rgba(0,0,0,0.4);">🏠</div>
                """
            else:
                icon_html = f"""
                <div style="display: flex; align-items: center; justify-content: center;
                            width: 30px; height: 30px; background-color: #2196F3;
                            border: 3px solid white; border-radius: 50%;
                            font-size: 14px; font-weight: bold; color: white;
                            box-shadow: 0 3px 10px rgba(0,0,0,0.4);">{i+1}</div>
                """
            
            icon = folium.DivIcon(html=icon_html)
            
            # ✅ POPUP WITH DELETE BUTTON
            popup_html = f"""
            <div style='font-family: Arial; min-width: 280px; padding: 12px;'>
                <h4 style='margin: 0 0 10px 0; color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 5px;'>
                    {'🏠 Depot' if is_depot else f'📦 Location {i+1}'}: {loc['name']}
                </h4>
                <p style='margin: 5px 0;'><b>📍 Address:</b><br>{loc.get('address', 'N/A')}</p>
                <p style='margin: 5px 0;'><b>🌍 Coordinates:</b><br>
                   <code style='background: #f0f0f0; padding: 2px 5px; border-radius: 3px;'>
                   {loc['lat']:.6f}, {loc['lng']:.6f}</code></p>
                <p style='margin: 5px 0;'><b>🏷️ Type:</b> {'Depot (Start/End)' if is_depot else 'Delivery Point'}</p>
                <hr style='margin: 10px 0; border: none; border-top: 1px solid #ddd;'>
                <p style='margin: 5px 0; color: #666; font-size: 11px;'>
                    💡 To delete this location, use the table below or "Clear All" button
                </p>
            </div>
            """
            
            folium.Marker(
                location=[loc['lat'], loc['lng']],
                icon=icon,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{'🏠 ' if is_depot else ''}Click for details: {loc['name']}"
            ).add_to(m)
    
    # Click handler
    m.add_child(folium.LatLngPopup())
    
    # ✅ Render map
    map_data = st_folium(
        m,
        width=None,
        height=height,
        returned_objects=['last_clicked'],
        key="integrated_map"
    )
    
    # ✅ Capture clicks (only in pick mode)
    if not optimized_route and map_data and map_data.get('last_clicked'):
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lng = map_data['last_clicked']['lng']
        
        # Check duplicate
        is_new = True
        for loc in st.session_state.clicked_locations:
            if abs(loc['lat'] - clicked_lat) < 0.00001 and abs(loc['lng'] - clicked_lng) < 0.00001:
                is_new = False
                break
        
        if is_new:
            with st.spinner("🔍 Getting address..."):
                address = get_address_from_coordinates(clicked_lat, clicked_lng)
            
            new_location = {
                'name': f'Location {st.session_state.location_counter}',
                'lat': clicked_lat,
                'lng': clicked_lng,
                'address': address,
                'isStart': False,
                'type': 'customer'
            }
            st.session_state.clicked_locations.append(new_location)
            st.session_state.location_counter += 1
            st.success(f"✅ Added: {address}")
            st.rerun()
    
    return st.session_state.clicked_locations