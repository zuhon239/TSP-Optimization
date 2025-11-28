"""
OpenRouteService - FREE Alternative to OSRM
2000 requests/day, excellent for academic projects
"""

import requests
import numpy as np

class OpenRouteServiceAPI:
    def __init__(self, api_key: str = None):
        # Get FREE API key: https://openrouteservice.org/dev/
        self.api_key = api_key or "5b3ce3597851110001cf6248YOUR_KEY_HERE"
        self.base_url = "https://api.openrouteservice.org"
        
    def get_distance_matrix(self, locations):
        """
        Get distance matrix from OpenRouteService
        FREE: 2000 requests/day
        """
        url = f"{self.base_url}/v2/matrix/driving-car"
        
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        
        # ORS format: [[lng, lat], [lng, lat], ...]
        locations_formatted = [[lng, lat] for lat, lng in locations]
        
        payload = {
            'locations': locations_formatted,
            'metrics': ['distance']
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Distance matrix in meters, convert to km
            distances = np.array(data['distances'])
            return distances / 1000.0
            
        except Exception as e:
            raise Exception(f"OpenRouteService failed: {str(e)}")

# Test OpenRouteService
def test_openroute():
    print("🧪 Testing OpenRouteService...")
    
    # You need to get FREE API key from: https://openrouteservice.org/dev/
    # For demo, I'll show with a placeholder
    
    locations = [
        (10.762622, 106.660172),  # Ben Thanh
        (10.786565, 106.695595),  # District 3
    ]
    
    # Demo request format (you need real API key)
    url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    
    headers = {
        'Authorization': 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImZmMmYzN2Y4YzA3NDQwMzliYjRhNzYyNWJlYzAzOTY2IiwiaCI6Im11cm11cjY0In0=',  # Get from openrouteservice.org
        'Content-Type': 'application/json'
    }
    
    payload = {
        'locations': [ [106.719318, 10.732932],[106.660172, 10.762622]],
        'metrics': ['distance']
    }
    
    try:
        # This will fail without real API key, but shows the format
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"Response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            distances = data['distances']
            distance_km = distances[0][1] / 1000
            print(f"✅ OpenRouteService works! Distance: {distance_km:.2f}km")
        else:
            print(f"❌ Need valid API key from openrouteservice.org")
            
    except Exception as e:
        print(f"❌ OpenRouteService test failed: {e}")

test_openroute()