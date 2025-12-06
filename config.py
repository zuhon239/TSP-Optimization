"""
Configuration file for TSP Optimization Project
Author: Hoàng (Team Leader)
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =====================================
# Routing Service Configuration
# =====================================
# OpenRouteService API for real road routing (FREE 2000 requests/day)
OPENROUTE_API_KEY = os.getenv('OPENROUTE_API_KEY', '')

# Default map settings
DEFAULT_CENTER = {'lat': 10.762622, 'lng': 106.660172}  # Ho Chi Minh City
DEFAULT_ZOOM = 13

# =====================================
# Genetic Algorithm (GA) Parameters
# =====================================
GA_DEFAULT_CONFIG = {
    'population_size': 100,
    'generations': 200,
    'crossover_probability': 0.8,
    'mutation_probability': 0.05,
    'tournament_size': 3,
    'elite_size': 5,  # Number of best individuals to preserve
}

# =====================================
# Particle Swarm Optimization (PSO) Parameters  
# =====================================
PSO_DEFAULT_CONFIG = {
    'swarm_size': 50,
    'max_iterations': 100,
    'w': 0.729,          # Inertia weight
    'c1': 1.494,         # Cognitive component
    'c2': 1.494,         # Social component
    'w_min': 0.4,        # Minimum inertia weight
    'w_max': 0.9,        # Maximum inertia weight
}

# =====================================
# Application Settings
# =====================================
APP_CONFIG = {
    'title': '🚛 TSP Optimization - Tối ưu hóa lộ trình giao hàng',
    'page_icon': '🚛',
    'layout': 'wide',
    'max_locations': 50,  # Maximum number of delivery points
    'cache_ttl': 3600,    # Cache time-to-live (seconds)
}

# =====================================
# File Paths
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')  
TEST_CASES_DIR = os.path.join(DATA_DIR, 'test_cases')

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, RESULTS_DIR, TEST_CASES_DIR]:
    os.makedirs(directory, exist_ok=True)

# =====================================
# Logging Configuration
# =====================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(BASE_DIR, 'tsp_optimizer.log'),
}

# =====================================
# Performance Settings
# =====================================
PERFORMANCE_CONFIG = {
    'max_runtime_seconds': 300,  # Maximum algorithm runtime
    'convergence_threshold': 1e-6,  # Convergence criterion
    'max_stagnant_generations': 50,  # Early stopping criterion
}

# =====================================
# Export current config for debugging
# =====================================
def print_config():
    """Print current configuration for debugging"""
    print("🔧 TSP Optimizer Configuration:")
    print(f"   OpenRouteService: {'✅ Configured' if OPENROUTE_API_KEY else '⚠️ Optional'}")
    print(f"   GA Population: {GA_DEFAULT_CONFIG['population_size']}")
    print(f"   PSO Swarm Size: {PSO_DEFAULT_CONFIG['swarm_size']}")
    print(f"   Cache Directory: {CACHE_DIR}")
    print(f"   Max Locations: {APP_CONFIG['max_locations']}")

if __name__ == "__main__":
    print_config()