"""
TSP Optimization Package
Genetic Algorithm and Particle Swarm Optimization for Travelling Salesman Problem
"""

__version__ = "1.0.0"
__author__ = "Team: Hoàng, Quang, Quân, Nhân"
__email__ = "team@huflit.edu.vn" 
__description__ = "TSP optimization using GA and PSO algorithms"

# Import main classes
from .tsp_solver import TSPSolver
from .ga_solver import GASolver
from .pso_solver import PSOSolver
from .geo_utils import GeoUtils
from .comparison import AlgorithmComparison

__all__ = [
    'TSPSolver',
    'GASolver', 
    'PSOSolver',
    'GeoUtils',
    'AlgorithmComparison'
]