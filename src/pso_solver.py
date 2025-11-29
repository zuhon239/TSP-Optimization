"""
Particle Swarm Optimization TSP Solver

Custom implementation of PSO for Travelling Salesman Problem with depot support

Author: Quang (Algorithm Specialist)
"""

import numpy as np
import random
import logging
from typing import List, Tuple, Dict

# TSP Solver import with better error handling
try:
    from .tsp_solver import TSPSolver  # Package import
except ImportError:
    from tsp_solver import TSPSolver  # Direct import
    
import config


class PSOSolver(TSPSolver):
    """
    Particle Swarm Optimization implementation for TSP
    Using discrete PSO with swap sequence representation and depot support
    """
    
    def __init__(self, distance_matrix: np.ndarray,
                 locations: List[dict] = None,
                 swarm_size: int = None,
                 max_iterations: int = None,
                 w: float = None,
                 c1: float = None,
                 c2: float = None,
                 w_min: float = None,
                 w_max: float = None):
        """Initialize PSO solver with depot support"""
        
        try:
            # ✅ Call parent class first
            super().__init__(distance_matrix, locations)
            
            # ✅ Verify logger exists after parent init
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = logging.getLogger(f"{self.__class__.__name__}")
                self.logger.warning("⚠️ Created fallback logger for PSOSolver")
            
        except Exception as e:
            # ✅ Last resort initialization if parent fails
            self.distance_matrix = np.array(distance_matrix)
            self.num_cities = len(distance_matrix)
            self.locations = locations or []
            self.start_point = 0
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.iteration_history = []
            self.best_distances = []
            self.logger.error(f"❌ Parent TSPSolver init failed: {e}")
        
        # Load parameters
        try:
            pso_config = config.PSO_DEFAULT_CONFIG
        except (AttributeError, NameError):
            pso_config = {
                'swarm_size': 30,
                'max_iterations': 100,
                'w': 0.729,
                'c1': 1.494,
                'c2': 1.494,
                'w_min': 0.4,
                'w_max': 0.9
            }
            
        self.swarm_size = swarm_size or pso_config['swarm_size']
        self.max_iterations = max_iterations or pso_config['max_iterations']
        self.w = w or pso_config['w']
        self.c1 = c1 or pso_config['c1']
        self.c2 = c2 or pso_config['c2']
        self.w_min = w_min or pso_config['w_min']
        self.w_max = w_max or pso_config['w_max']
        
        # PSO specific variables
        self.particles = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = float('inf')
        
        # ✅ Now safe to use logger (guaranteed to exist)
        self.logger.info(f"🐝 PSO initialized: swarm={self.swarm_size}, "
                        f"iterations={self.max_iterations}, depot={getattr(self, 'start_point', 0)}")

