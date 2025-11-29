"""
Algorithm Comparison Module

Compare GA vs PSO performance on TSP instances
"""

import os
import numpy as np
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
import config

try:
    from .ga_solver import GASolver  
    from .pso_solver import PSOSolver
except ImportError:
    from ga_solver import GASolver
    from pso_solver import PSOSolver


class AlgorithmComparison:
    """
    Framework for comparing GA and PSO algorithms on TSP instances
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize comparison framework
        
        Args:
            distance_matrix: Distance matrix for TSP instance
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.results = {}
        self.logger = logging.getLogger(__name__)

