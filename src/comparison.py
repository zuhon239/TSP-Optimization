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
    
    def run_algorithm(self, algorithm_class, algorithm_name: str, 
                     num_runs: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Run algorithm multiple times and collect statistics
        
        Args:
            algorithm_class: Algorithm class (GASolver or PSOSolver)
            algorithm_name: Name for logging/results
            num_runs: Number of independent runs
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Aggregated results dictionary
        """
        self.logger.info(f"Running {algorithm_name} for {num_runs} runs...")
        
        run_results = []
        best_overall_distance = float('inf')
        best_overall_route = None
        
        for run in range(num_runs):
            self.logger.info(f"  Run {run + 1}/{num_runs}")
            
            # Create and run algorithm
            solver = algorithm_class(self.distance_matrix, **kwargs)
            result = solver.solve_with_stats()
            
            # Track best overall solution
            if result['best_distance'] < best_overall_distance:
                best_overall_distance = result['best_distance']
                best_overall_route = result['best_route']
            
            run_results.append(result)
        
        # Calculate statistics
        distances = [r['best_distance'] for r in run_results if r['success']]
        runtimes = [r['runtime_seconds'] for r in run_results if r['success']]
        
        if not distances:
            self.logger.error(f"All runs failed for {algorithm_name}")
            return None
        
        stats = {
            'algorithm': algorithm_name,
            'num_runs': num_runs,
            'successful_runs': len(distances),
            'success_rate': len(distances) / num_runs,
            
            # Distance statistics
            'best_distance': min(distances),
            'worst_distance': max(distances),
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            
            # Runtime statistics  
            'mean_runtime': np.mean(runtimes),
            'median_runtime': np.median(runtimes),
            'std_runtime': np.std(runtimes),
            
            # Best solution
            'best_route': best_overall_route,
            
            # All run results
            'individual_runs': run_results,
            
            # Algorithm parameters
            'parameters': run_results[0]['algorithm'] if run_results else {}
        }
        
        self.logger.info(f"{algorithm_name} completed: "
                        f"Best={stats['best_distance']:.2f}, "
                        f"Mean={stats['mean_distance']:.2f}±{stats['std_distance']:.2f}")
        
        return stats

