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
    
    def run_comparison(self, algorithm_class, algorithm_name: str,
                 locations: List[dict] = None,
                 num_runs: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive comparison between GA and PSO
        
        Args:
            num_runs: Number of runs per algorithm
            ga_params: GA-specific parameters
            pso_params: PSO-specific parameters
            parallel: Whether to run algorithms in parallel
            
        Returns:
            Dictionary containing results for both algorithms
        """
        self.logger.info(f"Running {algorithm_name} for {num_runs} runs...")
    
        run_results = []
        best_overall_distance = float('inf')
        best_overall_route = None
        
        for run in range(num_runs):
            self.logger.info(f"  Run {run + 1}/{num_runs}")
            
            # ✅ Create and run algorithm with locations
            solver = algorithm_class(
                distance_matrix=self.distance_matrix,
                locations=locations,
                **kwargs
            )
            result = solver.solve_with_stats()
            
            # Track best overall solution
            if result['best_distance'] < best_overall_distance:
                best_overall_distance = result['best_distance']
                best_overall_route = result['best_route']
            
            run_results.append(result)
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Get summary comparison statistics
        
        Returns:
            Summary statistics comparing the algorithms
        """
        if not self.results or 'GA' not in self.results or 'PSO' not in self.results:
            raise ValueError("Must run comparison first")
        
        ga_results = self.results['GA']
        pso_results = self.results['PSO']
        
        # Performance comparison
        ga_best = ga_results['best_distance']
        pso_best = pso_results['best_distance']
        
        summary = {
            'winner': 'GA' if ga_best < pso_best else 'PSO',
            'performance_gap': abs(ga_best - pso_best),
            'performance_gap_percent': abs(ga_best - pso_best) / min(ga_best, pso_best) * 100,
            
            'ga_summary': {
                'best_distance': ga_best,
                'mean_distance': ga_results['mean_distance'],
                'std_distance': ga_results['std_distance'],
                'mean_runtime': ga_results['mean_runtime'],
                'success_rate': ga_results['success_rate']
            },
            
            'pso_summary': {
                'best_distance': pso_best,
                'mean_distance': pso_results['mean_distance'],
                'std_distance': pso_results['std_distance'],
                'mean_runtime': pso_results['mean_runtime'],
                'success_rate': pso_results['success_rate']
            }
        }
        
        return summary
    
    def export_results(self, filename: str = None) -> str:
        """
        Export comparison results to JSON file
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"tsp_comparison_{self.num_cities}cities_{timestamp}.json"
        
        filepath = os.path.join(config.RESULTS_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"Results exported to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {str(e)}")
            raise
