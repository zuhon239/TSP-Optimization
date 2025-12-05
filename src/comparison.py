"""
Algorithm Comparison Module
Compare GA vs PSO performance on TSP instances

Author: Quân ()
"""

import os
import numpy as np
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
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
        self.results = {} # Nơi lưu kết quả
        self.logger = logging.getLogger(__name__)
        
    def run_solver(self, algorithm_class, algorithm_name: str, 
                   locations: List[dict] = None, 
                   num_runs: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Run specific algorithm multiple times and collect statistics
        (Hàm chuẩn để chạy 1 thuật toán nhiều lần)
        """
        self.logger.info(f"Running {algorithm_name} for {num_runs} runs...")
        
        run_results = []
        best_overall_distance = float('inf')
        best_overall_route = None
        
        for run in range(num_runs):
            # ✅ Truyền locations để xác định Depot
            solver = algorithm_class(
                distance_matrix=self.distance_matrix,
                locations=locations,
                **kwargs
            )
            result = solver.solve_with_stats()
            
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
            'best_distance': min(distances),
            'worst_distance': max(distances),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'mean_runtime': np.mean(runtimes),
            'best_route': best_overall_route,
            'individual_runs': run_results,
            'parameters': kwargs
        }
        
        # ✅ Lưu kết quả vào biến chung để so sánh sau này
        self.results[algorithm_name] = stats
        
        self.logger.info(f"{algorithm_name} completed: Best={stats['best_distance']:.2f}km")
        return stats
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get summary comparison statistics"""
        if 'GA' not in self.results or 'PSO' not in self.results:
            return {"error": "Must run both GA and PSO first"}
        
        ga_results = self.results['GA']
        pso_results = self.results['PSO']
        
        ga_best = ga_results['best_distance']
        pso_best = pso_results['best_distance']
        
        summary = {
            'winner': 'GA' if ga_best < pso_best else 'PSO',
            'performance_gap': abs(ga_best - pso_best),
            'ga_summary': ga_results,
            'pso_summary': pso_results
        }
        
        return summary

# Convenience function for quick comparisons
def compare_algorithms(distance_matrix: np.ndarray, 
                      locations: List[dict] = None,
                      num_runs: int = 5,
                      ga_params: Dict = None,
                      pso_params: Dict = None) -> Dict:
    """
    Quick function to compare GA vs PSO algorithms
    """
    comparison = AlgorithmComparison(distance_matrix)
    
    # 1. Chạy GA
    ga_params = ga_params or {}
    comparison.run_solver(GASolver, 'GA', locations, num_runs, **ga_params)
    
    # 2. Chạy PSO
    pso_params = pso_params or {}
    comparison.run_solver(PSOSolver, 'PSO', locations, num_runs, **pso_params)
    
    # 3. Tổng hợp và trả về kết quả
    return comparison.get_comparison_summary()