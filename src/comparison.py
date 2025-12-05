"""
Algorithm Comparison Module - Comprehensive GA vs PSO Comparison
Compare Genetic Algorithm and Particle Swarm Optimization performance on TSP instances

Features:
- Single-run comparison (fast comparison mode)
- Multi-run comparison with statistical analysis
- Detailed convergence tracking
- Performance metrics calculation
- Parameter comparison
- Winner determination based on multiple criteria

Author: Hoàng (Team Leader)
"""

import os
import numpy as np
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import config
except ImportError:
    config = None

try:
    from .ga_solver import GASolver  
    from .pso_solver import PSOSolver
except ImportError:
    from ga_solver import GASolver
    from pso_solver import PSOSolver


class AlgorithmComparison:
    """
    Comprehensive framework for comparing GA and PSO algorithms on TSP instances
    
    Features:
    - Run single or multiple iterations of each algorithm
    - Track detailed performance metrics
    - Compare convergence behavior
    - Calculate statistical differences
    - Determine winners based on multiple criteria
    """
    
    def __init__(self, distance_matrix: np.ndarray, locations: List[Dict] = None):
        """
        Initialize comparison framework
        
        Args:
            distance_matrix: NxN square matrix of distances between cities
            locations: List of location dictionaries with location metadata
        """
        self.distance_matrix = np.array(distance_matrix)
        self.num_cities = len(distance_matrix)
        self.locations = locations or []
        self.results = {}  # Store results for both algorithms
        self.ga_results = None
        self.pso_results = None
        self.comparison_data = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Comparison framework initialized for {self.num_cities} cities")
    
    def _map_ga_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map parameter names from config format to GASolver constructor format
        
        Args:
            params: Parameters with potentially config format names
            
        Returns:
            Parameters with correct GASolver constructor format
        """
        mapped = params.copy()
        
        # Map config parameter names to GASolver constructor names
        if 'crossover_probability' in mapped:
            mapped['crossover_prob'] = mapped.pop('crossover_probability')
        if 'mutation_probability' in mapped:
            mapped['mutation_prob'] = mapped.pop('mutation_probability')
        
        return mapped
    
    def run_single_comparison(self, 
                             ga_params: Dict[str, Any] = None,
                             pso_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run single iteration of both GA and PSO (fast comparison mode)
        
        Args:
            ga_params: GA algorithm parameters
            pso_params: PSO algorithm parameters
            
        Returns:
            Dictionary with results from both algorithms
        """
        self.logger.info("Starting single-run comparison (fast mode)...")
        
        ga_params = ga_params or {}
        pso_params = pso_params or {}
        
        # Map parameters to correct constructor format
        ga_params_mapped = self._map_ga_params(ga_params)
        
        # Run GA
        try:
            ga_solver = GASolver(
                distance_matrix=self.distance_matrix,
                locations=self.locations,
                **ga_params_mapped
            )
            self.ga_results = ga_solver.solve_with_stats()
            self.ga_results['parameters'] = ga_params
            self.logger.info(f"GA completed: {self.ga_results['best_distance']:.2f}km in {self.ga_results['runtime_seconds']:.4f}s")
        except Exception as e:
            self.logger.error(f"GA failed: {str(e)}")
            self.ga_results = {
                'success': False,
                'error_message': str(e),
                'algorithm': 'Genetic Algorithm',
                'parameters': ga_params
            }
        
        # Run PSO
        try:
            pso_solver = PSOSolver(
                distance_matrix=self.distance_matrix,
                locations=self.locations,
                **pso_params
            )
            self.pso_results = pso_solver.solve_with_stats()
            self.pso_results['parameters'] = pso_params
            self.logger.info(f"PSO completed: {self.pso_results['best_distance']:.2f}km in {self.pso_results['runtime_seconds']:.4f}s")
        except Exception as e:
            self.logger.error(f"PSO failed: {str(e)}")
            self.pso_results = {
                'success': False,
                'error_message': str(e),
                'algorithm': 'Particle Swarm Optimization',
                'parameters': pso_params
            }
        
        # Generate comparison data
        self._generate_comparison_data()
        
        return self._build_results_dict()
    
    def run_multi_comparison(self, 
                            ga_params: Dict[str, Any] = None,
                            pso_params: Dict[str, Any] = None,
                            num_runs: int = 3) -> Dict[str, Any]:
        """
        Run multiple iterations of both algorithms with statistical analysis
        
        Args:
            ga_params: GA algorithm parameters
            pso_params: PSO algorithm parameters
            num_runs: Number of runs for each algorithm (default: 3)
            
        Returns:
            Dictionary with aggregated results and statistics
        """
        self.logger.info(f"Starting multi-run comparison ({num_runs} runs each)...")
        
        ga_params = ga_params or {}
        pso_params = pso_params or {}
        
        # Map GA parameters to correct constructor format
        ga_params_mapped = self._map_ga_params(ga_params)
        
        # Run GA multiple times
        ga_runs = []
        for i in range(num_runs):
            try:
                solver = GASolver(
                    distance_matrix=self.distance_matrix,
                    locations=self.locations,
                    **ga_params_mapped
                )
                result = solver.solve_with_stats()
                ga_runs.append(result)
                self.logger.debug(f"GA run {i+1}/{num_runs}: {result['best_distance']:.2f}km")
            except Exception as e:
                self.logger.warning(f"GA run {i+1} failed: {str(e)}")
        
        # Run PSO multiple times
        pso_runs = []
        for i in range(num_runs):
            try:
                solver = PSOSolver(
                    distance_matrix=self.distance_matrix,
                    locations=self.locations,
                    **pso_params
                )
                result = solver.solve_with_stats()
                pso_runs.append(result)
                self.logger.debug(f"PSO run {i+1}/{num_runs}: {result['best_distance']:.2f}km")
            except Exception as e:
                self.logger.warning(f"PSO run {i+1} failed: {str(e)}")
        
        # Aggregate GA results
        self.ga_results = self._aggregate_results(ga_runs, "Genetic Algorithm", ga_params)
        
        # Aggregate PSO results
        self.pso_results = self._aggregate_results(pso_runs, "Particle Swarm Optimization", pso_params)
        
        # Generate comparison data
        self._generate_comparison_data()
        
        return self._build_results_dict()
    
    def _aggregate_results(self, runs: List[Dict], algorithm_name: str, 
                          parameters: Dict) -> Dict[str, Any]:
        """
        Aggregate results from multiple algorithm runs
        
        Args:
            runs: List of individual run results
            algorithm_name: Name of the algorithm
            parameters: Algorithm parameters
            
        Returns:
            Aggregated results dictionary
        """
        successful_runs = [r for r in runs if r.get('success', False)]
        
        if not successful_runs:
            return {
                'success': False,
                'error_message': 'All runs failed',
                'algorithm': algorithm_name,
                'parameters': parameters
            }
        
        distances = [r['best_distance'] for r in successful_runs]
        runtimes = [r['runtime_seconds'] for r in successful_runs]
        
        # Find best run
        best_run = successful_runs[np.argmin(distances)]
        
        return {
            'success': True,
            'algorithm': algorithm_name,
            'parameters': parameters,
            'num_runs': len(runs),
            'successful_runs': len(successful_runs),
            'best_distance': np.min(distances),
            'worst_distance': np.max(distances),
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'min_runtime': np.min(runtimes),
            'max_runtime': np.max(runtimes),
            'mean_runtime': np.mean(runtimes),
            'best_route': best_run['best_route'],
            'best_run_data': best_run,
            'all_runs': successful_runs
        }
    
    def _generate_comparison_data(self) -> None:
        """Generate comprehensive comparison metrics"""
        if not self.ga_results or not self.pso_results:
            self.comparison_data = None
            return
        
        # Validate both completed successfully
        if not self.ga_results.get('success') or not self.pso_results.get('success'):
            self.logger.warning("One or both algorithms failed - limited comparison available")
            self.comparison_data = {
                'valid': False,
                'ga_success': self.ga_results.get('success', False),
                'pso_success': self.pso_results.get('success', False)
            }
            return
        
        ga_dist = self.ga_results['best_distance']
        pso_dist = self.pso_results['best_distance']
        ga_time = self.ga_results['runtime_seconds']
        pso_time = self.pso_results['runtime_seconds']
        
        # Determine winners
        winner_distance = 'GA' if ga_dist < pso_dist else 'PSO' if pso_dist < ga_dist else 'Tie'
        winner_runtime = 'GA' if ga_time < pso_time else 'PSO' if pso_time < ga_time else 'Tie'
        
        # Calculate improvement metrics
        ga_iterations = self.ga_results.get('num_iterations', 0)
        pso_iterations = self.pso_results.get('num_iterations', 0)
        
        # Get initial distances from convergence data
        ga_convergence = self.ga_results.get('convergence_data', {})
        pso_convergence = self.pso_results.get('convergence_data', {})
        
        ga_initial = ga_convergence.get('best_distances', [ga_dist])[0] if ga_convergence.get('best_distances') else ga_dist
        pso_initial = pso_convergence.get('best_distances', [pso_dist])[0] if pso_convergence.get('best_distances') else pso_dist
        
        ga_improvement = ((ga_initial - ga_dist) / ga_initial * 100) if ga_initial > 0 else 0
        pso_improvement = ((pso_initial - pso_dist) / pso_initial * 100) if pso_initial > 0 else 0
        
        # Calculate convergence speed (iterations to reach 95% improvement)
        ga_conv_speed = self._calculate_convergence_speed(ga_convergence, 0.95)
        pso_conv_speed = self._calculate_convergence_speed(pso_convergence, 0.95)
        
        self.comparison_data = {
            'valid': True,
            'ga_distance': ga_dist,
            'pso_distance': pso_dist,
            'ga_runtime': ga_time,
            'pso_runtime': pso_time,
            'ga_iterations': ga_iterations,
            'pso_iterations': pso_iterations,
            'distance_difference': abs(ga_dist - pso_dist),
            'distance_difference_pct': (abs(ga_dist - pso_dist) / min(ga_dist, pso_dist) * 100) if min(ga_dist, pso_dist) > 0 else 0,
            'runtime_difference': abs(ga_time - pso_time),
            'runtime_difference_pct': (abs(ga_time - pso_time) / max(ga_time, pso_time) * 100) if max(ga_time, pso_time) > 0 else 0,
            'winner_distance': winner_distance,
            'winner_runtime': winner_runtime,
            'ga_improvement_pct': ga_improvement,
            'pso_improvement_pct': pso_improvement,
            'winner_improvement': 'GA' if ga_improvement > pso_improvement else 'PSO' if pso_improvement > ga_improvement else 'Tie',
            'ga_convergence_speed': ga_conv_speed,
            'pso_convergence_speed': pso_conv_speed,
            'winner_convergence': 'GA' if ga_conv_speed > 0 and (pso_conv_speed == 0 or ga_conv_speed < pso_conv_speed) else 'PSO' if pso_conv_speed > 0 else 'Unknown'
        }
        
        self.logger.info(f"Comparison complete: {winner_distance} wins on distance, {winner_runtime} wins on runtime")
    
    def _calculate_convergence_speed(self, convergence_data: Dict, threshold: float = 0.95) -> int:
        """
        Calculate convergence speed as iterations needed to reach target improvement
        
        Args:
            convergence_data: Convergence history dictionary
            threshold: Improvement threshold (0.95 = 95%)
            
        Returns:
            Iteration count or -1 if not converged
        """
        distances = convergence_data.get('best_distances', [])
        if not distances or len(distances) < 2:
            return -1
        
        initial = distances[0]
        target = initial * (1 - threshold)
        
        for i, dist in enumerate(distances):
            if dist <= target:
                return i
        
        return -1  # Did not converge to threshold
    
    def _build_results_dict(self) -> Dict[str, Any]:
        """Build final results dictionary"""
        return {
            'ga_results': self.ga_results,
            'pso_results': self.pso_results,
            'comparison': self.comparison_data,
            'timestamp': time.time()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get high-level comparison summary"""
        if not self.comparison_data or not self.comparison_data.get('valid'):
            return {
                'valid': False,
                'message': 'Comparison data not available or incomplete'
            }
        
        return {
            'valid': True,
            'best_algorithm': self.comparison_data['winner_distance'],
            'best_distance': min(
                self.ga_results.get('best_distance', float('inf')),
                self.pso_results.get('best_distance', float('inf'))
            ),
            'fastest_algorithm': self.comparison_data['winner_runtime'],
            'distance_gap': self.comparison_data['distance_difference'],
            'runtime_gap': self.comparison_data['runtime_difference']
        }


def compare_algorithms(distance_matrix: np.ndarray, 
                      locations: List[Dict] = None,
                      ga_params: Dict[str, Any] = None,
                      pso_params: Dict[str, Any] = None,
                      multi_run: bool = False,
                      num_runs: int = 3) -> Dict[str, Any]:
    """
    Convenience function for quick algorithm comparison
    
    Args:
        distance_matrix: Distance matrix for TSP
        locations: Location metadata
        ga_params: GA parameters (optional)
        pso_params: PSO parameters (optional)
        multi_run: Run multiple iterations (default: False for speed)
        num_runs: Number of runs per algorithm (if multi_run=True)
        
    Returns:
        Comparison results dictionary
    """
    comparison = AlgorithmComparison(distance_matrix, locations)
    
    if multi_run:
        return comparison.run_multi_comparison(ga_params, pso_params, num_runs)
    else:
        return comparison.run_single_comparison(ga_params, pso_params)