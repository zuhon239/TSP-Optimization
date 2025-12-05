"""
TSP Solver Base Class
Abstract base class for TSP solving algorithms with depot support
Author: Hoàng (Team Leader)
"""

from abc import ABC, abstractmethod
import math
import numpy as np
import time
from typing import List, Tuple, Dict, Any
import logging

class TSPSolver:
    """
    Abstract base class for TSP solvers
    Provides common functionality for GA and PSO implementations with depot support
    """
    
    def __init__(self, distance_matrix: np.ndarray, locations: List[Dict] = None, **kwargs):
        """
        Initialize TSP solver with depot support
        
        Args:
            distance_matrix: Square matrix of distances between cities
            locations: List of location dictionaries with isStart flag  # ✅ ADD THIS
            **kwargs: Algorithm-specific parameters
        """
        self.distance_matrix = np.array(distance_matrix)
        self.num_cities = len(distance_matrix)
        self.locations = locations or []  # ✅ ADD THIS
        
        # Validate matrix first
        self.validate_distance_matrix()
        
        # ✅ Identify start point (depot/warehouse)
        self.start_point = self._find_start_point()
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.iteration_history = []
        self.best_distances = []
        self.initial_distance = None  # Track initial distance before optimization
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"TSP Solver initialized: {self.num_cities} cities, depot={self.start_point}")
    
    def _find_start_point(self) -> int:
        """Find the start point (depot) index from locations"""
        if not self.locations:
            self.logger.warning("No location data provided, using index 0 as depot")
            return 0
        
        # Find location with isStart=True
        for i, location in enumerate(self.locations):
            if location.get('isStart', False):
                depot_name = location.get('name', 'Unknown')
                self.logger.info(f"✅ Found depot at index {i}: {depot_name}")
                return i
        
        # If no explicit start point, use first location
        self.logger.warning("⚠️ No start point specified, using first location as depot")
        return 0
    
    def validate_distance_matrix(self) -> None:
        """Validate the distance matrix format and properties"""
        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        
        if self.num_cities < 3:
            raise ValueError("TSP requires at least 3 cities")
        
        if np.any(self.distance_matrix < 0):
            raise ValueError("Distance matrix cannot contain negative values")
        
        # Check diagonal is zero
        if not np.allclose(np.diag(self.distance_matrix), 0):
            self.logger.warning("Distance matrix diagonal is not zero")
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """
        Calculate total distance for a route WITH return to depot
        
        Args:
            route: List of city indices (complete route including depot)
            
        Returns:
            Total distance including return to depot
        """
        if len(route) != self.num_cities:
            raise ValueError(f"Route must contain exactly {self.num_cities} cities")
        
        total_distance = 0.0
        
        # ✅ Visit all cities in route order
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]  # Return to start (depot)
            total_distance += self.distance_matrix[from_city][to_city]
        
        return total_distance
    
    def ensure_route_starts_with_depot(self, route: List[int]) -> List[int]:
        """
        Ensure route starts with depot (start_point)
        
        Args:
            route: Original route
            
        Returns:
            Route starting with depot
        """
        if route[0] == self.start_point:
            return route.copy()
        
        # Find depot position and rotate
        try:
            depot_idx = route.index(self.start_point)
            rotated_route = route[depot_idx:] + route[:depot_idx]
            return rotated_route
        except ValueError:
            self.logger.error(f"Depot {self.start_point} not found in route {route}")
            return route
    
    def validate_route(self, route: List[int]) -> bool:
        """
        Validate if route is a valid TSP solution
        
        Args:
            route: List of city indices
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check length
            if len(route) != self.num_cities:
                return False
            
            # Check all cities are included exactly once
            if set(route) != set(range(self.num_cities)):
                return False
            
            # Check all indices are valid
            if any(city < 0 or city >= self.num_cities for city in route):
                return False
            
            return True
        
        except Exception:
            return False
    
    def format_route_for_display(self, route: List[int]) -> Dict:
        """
        Format route for display with location names
        
        Args:
            route: List of city indices
            
        Returns:
            Dictionary with formatted route information
        """
        # Ensure route starts with depot
        display_route = self.ensure_route_starts_with_depot(route)
        
        if not self.locations:
            location_names = [f"City {i}" for i in range(self.num_cities)]
        else:
            location_names = [loc.get('name', f'Location {i}') for i, loc in enumerate(self.locations)]
        
        route_names = [location_names[i] for i in display_route]
        total_distance = self.calculate_route_distance(display_route)
        
        # Create route display string
        route_display = " → ".join(route_names)
        route_display += f" → {location_names[self.start_point]} (Return)"
        
        return {
            'route_indices': display_route,
            'route_names': route_names,
            'total_distance': total_distance,
            'start_point': self.start_point,
            'start_point_name': location_names[self.start_point],
            'num_stops': len(display_route),
            'formatted_route': route_display,
            'is_valid': self.validate_route(display_route)
        }
    
    def start_timer(self) -> None:
        """Start timing the algorithm execution"""
        self.start_time = time.time()
    
    def end_timer(self) -> None:
        """End timing the algorithm execution"""
        self.end_time = time.time()
    
    def get_runtime(self) -> float:
        """Get algorithm runtime in seconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def log_iteration(self, iteration: int, best_distance: float, **kwargs) -> None:
        """
        Log iteration progress
        
        Args:
            iteration: Current iteration number
            best_distance: Best distance found so far
            **kwargs: Additional algorithm-specific metrics
        """
        self.iteration_history.append({
            'iteration': iteration,
            'best_distance': best_distance,
            'timestamp': time.time(),
            **kwargs
        })
        self.best_distances.append(best_distance)
    
    def get_convergence_data(self) -> Dict[str, List]:
        """Get convergence history for visualization"""
        return {
            'iterations': [h['iteration'] for h in self.iteration_history],
            'best_distances': self.best_distances,
            'timestamps': [h['timestamp'] for h in self.iteration_history]
        }
    
    @abstractmethod
    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the TSP problem
        
        Returns:
            Tuple of (best_route, best_distance)
        """
        pass
    
    def solve_with_stats(self) -> Dict[str, Any]:
        """
        Solve TSP and return detailed statistics
        
        Returns:
            Dictionary containing solution and performance metrics
        """
        self.start_timer()
        
        try:
            best_route, best_distance = self.solve()
            self.end_timer()
            
            # Validate solution
            if not self.validate_route(best_route):
                raise ValueError("Algorithm returned invalid route")
            
            # Ensure route starts with depot for display
            display_route = self.ensure_route_starts_with_depot(best_route)
            route_info = self.format_route_for_display(display_route)
            
            # Calculate statistics
            improvement_pct = 0.0
            if self.initial_distance and self.initial_distance > 0:
                improvement_pct = ((self.initial_distance - best_distance) / self.initial_distance) * 100

            stats = {
                'algorithm': self.__class__.__name__,
                'best_route': display_route,
                'best_distance': best_distance,
                'initial_distance': self.initial_distance if self.initial_distance else best_distance,
                'improvement_percentage': improvement_pct,
                'runtime_seconds': self.get_runtime(),
                'num_cities': self.num_cities,
                'num_iterations': len(self.iteration_history),
                'convergence_data': self.get_convergence_data(),
                'route_info': route_info,
                'success': True,
                'error_message': None
            }
            
            self.logger.info(f"✅ TSP solved successfully: distance={best_distance:.2f}km, "
                           f"runtime={stats['runtime_seconds']:.2f}s")
            
            return stats
            
        except Exception as e:
            self.end_timer()
            self.logger.error(f"❌ TSP solving failed: {str(e)}")
            
            best_distance = float('inf')
            improvement_pct = 0.0
            if self.initial_distance and self.initial_distance > 0 and math.isfinite(self.initial_distance) and math.isfinite(best_distance):
                improvement_pct = ((self.initial_distance - best_distance) / self.initial_distance) * 100
            
            return {
                'algorithm': self.__class__.__name__,
                'best_route': None,
                'best_distance': float('inf'),
                'initial_distance': self.initial_distance,
                'improvement_percentage': improvement_pct,
                'runtime_seconds': self.get_runtime(),
                'num_cities': self.num_cities,
                'num_iterations': len(self.iteration_history),
                'convergence_data': self.get_convergence_data(),
                'route_info': None,
                'success': False,
                'error_message': str(e)
            }
    
    def __str__(self) -> str:
        """String representation of the solver"""
        return f"{self.__class__.__name__}(cities={self.num_cities}, depot={self.start_point})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}("
                f"num_cities={self.num_cities}, "
                f"depot={self.start_point}, "
                f"matrix_shape={self.distance_matrix.shape})")