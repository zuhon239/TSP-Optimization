"""
Particle Swarm Optimization TSP Solver

Custom implementation of PSO for Travelling Salesman Problem with depot support
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
    
    def _create_depot_route(self) -> List[int]:
        """Create a random route that starts from depot"""
        other_cities = [i for i in range(self.num_cities) if i != getattr(self, 'start_point', 0)]
        random.shuffle(other_cities)
        return [getattr(self, 'start_point', 0)] + other_cities
    
    def solve(self) -> Tuple[List[int], float]:
        """Solve TSP using PSO with depot support"""
        self.logger.info("🐝 Starting PSO optimization...")
        
        # Initialize swarm
        self.particles = []
        self.personal_best = []
        self.personal_best_fitness = []
        
        for _ in range(self.swarm_size):
            particle = self._create_depot_route()
            fitness = self.calculate_route_distance(particle)
            
            self.particles.append(particle)
            self.personal_best.append(particle[:])
            self.personal_best_fitness.append(fitness)
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = particle[:]
        
        # PSO main loop
        for iteration in range(self.max_iterations):
            
            # Simple PSO update (simplified for TSP)
            for i in range(self.swarm_size):
                
                # Apply random swaps based on personal and global best
                if random.random() < 0.5:  # Cognitive component
                    self._apply_random_swaps(self.particles[i], self.personal_best[i])
                
                if random.random() < 0.5:  # Social component
                    self._apply_random_swaps(self.particles[i], self.global_best)
                
                # Evaluate fitness
                fitness = self.calculate_route_distance(self.particles[i])
                
                # Update personal best
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best[i] = self.particles[i][:]
                    self.personal_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best = self.particles[i][:]
            
            # Log progress
            if hasattr(self, 'log_iteration'):
                self.log_iteration(iteration=iteration, best_distance=self.global_best_fitness)
            
            if iteration % 25 == 0 or iteration == self.max_iterations - 1:
                self.logger.info(f"Iteration {iteration}: Best={self.global_best_fitness:.2f}km")
        
        # ✅ Ensure final route starts with depot
        if hasattr(self, 'ensure_route_starts_with_depot'):
            final_route = self.ensure_route_starts_with_depot(self.global_best)
        else:
            final_route = self.global_best
        
        self.logger.info(f"🐝 PSO completed. Best distance: {self.global_best_fitness:.2f}km")
        
        return final_route, self.global_best_fitness
    
    def _apply_random_swaps(self, route1: List[int], route2: List[int], max_swaps: int = 2):
        """
        Apply directed swaps to move route1 closer to route2
        (Thực hiện hoán đổi có định hướng để route1 dần dần giống route2)
        Args:
            route1: Lộ trình cần sửa (Hạt hiện tại)
            route2: Lộ trình mẫu (PBest hoặc GBest)
            max_swaps: Số lần hoán đổi tối đa (Tương đương vận tốc)
        """
        swaps_done = 0
        
        # Duyệt qua từng vị trí trong lộ trình (bỏ qua Depot ở vị trí 0)
        for i in range(1, len(route1)):
            # Nếu đã hoán đổi đủ số lượng cho phép thì dừng lại (Giữ lại sự đa dạng)
            if swaps_done >= max_swaps:
                break
                
            # Nếu tại vị trí i, thành phố của mình khác thành phố mẫu
            if route1[i] != route2[i]:
                target_city = route2[i]
                
                # Tìm vị trí hiện tại của thành phố đó trong route1
                # (Chỉ tìm từ i trở về sau để không phá hỏng phần đã làm xong)
                try:
                    current_pos = -1
                    # Tìm thủ công hoặc dùng index (nhưng phải cẩn thận vùng tìm kiếm)
                    for k in range(i, len(route1)):
                        if route1[k] == target_city:
                            current_pos = k
                            break
                    
                    if current_pos != -1:
                        # Thực hiện hoán đổi (Swap)
                        route1[i], route1[current_pos] = route1[current_pos], route1[i]
                        swaps_done += 1
                        
                except ValueError:
                    continue