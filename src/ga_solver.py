"""
Genetic Algorithm TSP Solver - Custom Implementation
Pure implementation of Genetic Algorithm for TSP without external metaheuristic libraries
Author: Hoàng (Team Leader)
"""

import random
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from copy import deepcopy

# TSP Solver import with comprehensive error handling
try:
    from .tsp_solver import TSPSolver  # Package import
except ImportError:
    from tsp_solver import TSPSolver  # Direct import

try:
    import config
except ImportError:
    config = None


class GASolver(TSPSolver):
    """
    Custom Genetic Algorithm implementation for TSP
    Fully implemented from scratch without DEAP or other metaheuristic libraries
    Supports depot-based routing (VRP-style TSP)
    """
    
    def __init__(self, distance_matrix: np.ndarray,
                 locations: List[dict] = None,
                 population_size: int = None,
                 generations: int = None,
                 crossover_prob: float = None,
                 mutation_prob: float = None,
                 tournament_size: int = None,
                 elite_size: int = None):
        """Initialize GA solver with custom implementation"""
        
        try:
            # Call parent class first
            super().__init__(distance_matrix, locations)
            
            # Verify logger exists after parent init
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = logging.getLogger(f"{self.__class__.__name__}")
                self.logger.warning("⚠️ Created fallback logger for GASolver")
                
        except Exception as e:
            # Last resort initialization if parent fails
            self.distance_matrix = np.array(distance_matrix)
            self.num_cities = len(distance_matrix)
            self.locations = locations or []
            self.start_point = 0
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.iteration_history = []
            self.best_distances = []
            self.logger.error(f"❌ Parent TSPSolver init failed: {e}")
        
        # Load parameters from config or use provided values
        if config and hasattr(config, 'GA_DEFAULT_CONFIG'):
            ga_config = config.GA_DEFAULT_CONFIG
        else:
            ga_config = {
                'population_size': 50,
                'generations': 100,
                'crossover_probability': 0.8,
                'mutation_probability': 0.05,
                'tournament_size': 3,
                'elite_size': 2
            }
        
        self.population_size = population_size or ga_config['population_size']
        self.generations = generations or ga_config['generations']
        self.crossover_prob = crossover_prob or ga_config['crossover_probability']
        self.mutation_prob = mutation_prob or ga_config['mutation_probability']
        self.tournament_size = tournament_size or ga_config['tournament_size']
        self.elite_size = elite_size or ga_config['elite_size']
        
        # Population tracking
        self.population = []
        self.fitness_scores = []
        
        self.logger.info(f"🧬 GA initialized (CUSTOM): pop={self.population_size}, "
                        f"gen={self.generations}, depot={getattr(self, 'start_point', 0)}")
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def create_individual(self) -> List[int]:
        """
        Create a random valid individual (route)
        Always starts with depot
        """
        depot = getattr(self, 'start_point', 0)
        other_cities = [i for i in range(self.num_cities) if i != depot]
        random.shuffle(other_cities)
        return [depot] + other_cities
    
    def initialize_population(self) -> List[List[int]]:
        """
        Create initial population of random routes
        
        Returns:
            List of individuals (routes)
        """
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        
        self.logger.debug(f"✅ Created initial population of {len(population)} individuals")
        return population
    
    # =========================================================================
    # FITNESS EVALUATION
    # =========================================================================
    
    def evaluate_fitness(self, individual: List[int]) -> float:
        """
        Evaluate fitness of an individual
        Lower distance = better fitness
        
        Args:
            individual: Route to evaluate
            
        Returns:
            Total distance (fitness score)
        """
        # Ensure route starts with depot
        if hasattr(self, 'ensure_route_starts_with_depot'):
            route = self.ensure_route_starts_with_depot(individual)
        else:
            route = individual
        
        return self.calculate_route_distance(route)
    
    def evaluate_population(self, population: List[List[int]]) -> List[float]:
        """
        Evaluate fitness for entire population
        
        Args:
            population: List of individuals
            
        Returns:
            List of fitness scores
        """
        return [self.evaluate_fitness(individual) for individual in population]
    
    # =========================================================================
    # SELECTION
    # =========================================================================
    
    def tournament_selection(self, population: List[List[int]], 
                           fitness_scores: List[float],
                           tournament_size: int = None) -> List[int]:
        """
        Tournament Selection - Custom Implementation
        Select best individual from random tournament
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for population
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual (winner)
        """
        if tournament_size is None:
            tournament_size = self.tournament_size
        
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        # Find best individual in tournament (lowest distance)
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        
        return population[best_idx].copy()
    
    def select_parents(self, population: List[List[int]], 
                      fitness_scores: List[float],
                      num_parents: int) -> List[List[int]]:
        """
        Select multiple parents using tournament selection
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            num_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        parents = []
        for _ in range(num_parents):
            parent = self.tournament_selection(population, fitness_scores)
            parents.append(parent)
        
        return parents
    
    # =========================================================================
    # CROSSOVER (LAI GHÉP)
    # =========================================================================
    
    def ordered_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Ordered Crossover (OX) - Custom Implementation
        Preserves order and position of cities from parents
        
        Args:
            parent1: First parent route
            parent2: Second parent route
            
        Returns:
            Tuple of two offspring
        """
        size = len(parent1)
        
        # Select two random crossover points
        cx_point1 = random.randint(1, size - 2)  # Skip depot (index 0)
        cx_point2 = random.randint(cx_point1 + 1, size - 1)
        
        # Initialize offspring
        child1 = [None] * size
        child2 = [None] * size
        
        # Copy depot to children
        depot = parent1[0]
        child1[0] = depot
        child2[0] = depot
        
        # Copy segment from parent1 to child1 and parent2 to child2
        child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
        
        # Fill remaining positions with genes from other parent (preserving order)
        self._fill_offspring(child1, parent2, cx_point2, size, depot)
        self._fill_offspring(child2, parent1, cx_point2, size, depot)
        
        return child1, child2
    
    def _fill_offspring(self, child: List[int], parent: List[int], 
                       start_pos: int, size: int, depot: int) -> None:
        """
        Helper function to fill remaining positions in offspring
        
        Args:
            child: Offspring to fill
            parent: Parent to take genes from
            start_pos: Starting position for filling
            size: Size of route
            depot: Depot index (to skip)
        """
        current_pos = start_pos
        
        for gene in parent:
            if gene == depot:
                continue
            
            if gene not in child:
                # Find next empty position
                while child[current_pos % size] is not None:
                    current_pos += 1
                
                child[current_pos % size] = gene
                current_pos += 1
    
    # =========================================================================
    # MUTATION (ĐỘT BIẾN)
    # =========================================================================
    
    def swap_mutation(self, individual: List[int]) -> List[int]:
        """
        Swap Mutation - Custom Implementation
        Randomly swap two cities (excluding depot)
        
        Args:
            individual: Route to mutate
            
        Returns:
            Mutated route
        """
        mutant = individual.copy()
        
        # Only mutate positions after depot (index 1 onwards)
        if len(mutant) > 2:
            pos1 = random.randint(1, len(mutant) - 1)
            pos2 = random.randint(1, len(mutant) - 1)
            
            # Ensure different positions
            while pos1 == pos2:
                pos2 = random.randint(1, len(mutant) - 1)
            
            # Swap
            mutant[pos1], mutant[pos2] = mutant[pos2], mutant[pos1]
        
        return mutant
    
    def shuffle_mutation(self, individual: List[int]) -> List[int]:
        """
        Shuffle Mutation - Custom Implementation
        Randomly shuffle a subsequence (excluding depot)
        
        Args:
            individual: Route to mutate
            
        Returns:
            Mutated route
        """
        mutant = individual.copy()
        
        if len(mutant) > 3:
            # Select random segment to shuffle (skip depot)
            start = random.randint(1, len(mutant) - 2)
            end = random.randint(start + 1, len(mutant))
            
            # Shuffle segment
            segment = mutant[start:end]
            random.shuffle(segment)
            mutant[start:end] = segment
        
        return mutant
    
    # =========================================================================
    # ELITISM
    # =========================================================================
    
    def select_elites(self, population: List[List[int]], 
                     fitness_scores: List[float],
                     num_elites: int) -> List[List[int]]:
        """
        Select best individuals (elites) from population
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            num_elites: Number of elites to select
            
        Returns:
            List of elite individuals
        """
        # Sort population by fitness (ascending - lower is better)
        sorted_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i])
        
        # Select top num_elites individuals
        elites = [population[i].copy() for i in sorted_indices[:num_elites]]
        
        return elites
    
    # =========================================================================
    # MAIN GA ALGORITHM
    # =========================================================================
    
    def solve(self) -> Tuple[List[int], float]:
        """
        Solve TSP using custom Genetic Algorithm
        
        Returns:
            Tuple of (best_route, best_distance)
        """
        try:
            self.logger.info("🧬 Starting GA optimization (CUSTOM)...")
            
            # Initialize population
            self.population = self.initialize_population()
            self.fitness_scores = self.evaluate_population(self.population)
            
            # Track best solution
            best_idx = np.argmin(self.fitness_scores)
            best_individual = self.population[best_idx].copy()
            best_fitness = self.fitness_scores[best_idx]
            
            # Store initial distance for improvement tracking
            self.initial_distance = best_fitness
            
            self.logger.info(f"Initial best distance: {best_fitness:.2f}km")
            
            # Evolution loop
            for generation in range(self.generations):
                try:
                    # Create new population
                    new_population = []
                    
                    # Elitism: Keep best individuals
                    if self.elite_size > 0:
                        elites = self.select_elites(self.population, self.fitness_scores, 
                                                   self.elite_size)
                        new_population.extend(elites)
                    
                    # Generate offspring
                    while len(new_population) < self.population_size:
                        # Selection
                        parent1 = self.tournament_selection(self.population, self.fitness_scores)
                        parent2 = self.tournament_selection(self.population, self.fitness_scores)
                        
                        # Crossover
                        if random.random() < self.crossover_prob:
                            child1, child2 = self.ordered_crossover(parent1, parent2)
                        else:
                            child1, child2 = parent1.copy(), parent2.copy()
                        
                        # Mutation
                        if random.random() < self.mutation_prob:
                            child1 = self.swap_mutation(child1)
                        
                        if random.random() < self.mutation_prob:
                            child2 = self.swap_mutation(child2)
                        
                        # Ensure depot at start
                        if hasattr(self, 'ensure_route_starts_with_depot'):
                            child1 = self.ensure_route_starts_with_depot(child1)
                            child2 = self.ensure_route_starts_with_depot(child2)
                        
                        # Add to new population
                        new_population.append(child1)
                        if len(new_population) < self.population_size:
                            new_population.append(child2)
                    
                    # Trim to exact population size
                    new_population = new_population[:self.population_size]
                    
                    # Replace old population
                    self.population = new_population
                    self.fitness_scores = self.evaluate_population(self.population)
                    
                    # Update best solution
                    current_best_idx = np.argmin(self.fitness_scores)
                    current_best_fitness = self.fitness_scores[current_best_idx]
                    
                    if current_best_fitness < best_fitness:
                        best_fitness = current_best_fitness
                        best_individual = self.population[current_best_idx].copy()
                    
                    # Log progress
                    if hasattr(self, 'log_iteration'):
                        self.log_iteration(
                            iteration=generation,
                            best_distance=best_fitness,
                            current_distance=current_best_fitness,
                            avg_fitness=np.mean(self.fitness_scores)
                        )
                    
                    # Progress logging
                    if generation % 25 == 0 or generation == self.generations - 1:
                        avg_fitness = np.mean(self.fitness_scores)
                        self.logger.info(f"Generation {generation}/{self.generations}: "
                                       f"Best={best_fitness:.2f}km, "
                                       f"Avg={avg_fitness:.2f}km")
                
                except Exception as e:
                    self.logger.error(f"❌ Generation {generation} failed: {e}")
                    continue
            
            # Ensure final route starts with depot
            if hasattr(self, 'ensure_route_starts_with_depot'):
                final_route = self.ensure_route_starts_with_depot(best_individual)
            else:
                final_route = best_individual
            
            self.logger.info(f"🧬 GA completed. Best distance: {best_fitness:.2f}km")
            
            return final_route, best_fitness
            
        except Exception as e:
            self.logger.error(f"❌ GA solve failed: {e}")
            # Return fallback solution
            fallback_route = list(range(self.num_cities))
            fallback_distance = float('inf')
            return fallback_route, fallback_distance
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_algorithm_params(self) -> Dict[str, Any]:
        """Get current algorithm parameters"""
        return {
            'algorithm': 'Genetic Algorithm (Custom)',
            'population_size': self.population_size,
            'generations': self.generations,
            'crossover_probability': self.crossover_prob,
            'mutation_probability': self.mutation_prob,
            'tournament_size': self.tournament_size,
            'elite_size': self.elite_size,
            'depot_index': getattr(self, 'start_point', 0),
            'implementation': 'Custom (No DEAP)'
        }
    
    def __str__(self) -> str:
        """String representation of the solver"""
        return f"GASolver(cities={self.num_cities}, pop={self.population_size}, gen={self.generations})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"GASolver(CUSTOM, "
                f"num_cities={self.num_cities}, "
                f"population_size={self.population_size}, "
                f"generations={self.generations}, "
                f"depot={getattr(self, 'start_point', 0)})")
