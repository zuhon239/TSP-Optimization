"""
Genetic Algorithm TSP Solver using DEAP 1.4.3
Implementation of Genetic Algorithm for Travelling Salesman Problem with depot support
Author: Hoàng (Team Leader)
"""

import random
import numpy as np
import logging
from typing import List, Tuple

# DEAP framework
from deap import base, creator, tools, algorithms

# TSP Solver import with comprehensive error handling
try:
    from .tsp_solver import TSPSolver  # Package import
except ImportError:  
    from tsp_solver import TSPSolver  # Direct import  
import config
class GASolver(TSPSolver):
    """
    Genetic Algorithm implementation for TSP using DEAP 1.4.3
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
        """Initialize GA solver with DEAP and depot support"""
        
        try:
            # ✅ Call parent class first
            super().__init__(distance_matrix, locations)
            
            # ✅ Verify logger exists after parent init
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = logging.getLogger(f"{self.__class__.__name__}")
                self.logger.warning("⚠️ Created fallback logger for GASolver")
            
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
        
        # Load parameters from config or use provided values
        try:
            ga_config = config.GA_DEFAULT_CONFIG
        except (AttributeError, NameError):
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
        
        # Setup DEAP framework
        self.setup_deap()
        
        # ✅ Now safe to use logger (guaranteed to exist)
        self.logger.info(f"🧬 GA initialized: pop={self.population_size}, "
                        f"gen={self.generations}, depot={getattr(self, 'start_point', 0)}")
    
    def setup_deap(self) -> None:
        """Setup DEAP framework for TSP with depot support"""
        try:
            # Clear any existing classes (important for re-runs)
            if hasattr(creator, "FitnessMin"):
                del creator.FitnessMin
            if hasattr(creator, "Individual"):
                del creator.Individual
            
            # Create fitness class (minimization problem)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            
            # Create individual class
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            # Initialize toolbox
            self.toolbox = base.Toolbox()
            
            # ✅ Register depot-aware individual creation
            self.toolbox.register("indices", self._create_depot_route)
            self.toolbox.register("individual", tools.initIterate, 
                                 creator.Individual, self.toolbox.indices)
            self.toolbox.register("population", tools.initRepeat, 
                                 list, self.toolbox.individual)
            
            # Genetic operators
            self.toolbox.register("mate", tools.cxOrdered)
            self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.mutation_prob)
            self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
            self.toolbox.register("evaluate", self.evaluate_individual)
            
            # Statistics
            self.stats = tools.Statistics(lambda ind: ind.fitness.values)
            self.stats.register("avg", np.mean)
            self.stats.register("min", np.min)
            self.stats.register("max", np.max)
            self.stats.register("std", np.std)
            
            self.logger.debug("✅ DEAP framework setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ DEAP setup failed: {e}")
            raise e
    
    def _create_depot_route(self) -> List[int]:
        """Create a valid route that starts from depot"""
        try:
            # Get all cities except depot
            depot = getattr(self, 'start_point', 0)
            other_cities = [i for i in range(self.num_cities) if i != depot]
            
            # Randomize other cities
            random.shuffle(other_cities)
            
            # ✅ Always start with depot
            route = [depot] + other_cities
            
            return route
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create depot route: {e}")
            # Fallback to simple route
            return list(range(self.num_cities))
    
    def evaluate_individual(self, individual: List[int]) -> Tuple[float]:
        """
        Evaluate fitness of an individual (route)
        Ensures route starts from depot
        """
        try:
            # ✅ Ensure route starts with depot
            if hasattr(self, 'ensure_route_starts_with_depot'):
                route = self.ensure_route_starts_with_depot(individual)
            else:
                route = individual
                
            distance = self.calculate_route_distance(route)
            return (distance,)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate individual: {e}")
            return (float('inf'),)  # Worst possible fitness
    
    def solve(self) -> Tuple[List[int], float]:
        """Solve TSP using Genetic Algorithm with depot support"""
        try:
            self.logger.info("🧬 Starting GA optimization...")
            
            # Create initial population
            population = self.toolbox.population(n=self.population_size)
            self.logger.debug(f"Created population of {len(population)} individuals")
            
            # Evaluate the entire population
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Track best solution
            best_individual = None
            best_fitness = float('inf')
            
            # Evolution loop
            for generation in range(self.generations):
                try:
                    # Select, crossover, mutate
                    offspring = self.toolbox.select(population, len(population))
                    offspring = list(map(self.toolbox.clone, offspring))
                    
                    # Apply crossover and mutation
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < self.crossover_prob:
                            self.toolbox.mate(child1, child2)
                            # ✅ Ensure both children start with depot
                            if hasattr(self, 'ensure_route_starts_with_depot'):
                                child1[:] = self.ensure_route_starts_with_depot(child1)
                                child2[:] = self.ensure_route_starts_with_depot(child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    
                    for mutant in offspring:
                        if random.random() < self.mutation_prob:
                            self.toolbox.mutate(mutant)
                            # ✅ Ensure mutant starts with depot
                            if hasattr(self, 'ensure_route_starts_with_depot'):
                                mutant[:] = self.ensure_route_starts_with_depot(mutant)
                            del mutant.fitness.values
                    
                    # Evaluate invalid individuals
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = map(self.toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                    
                    # Elitism: preserve best individuals
                    if self.elite_size > 0:
                        elite = tools.selBest(population, self.elite_size)
                        offspring[-self.elite_size:] = elite
                    
                    population = offspring
                    
                    # Track statistics
                    current_best = tools.selBest(population, 1)[0]
                    current_best_fitness = current_best.fitness.values[0]
                    
                    if current_best_fitness < best_fitness:
                        best_fitness = current_best_fitness
                        best_individual = current_best[:]
                    
                    # Log progress
                    if hasattr(self, 'log_iteration'):
                        self.log_iteration(
                            iteration=generation,
                            best_distance=best_fitness,
                            current_distance=current_best_fitness,
                            avg_fitness=np.mean([ind.fitness.values[0] for ind in population])
                        )
                    
                    # Progress logging
                    if generation % 25 == 0 or generation == self.generations - 1:
                        avg_fitness = np.mean([ind.fitness.values[0] for ind in population])
                        self.logger.info(f"Generation {generation}: "
                                       f"Best={best_fitness:.2f}km, "
                                       f"Avg={avg_fitness:.2f}km")
                        
                except Exception as e:
                    self.logger.error(f"❌ Generation {generation} failed: {e}")
                    continue
            
            # ✅ Ensure final route starts with depot
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
    
    def get_algorithm_params(self) -> dict:
        """Get current algorithm parameters"""
        return {
            'algorithm': 'Genetic Algorithm',
            'population_size': self.population_size,
            'generations': self.generations,
            'crossover_probability': self.crossover_prob,
            'mutation_probability': self.mutation_prob,
            'tournament_size': self.tournament_size,
            'elite_size': self.elite_size,
            'depot_index': getattr(self, 'start_point', 0)
        }
    
    def __str__(self) -> str:
        """String representation of the solver"""
        return f"GASolver(cities={self.num_cities}, pop={self.population_size}, gen={self.generations})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"GASolver("
                f"num_cities={self.num_cities}, "
                f"population_size={self.population_size}, "
                f"generations={self.generations}, "
                f"depot={getattr(self, 'start_point', 0)})")