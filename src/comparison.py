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
    pass

