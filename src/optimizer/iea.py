import math
import random
import numpy as np
from typing import List, Callable, Any, Tuple, Set
from src.simulator.domain import Domain
from src.config import SimulationConfig, OptimizerStrategy
from .common import FitnessEvaluator, Individual
from .psi import psi as psi_function


class IEAOptimizer:
    def __init__(self, domain: Domain):
        self.domain = domain

    def do(self):
        raise NotImplementedError("main finction to run algorithm, not iemented")
