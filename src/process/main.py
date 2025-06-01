import random
from typing import Callable, Any

from src.simulator.environment import Environment
from src.simulator.domain import Access, Domain
from src.config import SimulationConfig, OptimizerStrategy
from src.simulator.simulation_engine import main as main_engine
from src.optimizer.ea import evolutionary_algorithm
from src.optimizer.iea import island_evolutionary_algorithm
from src.optimizer.greedy import greedy_algorithm


class MainProcess:
    def __init__(self, show_process: bool = False):
        self.domains = Environment.from_json_file(
            "dataset/environments/environments_supermarket.json"
        ).domains
        self.show_process = show_process

    def run(self):
        random_domain = [domain for domain in self.domains if domain.id==10][0]
        emergency_accesses, fitness_value, evals, history = evolutionary_algorithm(
            random_domain
        )
        print(history)
        print("optimizing completed!")
        print(emergency_accesses, fitness_value, evals)
