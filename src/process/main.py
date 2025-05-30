import random
from typing import Callable, Any

from src.optimizer import calculate_fitness
from src.simulator.environment import Environment
from src.simulator.domain import Access, Domain
from src.config import SimulationConfig
from src.simulator.simulation_engine import main as main_engine
from src.optimizer.ea import EAOptimizer
from src.optimizer.iea import IEAOptimizer
from src.optimizer.cheetah import CheetahOptimizer
from src.optimizer.greedy import GreedyOptimizer

class MainProcess:
    def __init__(self, show_process: bool = False):
        self.domains = Environment.from_json_file(
            "dataset/environments/environments_supermarket.json"
        ).domains
        self.show_process = show_process

    def run(self):
        random.shuffle(self.domains)
        for domain in self.domains:
            resutl = CheetahOptimizer(domain).do()
            print(resutl)
