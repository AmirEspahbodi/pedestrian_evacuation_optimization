from typing import List

from .psi import psi as psi_function


class FitnessEvaluator:
    def __init__(self, gird, pedestrians_confs, simulator_config):
        self.evaluations: int = 0
        self.pedestrians_confs = pedestrians_confs
        self.gird = gird
        self.simulator_config = simulator_config

    def evaluate(self, emergency_accesses) -> float:
        emergency_accesses = [int(i) for i in emergency_accesses]
        if not emergency_accesses:
            return float("inf")
        self.evaluations += 1
        fitness_value = psi_function(
            self.gird, self.pedestrians_confs, emergency_accesses, self.simulator_config
        )
        print(
            f"     evaluation cuont={self.evaluations},     fitness value={fitness_value}"
        )
        return fitness_value

    def get_evaluation_count(self) -> int:
        return self.evaluations


class Individual:
    def __init__(self, genes: List[int]):
        self.genes: List[int] = genes
        self.fitness: float = float("inf")

    def __lt__(self, other: "Individual") -> bool:
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        return f"Individual(genes={self.genes}, fitness={self.fitness})"
