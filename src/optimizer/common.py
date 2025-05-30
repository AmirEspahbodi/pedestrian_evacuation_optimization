from typing import List, Callable
from src.simulator.domain import Domain
from .psi import psi as psi_function
from src.config import OptimizerStrategy


class FitnessEvaluator:
    def __init__(self, domain: Domain, optimizer_strategy: OptimizerStrategy):
        self.domain = domain
        self.evaluations: int = 0
        self.optimizer_strategy: OptimizerStrategy = optimizer_strategy

    def evaluate(self, emergency_accesses: List[float]) -> float:
        if not emergency_accesses:
            return float("inf")
        self.evaluations += 1
        return psi_function(self.domain, self.optimizer_strategy, emergency_accesses)

    def get_evaluation_count(self) -> int:
        return self.evaluations


class Individual:
    def __init__(self, genes: List[float]):
        self.genes: List[float] = genes
        self.fitness: float = float("inf")

    def __lt__(self, other: "Individual") -> bool:
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        return f"Individual(genes={self.genes}, fitness={self.fitness})"
