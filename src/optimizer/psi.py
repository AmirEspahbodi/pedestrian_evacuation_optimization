from typing import Callable
from src.simulator.domain import Domain, Pedestrian
from src.simulator.simulation_engine import main as main_engine
from src.config import (
    OptimizerStrategy,
    EAConfig,
    GreedyConfig,
    IEAConfig,
    SimulationConfig,
)
from .fitness import calculate_fitness


def psi_helper(
    num_runs: int, domain: Domain, emergency_accesses: list[tuple[int, int]]
):
    domain.add_emergency_accesses(emergency_accesses)
    sum_fitness = 0
    for _ in range(num_runs):
        domain.reset_pedestrians()
        main_engine(domain)
        domain.calculate_peds_distance_to_nearest_exit()
        sum_fitness += calculate_fitness(domain)
    domain.remove_emergency_accesses
    return sum_fitness / num_runs


def psi(
    domain: Domain, optimizer_strategy: OptimizerStrategy, emergency_accesses: list[int]
):
    new_emergency_accesses = [(i, SimulationConfig.omega) for i in emergency_accesses]

    match optimizer_strategy:
        case OptimizerStrategy.EA:
            psi_helper(EAConfig.numruns, domain, new_emergency_accesses)

        case OptimizerStrategy.IEA:
            psi_helper(IEAConfig.numruns, domain, new_emergency_accesses)

        case OptimizerStrategy.GREEDY:
            psi_helper(GreedyConfig.numruns, domain, new_emergency_accesses)
