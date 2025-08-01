from typing import List
import math
from src.simulator.domain import Domain, Pedestrian
from src.simulator.simulation_engine import main as main_engine
from src.config import (
    OptimizerStrategy,
    EAConfig,
    GreedyConfig,
    IEAConfig,
    SimulationConfig,
)
from src.simulator.domain import Domain, Pedestrian
from src.config import SimulationConfig


def calculate_fitness(
    domain: Domain,
) -> float:
    w1, w2, w3, w4, w5 = 3, 2, 1, 1, 2
    pedestrians = domain.get_pedestrians()
    num_total_pedestrians = len(pedestrians)
    evacuees: List[Pedestrian] = []
    non_evacuees: List[Pedestrian] = []

    for p in pedestrians:
        if p.is_exited:
            evacuees.append(p)
        else:
            non_evacuees.append(p)

    D_diagonal = math.sqrt(domain.width**2 + domain.height**2)

    num_non_evacuees = len(non_evacuees)
    t_stars = [p.t_star for p in evacuees]
    d_stars = [p.d_star for p in non_evacuees if p.d_star is not None]
    
    fitness_value = w1 * float(num_non_evacuees)
    max_t_star = w2 * float(max(t_stars))/(SimulationConfig.simulator.time_limit+1)
    agg_t_star = w3 * (float(sum(t_stars))/(num_total_pedestrians)/SimulationConfig.simulator.time_limit+1)
    if num_non_evacuees:
        min_d_star = w4 * float(min(d_stars))/D_diagonal
        avg_d_star = w5 * (float(sum(d_stars))/num_total_pedestrians)/D_diagonal
        fitness_value += min_d_star + avg_d_star

    fitness_value += max_t_star + agg_t_star

    return fitness_value


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
    domain.remove_emergency_accesses()
    fitness = sum_fitness / num_runs
    print(
        f"     fitness={fitness}, exits added={emergency_accesses}, current exits = {len(domain.get_exit_cells())}"
    )
    return fitness


def psi(
    domain: Domain, optimizer_strategy: OptimizerStrategy, emergency_accesses: list[int]
):
    new_emergency_accesses = [(i, SimulationConfig.omega) for i in emergency_accesses]

    match optimizer_strategy:
        case OptimizerStrategy.EA:
            resutl = psi_helper(EAConfig.numruns, domain, new_emergency_accesses)

        case OptimizerStrategy.IEA:
            resutl = psi_helper(IEAConfig.numruns, domain, new_emergency_accesses)

        case OptimizerStrategy.GREEDY:
            resutl = psi_helper(GreedyConfig.numruns, domain, new_emergency_accesses)

        case OptimizerStrategy.QL:
            resutl = psi_helper(EAConfig.numruns, domain, new_emergency_accesses)
        case _:
            print(f"optimizer_strategy = {optimizer_strategy}")
            print(f"emergency_accesses = {emergency_accesses}")

    return resutl
