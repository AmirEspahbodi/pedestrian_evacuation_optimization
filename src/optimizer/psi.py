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
    pedestrians = domain.get_pedestrians()
    num_total_pedestrians = len(pedestrians)
    evacuees: List[Pedestrian] = []
    non_evacuees: List[Pedestrian] = []

    for p in pedestrians:
        if p.is_exited:
            evacuees.append(p)
        else:
            non_evacuees.append(p)
    num_non_evacuees = len(non_evacuees)
    fitness_value = float(num_non_evacuees)

    D_diagonal = math.sqrt(domain.width**2 + domain.height**2)

    if num_non_evacuees == 0:
        t_stars = [p.t_star for p in evacuees]
        max_t_star = float(max(t_stars))
        sum_t_star = float(sum(t_stars))
        term_2a = (1 / SimulationConfig.num_simulations) * max_t_star
        term_3a = (
            1 / (num_total_pedestrians * SimulationConfig.num_simulations**2)
        ) * sum_t_star
        fitness_value += term_2a + term_3a
    else:
        d_stars = [p.d_star for p in non_evacuees if p.d_star is not None]
        min_d_star = float(min(d_stars))
        sum_d_star = float(sum(d_stars))
        term_2b = (1 / D_diagonal) * min_d_star
        term_3b = (1 / ((num_total_pedestrians) * D_diagonal**2)) * sum_d_star
        fitness_value += term_2b + term_3b

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
