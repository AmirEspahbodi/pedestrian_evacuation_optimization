from typing import List
import math

from src.simulator.domain import Domain, Pedestrian
from src.config import SimulationConfig


def calculate_fitness(
    domain: Domain,
) -> float:
    print("- - - - -" * 10)
    pedestrians = domain.get_pedestrians()
    num_total_pedestrians = len(pedestrians)
    print(num_total_pedestrians)
    evacuees: List[Pedestrian] = []
    non_evacuees: List[Pedestrian] = []

    for p in pedestrians:
        if p.is_exited:
            evacuees.append(p)
        else:
            non_evacuees.append(p)
    print(len(evacuees))
    print(len(non_evacuees))
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
        term_3b = (1 / (num_total_pedestrians * D_diagonal**2)) * sum_d_star
        fitness_value += term_2b + term_3b

    return fitness_value
