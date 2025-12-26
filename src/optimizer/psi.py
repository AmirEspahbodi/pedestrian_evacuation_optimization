import math
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any

from src.config.simulation_config import SimulationConfig
from src.simulator.simulation_engine import main as evacuate_engine
from src.utils import add_emergency_exits, calculate_p_star


def calculate_fitness(
    num_non_evacuated_peds,
    t_stars,
    d_stars,
    width,
    height,
    num_total_peds,
    simulator_config,
) -> float:
    w1, w2, w3, w4, w5 = 3, 2, 1, 1, 2

    D_diagonal = math.sqrt(width**2 + height**2)

    fitness_value = w1 * float(num_non_evacuated_peds)
    max_t_star = w2 * float(max(t_stars)) / (simulator_config.simulator.timeLimit + 1)
    agg_t_star = w3 * (
        float(sum(t_stars)) / (num_total_peds) / simulator_config.simulator.timeLimit
        + 1
    )
    if num_non_evacuated_peds:
        min_d_star = w4 * float(min(d_stars)) / D_diagonal
        avg_d_star = w5 * (float(sum(d_stars)) / num_total_peds) / D_diagonal
        fitness_value += min_d_star + avg_d_star

    fitness_value += max_t_star + agg_t_star

    return fitness_value


def psi(
    gird: list[list[int]],
    pedestrian_confs: list[NDArray[Any]],
    emergency_accesses: list[int],
    simulator_config: SimulationConfig,
):
    new_emergency_accesses = [(i, simulator_config.omega) for i in emergency_accesses]

    gird_copy = deepcopy(gird)
    add_emergency_exits(gird_copy, new_emergency_accesses)
    sum_fitness = 0
    for pedestrian in pedestrian_confs:
        pedestrians_copy = np.empty_like(pedestrian)
        np.copyto(pedestrians_copy, pedestrian)
        num_total_pedestrians = np.sum(pedestrians_copy)
        t_star = evacuate_engine(simulator_config, gird_copy, pedestrians_copy)
        d_star = calculate_p_star(gird_copy, pedestrians_copy)
        num_peds = np.sum(pedestrians_copy)
        fitness_value = calculate_fitness(
            num_peds,
            t_star,
            d_star,
            len(pedestrians_copy),
            len(pedestrians_copy[0]),
            num_total_pedestrians,
            simulator_config,
        )
        sum_fitness += fitness_value
        del pedestrians_copy
    del gird_copy
    return sum_fitness / len(pedestrian_confs)
