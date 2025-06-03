import random
import math
from typing import List, Tuple
from src.simulator.domain import Domain
from src.config import SimulationConfig, OptimizerStrategy
from .common import FitnessEvaluator



def greedy_algorithm(
    domain: Domain,
) -> Tuple[List[float], float]:
    psi_evaluator: FitnessEvaluator = FitnessEvaluator(domain, OptimizerStrategy.GREEDY)

    omega_exit_width: float = SimulationConfig.omega
    perimeter_length = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits

    E_solutions: List[float] = []
    current_fitness = float("inf")

    eta =  math.ceil(perimeter_length / omega_exit_width)

    print(f"k_exits = {k_exits}")
    print(f"eta = {eta}")
    for i in range(k_exits):
        print(f"looking for {i + 1}th emergency exit place")
        best_psi_for_this_exit = float("inf")
        chosen_exit_for_this_iteration = -1
        p_start_scan = random.randint(0, perimeter_length)

        candidate_location = p_start_scan

        for j in range(eta):
            temp_accesses = E_solutions + [candidate_location]
            current_eval_psi = psi_evaluator.evaluate(temp_accesses)
            print(
                f"     j={j}, temp_accesses={temp_accesses}, current_eval_psi={current_eval_psi}\n"
            )

            if current_eval_psi < best_psi_for_this_exit:
                best_psi_for_this_exit = current_eval_psi
                chosen_exit_for_this_iteration = candidate_location

            candidate_location += omega_exit_width
            if candidate_location >= perimeter_length:
                candidate_location -= perimeter_length

        E_solutions.append(chosen_exit_for_this_iteration)
        current_fitness = best_psi_for_this_exit

    return E_solutions, current_fitness
