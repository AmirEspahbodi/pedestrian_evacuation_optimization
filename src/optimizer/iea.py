import random
from typing import List, Tuple
from src.simulator.domain import Domain
from src.config import SimulationConfig, OptimizerStrategy, IEAConfig
from .common import FitnessEvaluator, Individual
from .psi import psi as psi_function
from .ea import (
    _binary_tournament_selection,
    _set_based_recombination,
    _mutate_individual_ea,
)


def island_evolutionary_algorithm(
    domain: Domain
) -> Tuple[List[float], float, int]:
    migration_frequency_generations = IEAConfig.islands[
        0
    ].migration_frequency_generations
    num_islands = IEAConfig.islands[0].numislands
    perimeter_length = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits
    psi_evaluator = FitnessEvaluator(domain, OptimizerStrategy.IEA)
    island_population_size: int = IEAConfig.islands[0].popsize
    recombination_prob: float = IEAConfig.islands[0].recombination_prob
    mutation_gamma: float = IEAConfig.islands[0].mutation_gamma
    max_evals: int = IEAConfig.islands[0].maxevals


    hostory: dict[str, dict[int, list]] = {
        "island1": {},
        "island2": {},
        "island3": {},
        "island4": {},
    }
    generation = 0

    islands: List[List[Individual]] = []
    for i in range(num_islands):
        island_pop: List[Individual] = []
        for _ in range(island_population_size):
            genes = [random.randint(0, perimeter_length) for _ in range(k_exits)]
            individual = Individual(genes)
            island_pop.append(individual)
        islands.append(island_pop)

    for index, island_pop in enumerate(islands):
        hostory[f'island{index+1}'][generation] = []
        for ind in island_pop:
            if psi_evaluator.get_evaluation_count() >= max_evals:
                break
            ind.fitness = psi_evaluator.evaluate(ind.genes)
            hostory[f'island{index+1}'][generation].append(ind.fitness)
        if psi_evaluator.get_evaluation_count() >= max_evals:
            break
        island_pop.sort()

    best_overall_individual = Individual([]) 
    for island_pop in islands:
        if island_pop and (
            not best_overall_individual.genes
            or island_pop[0].fitness < best_overall_individual.fitness
        ):
            best_overall_individual = island_pop[0]


    while psi_evaluator.get_evaluation_count() < max_evals:
        generation += 1
        print(f"generation = {generation}")

        for i in range(num_islands):
            if psi_evaluator.get_evaluation_count() >= max_evals:
                break
            hostory[f'island{i+1}'][generation] = []

            current_island_pop = islands[i]

            next_island_pop_segment: List[Individual] = []
            for _ in range(
                island_population_size // 2
            ):  
                if psi_evaluator.get_evaluation_count() >= max_evals:
                    break

                parent1 = _binary_tournament_selection(current_island_pop)
                parent2 = _binary_tournament_selection(current_island_pop)

                offspring1_genes: List[float]
                offspring2_genes: List[float]

                if random.random() < recombination_prob:
                    offspring1_genes = _set_based_recombination(
                        parent1.genes, parent2.genes, k_exits
                    )
                    offspring2_genes = _set_based_recombination(
                        parent2.genes, parent1.genes, k_exits
                    )
                else:
                    offspring1_genes = parent1.genes[:]
                    offspring2_genes = parent2.genes[:]

                _mutate_individual_ea(
                    offspring1_genes, mutation_gamma, perimeter_length, k_exits
                )
                _mutate_individual_ea(
                    offspring2_genes, mutation_gamma, perimeter_length, k_exits
                )

                offspring1 = Individual(offspring1_genes)
                if psi_evaluator.get_evaluation_count() < max_evals:
                    offspring1.fitness = psi_evaluator.evaluate(offspring1.genes)
                    hostory[f'island{i+1}'][generation].append(ind.fitness)
                else:
                    offspring1.fitness = float("inf")
                next_island_pop_segment.append(offspring1)

                if len(next_island_pop_segment) <= island_population_size:
                    offspring2 = Individual(offspring2_genes)
                    if psi_evaluator.get_evaluation_count() < max_evals:
                        offspring2.fitness = psi_evaluator.evaluate(offspring2.genes)
                        hostory[f'island{i+1}'][generation].append(ind.fitness)
                    else:
                        offspring2.fitness = float("inf")
                    next_island_pop_segment.append(offspring2)

            if (
                not next_island_pop_segment
                and psi_evaluator.get_evaluation_count() >= max_evals
            ):
                break
            
            current_island_pop.extend(next_island_pop_segment)
            current_island_pop.sort()
            islands[i] = current_island_pop[:island_population_size]

            if islands[i] and islands[i][0].fitness < best_overall_individual.fitness:
                best_overall_individual = islands[i][0]

        if psi_evaluator.get_evaluation_count() >= max_evals:
            break

        if generation % migration_frequency_generations == 0 and num_islands > 1:
            migrants_to_send: List[Individual] = [
                island_pop[0] for island_pop in islands if island_pop
            ] 
            if not migrants_to_send:
                continue 
            next_islands_state = [list(island_pop) for island_pop in islands]

            for i in range(num_islands):
                if not islands[i]:
                    continue

                prev_island_idx = (i - 1 + num_islands) % num_islands
                migrant_from_prev = (
                    migrants_to_send[prev_island_idx]
                    if prev_island_idx < len(migrants_to_send)
                    else None
                )

                next_island_idx = (i + 1) % num_islands
                migrant_from_next = (
                    migrants_to_send[next_island_idx]
                    if next_island_idx < len(migrants_to_send)
                    else None
                )

                potential_immigrants = []
                if migrant_from_prev:
                    potential_immigrants.append(migrant_from_prev)
                if (
                    migrant_from_next and migrant_from_next is not migrant_from_prev
                ): 
                    potential_immigrants.append(migrant_from_next)

                potential_immigrants.sort()

                current_receiving_pop = next_islands_state[i]
                current_receiving_pop.sort()  

                replaced_count = 0
                for immigrant in potential_immigrants:
                    if (
                        replaced_count < len(current_receiving_pop)
                        and immigrant.fitness
                        < current_receiving_pop[
                            len(current_receiving_pop) - 1 - replaced_count
                        ].fitness
                    ):
                        current_receiving_pop[
                            len(current_receiving_pop) - 1 - replaced_count
                        ] = Individual(immigrant.genes[:])  
                        current_receiving_pop[
                            len(current_receiving_pop) - 1 - replaced_count
                        ].fitness = immigrant.fitness
                        replaced_count += 1
                    if (
                        replaced_count >= 2
                    ): 
                        break

                current_receiving_pop.sort() 
                next_islands_state[i] = current_receiving_pop

            islands = next_islands_state

            for island_pop in islands:
                if (
                    island_pop
                    and island_pop[0].fitness < best_overall_individual.fitness
                ):
                    best_overall_individual = island_pop[0]

    return (
        best_overall_individual.genes,
        best_overall_individual.fitness,
        psi_evaluator.get_evaluation_count(),
        hostory
    )
