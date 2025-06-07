import random
import math
from typing import List, Tuple, Dict, Any
from src.simulator.domain import Domain
from src.config import SimulationConfig, OptimizerStrategy, IEAConfig
from .common import FitnessEvaluator


Individual = List[int] # Changed to int for discrete environment
Population = List[Individual]

# --- Core Algorithm Components ---

def set_based_recombination(parent1: Individual, parent2: Individual) -> Individual:
    combined_pool = parent1 + parent2
    random.shuffle(combined_pool)
    
    offspring = random.sample(combined_pool, k=len(parent1))
    return offspring

def gaussian_mutation_discrete(
    individual: Individual,
    gamma: float,
    perimeter_length: int
) -> Individual:
    mutated_individual = individual[:]
    
    # Select a single exit to mutate
    idx_to_mutate = random.randint(0, len(mutated_individual) - 1)
    gene = mutated_individual[idx_to_mutate]
    
    # Apply Gaussian perturbation: e' = e * (1 + gamma * N(0,1))
    perturbation = gamma * random.normalvariate(0, 1)
    mutated_gene_float = gene * (1 + perturbation)
    
    # --- DISCRETE ADAPTATION ---
    # Convert to a discrete value by rounding to the nearest integer
    mutated_gene_int = int(round(mutated_gene_float))
    
    # Apply wrap-around logic for the discrete perimeter
    mutated_gene_int %= perimeter_length
    
    mutated_individual[idx_to_mutate] = mutated_gene_int
    return mutated_individual

def tournament_selection(population: Population, fitnesses: List[float]) -> Individual:
    idx1 = random.randint(0, len(population) - 1)
    idx2 = random.randint(0, len(population) - 1)
    
    if fitnesses[idx1] < fitnesses[idx2]:
        return population[idx1]
    else:
        return population[idx2]

# --- Main iEA Optimizer Function ---

def iEA_optimizer(
    domain: Domain,
) -> Dict[str, Any]:
    # --- Algorithm Parameters from Paper Section 5.1 ---
    perimeter_length = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits
    max_evals: int = IEAConfig.islands[0].maxevals
    
    num_islands: int = 4  # 
    island_pop_size: int = 25  # 
    migration_frequency: int = 10  # in generations 
    p_recombination: float = 0.9  # 
    p_mutation: float = 1.0 
    gamma_mutation: float = 0.05  # 
    
    # --- Initialization ---
    islands = []
    fitness_evaluator: FitnessEvaluator = FitnessEvaluator(domain=domain, optimizer_strategy=OptimizerStrategy.IEA)
    generation_count = 0
    
    history = {
        "island1": {},
        "island2": {},
        "island3": {},
        "island4": {},
    }
    
    print("Initializing islands for discrete environment...")
    for i in range(num_islands):
        # --- DISCRETE ADAPTATION ---
        # Generate integer locations for the discrete perimeter
        population = [
            [random.randint(0, perimeter_length - 1) for _ in range(k_exits)]
            for _ in range(island_pop_size)
        ]
        fitnesses = [fitness_evaluator.evaluate(ind) for ind in population]
        history[f"island{i+1}"][f"{generation_count}"] = fitnesses
        
        islands.append({
            "population": population,
            "fitnesses": fitnesses
        })
    print(f"Initialization complete. Performed {fitness_evaluator.get_evaluation_count()} evaluations.")

    # --- Main Evolution Loop ---
    while fitness_evaluator.get_evaluation_count() < max_evals:
        # Migration Event
        if generation_count > 0 and generation_count % migration_frequency == 0:
            island_bests = []
            island_worsts_indices = []
            for island in islands:
                best_idx = min(range(len(island["fitnesses"])), key=island["fitnesses"].__getitem__)
                worst_idx = max(range(len(island["fitnesses"])), key=island["fitnesses"].__getitem__)
                island_bests.append(island["population"][best_idx][:])
                island_worsts_indices.append(worst_idx)

            for i in range(num_islands):
                left_neighbor_idx = (i - 1 + num_islands) % num_islands
                right_neighbor_idx = (i + 1) % num_islands
                migrant = random.choice([island_bests[left_neighbor_idx], island_bests[right_neighbor_idx]])
                worst_idx = island_worsts_indices[i]
                islands[i]["population"][worst_idx] = migrant
                islands[i]["fitnesses"][worst_idx] = float('inf')
        
        generation_count += 1

        # Evolve each island
        for i, island in enumerate(islands):
            best_idx = min(range(len(island["fitnesses"])), key=island["fitnesses"].__getitem__)
            elite_ind = island["population"][best_idx][:]
            elite_fitness = island["fitnesses"][best_idx]

            new_population = [elite_ind]
            new_fitnesses = [elite_fitness]
            
            for _ in range(island_pop_size - 1):
                parent1 = tournament_selection(island["population"], island["fitnesses"])
                parent2 = tournament_selection(island["population"], island["fitnesses"])
                
                if random.random() < p_recombination:
                    offspring = set_based_recombination(parent1, parent2)
                else:
                    offspring = parent1[:]
                
                if random.random() < p_mutation:
                    # --- DISCRETE ADAPTATION ---
                    offspring = gaussian_mutation_discrete(offspring, gamma_mutation, perimeter_length)
                
                new_population.append(offspring)
                temp = fitness_evaluator.evaluate(offspring)
                if f"{generation_count}" in history[f"island{i+1}"].keys():
                    history[f"island{i+1}"][f"{generation_count}"].append(temp)
                else:
                    history[f"island{i+1}"][f"{generation_count}"] = [temp]
                
                new_fitnesses.append(temp)
                if fitness_evaluator.get_evaluation_count() >= max_evals: break
            
            island["population"] = new_population
            island["fitnesses"] = new_fitnesses
            if fitness_evaluator.get_evaluation_count() >= max_evals: break
        
        if fitness_evaluator.get_evaluation_count() >= max_evals: break

    print(f"\n--- Optimization Finished (Evaluations: {fitness_evaluator.get_evaluation_count()}) ---")
    
    # Find the globally best solution
    global_best_fitness = float('inf')
    global_best_individual = None
    for island in islands:
        best_idx = min(range(len(island["fitnesses"])), key=island["fitnesses"].__getitem__)
        if island["fitnesses"][best_idx] < global_best_fitness:
            global_best_fitness = island["fitnesses"][best_idx]
            global_best_individual = island["population"][best_idx]

    return global_best_individual, global_best_fitness, history

