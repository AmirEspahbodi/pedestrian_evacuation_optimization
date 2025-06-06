import random
from typing import List, Tuple, Optional
from src.simulator.cellular_automata import CellularAutomata
from src.config import SimulationConfig
import math


def select_by_probability_normalized(
    transition_probability: List[Tuple[CellularAutomata, float]],
) -> CellularAutomata:
    transition_probability.sort(key=lambda x: x[1], reverse=True)

    transition_probability_topk = transition_probability[: SimulationConfig.K]

    if not transition_probability_topk:
        return None

    items = [item for item, _ in transition_probability_topk]
    weights = [probability for _, probability in transition_probability_topk]

    print(weights)
    if any(weight < 0 for weight in weights):
        raise ValueError("Probability values must be non-negative")

    total_weight = sum([math.exp(i) for i in weights])
    if total_weight == 0:
        raise ValueError("At least one probability must be greater than zero")

    normalized_weights = [math.exp(w) / total_weight for w in weights]

    selected = random.choices(items, weights=normalized_weights, k=1)[0]

    return selected
