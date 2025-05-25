import random
from typing import List, Tuple, Optional
from src.simulator.environment import CellularAutomata

def select_by_probability_normalized(
    transition_probability: List[Tuple[CellularAutomata, float]]
) -> Optional[CellularAutomata]:
    if not transition_probability:
        return None
    
    items = [item for item, _ in transition_probability]
    weights = [probability for _, probability in transition_probability]
    
    if any(weight < 0 for weight in weights):
        raise ValueError("Probability values must be non-negative")
    
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("At least one probability must be greater than zero")
    
    if abs(total_weight - 1.0) > 1e-9:
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = weights
    
    selected = random.choices(items, weights=normalized_weights, k=1)[0]
    
    return selected