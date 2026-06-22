from collections import deque

import numpy as np
from numpy import int8
from numpy._typing import NDArray


def get_pedestrian_distances(
    grid_map: np.ndarray, pedestrians: NDArray[int8]
) -> list[float]:
    # grid_map strictly represents the environment map (2 = Exits, -1 = Walls)
    padded_grid = np.pad(grid_map, pad_width=1, mode="constant", constant_values=-1)
    exit_rows, exit_cols = np.where(grid_map == 2)
    exit_rows += 1
    exit_cols += 1

    # ROBUST FIX: Extract coordinates directly from the uncontaminated pedestrians array
    ped_rows, ped_cols = np.where(pedestrians == 1)

    if len(exit_rows) == 0:
        return [float("inf")] * len(ped_rows)

    rows, cols = padded_grid.shape
    dist_map = np.full((rows, cols), np.inf)
    queue = deque()

    for r, c in zip(exit_rows, exit_cols):
        dist_map[r, c] = 0.0
        queue.append((r, c))

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        current_dist = dist_map[r, c]
        next_dist = current_dist + 1.0
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if padded_grid[nr, nc] != -1 and dist_map[nr, nc] == np.inf:
                dist_map[nr, nc] = next_dist
                queue.append((nr, nc))

    # Safely fetch all distances mapping exactly to the remaining pedestrians
    distances = dist_map[ped_rows + 1, ped_cols + 1].tolist()
    return distances


def calculate_d_star(gird: list[list[int]], pedestrians: NDArray[int8]):
    gird_np = np.array(gird)

    # Create an independent environment map
    grid_map = np.zeros_like(gird_np, dtype=np.int8)
    grid_map[(gird_np == 1) | (gird_np == 2)] = -1  # Obstacles/Walls
    grid_map[(gird_np == 4) | (gird_np == 5)] = 2  # Exits

    # Pass both independent arrays
    distances = get_pedestrian_distances(grid_map, pedestrians)
    return distances
