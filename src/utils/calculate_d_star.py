from collections import deque

import numpy as np
from numpy import int8
from numpy._typing import NDArray


def _helper_array(gird: list[list[int]], pedestrians: NDArray[int8]) -> NDArray[int8]:
    gird_np = np.array(gird)
    helper = np.zeros_like(pedestrians)
    np.copyto(helper, pedestrians)
    helper[(gird_np == 1) | (gird_np == 2)] = -1
    helper[(gird_np == 4) | (gird_np == 5)] = 2
    return helper


def get_pedestrian_distances(grid: np.ndarray) -> list[float]:
    padded_grid = np.pad(grid, pad_width=1, mode="constant", constant_values=-1)
    exit_rows, exit_cols = np.where(grid == 2)
    exit_rows += 1
    exit_cols += 1
    if len(exit_rows) == 0:
        ped_count = np.count_nonzero(grid == 1)
        return [float("inf")] * ped_count
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
    ped_rows, ped_cols = np.where(grid == 1)
    distances = dist_map[ped_rows + 1, ped_cols + 1].tolist()
    return distances


def calculate_p_star(gird: list[list[int]], pedestrians: NDArray[int8]):
    helper = _helper_array(gird, pedestrians)
    distances = get_pedestrian_distances(helper)
    return distances
