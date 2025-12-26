from typing import List, Tuple


def add_emergency_exits(
    grid: List[List[int]], emergency_exits: List[Tuple[int, int]]
) -> None:
    H = len(grid)
    W = len(grid[0])

    perimeter_len = 2 * W + 2 * H

    def get_grid_coord(dist: int) -> Tuple[int, int]:
        d = dist % perimeter_len
        if d < W:
            return (0, d)
        d -= W
        if d < H:
            return (d, W - 1)
        d -= H

        if d < W:
            return (H - 1, (W - 1) - d)
        d -= W
        return ((H - 1) - d, 0)

    for start_pa, omega in emergency_exits:
        for step in range(omega):
            current_dist = start_pa + step

            r, c = get_grid_coord(current_dist)

            if 0 <= r < H and 0 <= c < W:
                grid[r][c] = 5
