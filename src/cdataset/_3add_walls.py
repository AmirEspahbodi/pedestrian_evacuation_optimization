import os

import numpy as np


def enforce_boundary_walls(input_path: str, output_path: str):
    """
    Loads a dense integer grid, enforces walls at y=0 and x=0,
    and saves strictly preserving the continuous digit string format.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source grid not found: {input_path}")

    # 1. Load: 'delimiter=1' parses fixed-width single char integers efficiently
    # dtype=np.int8 is sufficient for values 0-2 and saves memory
    grid = np.genfromtxt(input_path, delimiter=1, dtype=np.int8)

    # 2. Modify: Vectorized assignment for O(1) syntax complexity
    # Set entire Top Row (y=0) to Wall (1)
    grid[0, :] = 1

    # Set entire Left Column (x=0) to Wall (1)
    grid[:, 0] = 1

    grid[:, -1] = 1
    grid[-1, :] = 1
    # 3. Save: fmt='%d' ensures integers, delimiter='' removes spacing between cols
    np.savetxt(output_path, grid, fmt="%d", delimiter="")

    print(f"Grid processed. Shape: {grid.shape}. Saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "obstacle_grid_400.txt"
    OUTPUT_FILE = "obstacle_grid_walled_400.txt"

    enforce_boundary_walls(INPUT_FILE, OUTPUT_FILE)
