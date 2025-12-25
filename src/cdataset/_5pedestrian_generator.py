import sys
from tkinter.constants import N

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from scipy.spatial.distance import cdist

# Configuration
INPUT_FILE = "obstacle_grid_walled_with_access_400.txt"
OUTPUT_FILE = "finall_grid_400_NUM.txt"
TARGET_PEDESTRIANS = 200
CANDIDATES_PER_STEP = 15
NUM_EVALUATE = 10

# Color Mapping for Visualization (R, G, B)
COLORS = {
    0: (240, 240, 240),  # Empty (Light Gray/White)
    1: (0, 0, 0),  # Wall (Black)
    2: (0, 0, 255),  # Obstacle (Blue)
    3: (255, 0, 0),  # Pedestrian (Red)
}


def load_grid(filename):
    """
    Parses a dense text grid without delimiters into a NumPy array.
    """
    try:
        with open(filename, "r") as f:
            lines = f.read().splitlines()

        # Convert string lines to list of lists of integers
        grid_data = [[int(char) for char in line] for line in lines if line]
        return np.array(grid_data, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        # Create a dummy 400x400 grid for testing if file missing
        print("Creating dummy 400x400 grid for demonstration...")
        dummy = np.zeros((400, 400), dtype=np.uint8)
        dummy[0, :] = 1
        dummy[-1, :] = 1
        dummy[:, 0] = 1
        dummy[:, -1] = 1  # Walls
        return dummy


def save_grid(grid, filename):
    """
    Saves the grid back to text format (dense strings, no delimiters).
    """
    with open(filename, "w") as f:
        for row in grid:
            # Join integers as a string
            line = "".join(row.astype(str))
            f.write(line + "\n")
    print(f"Grid saved to {filename}")


def place_pedestrians(grid, count=100, k_candidates=15):
    """
    Places pedestrians using Mitchell's Best-Candidate Algorithm.
    Maximizes the distance to the nearest existing neighbor.
    """
    rows, cols = grid.shape

    # 1. Identify valid walkable coordinates (where grid == 0)
    # returns (row_indices, col_indices), we stack them to get (N, 2) coords
    walkable_y, walkable_x = np.where(grid == 0)
    walkable_coords = np.column_stack((walkable_y, walkable_x))

    if len(walkable_coords) < count:
        raise ValueError("Not enough empty space to place pedestrians.")

    existing_pedestrians = []

    print(f"Starting placement of {count} pedestrians...")

    for i in range(count):
        # 3. Candidate Generation
        # Randomly sample indices from the available walkable coordinates
        # We use random choice of indices to avoid copying large arrays
        candidate_indices = np.random.choice(
            len(walkable_coords), size=k_candidates, replace=False
        )
        candidates = walkable_coords[candidate_indices]

        # 4. Evaluation
        if i == 0:
            # First placement: just pick the first candidate
            best_candidate = candidates[0]
            # Remove this spot from walkable to avoid collisions (optional but clean)
            walkable_coords = np.delete(walkable_coords, candidate_indices[0], axis=0)
        else:
            # Calculate distance from *each* candidate to *all* existing pedestrians
            # cdist returns matrix (k_candidates x existing_count)
            dists = cdist(candidates, np.array(existing_pedestrians))

            # Find the distance to the *nearest* neighbor for every candidate
            # axis=1 means min across the columns (existing peds)
            min_dists = dists.min(axis=1)

            # Selection: Choose candidate with the *largest* nearest-neighbor distance
            best_index_local = np.argmax(min_dists)
            best_candidate = candidates[best_index_local]

            # Remove used coordinate from walkable pool to prevent overlap
            # (Map back to original array index)
            actual_index_in_walkable = candidate_indices[best_index_local]
            walkable_coords = np.delete(
                walkable_coords, actual_index_in_walkable, axis=0
            )

        # 5. Update
        existing_pedestrians.append(best_candidate)

        # Mark grid
        r, c = best_candidate
        grid[r, c] = 3

    print("Placement complete.")
    return grid


class GridVisualizer(QMainWindow):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.setWindowTitle("Pedestrian Distribution (Blue Noise)")
        self.setGeometry(100, 100, 800, 800)

        # Prepare the image once to save resources
        self.image = self.create_image_from_grid()

    def create_image_from_grid(self):
        """
        Converts the numpy grid into a QImage efficiently.
        """
        height, width = self.grid.shape

        # Create an RGB image buffer (Height, Width, 3)
        # Initialize with zeros
        rgb_data = np.zeros((height, width, 3), dtype=np.uint8)

        # Vectorized color mapping
        for val, color in COLORS.items():
            mask = self.grid == val
            rgb_data[mask] = color

        # Ensure data is contiguous for QImage
        rgb_data = np.ascontiguousarray(rgb_data)

        # Create QImage from data
        # Format_RGB888 expects 3 bytes per pixel
        image = QImage(
            rgb_data.data, width, height, 3 * width, QImage.Format.Format_RGB888
        )

        # We must copy the image, otherwise it points to deleted numpy memory later
        return image.copy()

    def paintEvent(self, event):
        painter = QPainter(self)

        # Scale image to fit window while keeping aspect ratio
        target_rect = self.rect()
        scaled_image = self.image.scaled(
            target_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        # Draw centered
        x = (target_rect.width() - scaled_image.width()) // 2
        y = (target_rect.height() - scaled_image.height()) // 2
        painter.drawImage(x, y, scaled_image)


def main():
    # 1. Load Data
    last_populated_grid = None
    for i in range(0, NUM_EVALUATE):
        grid = load_grid(INPUT_FILE)

        # 2. Algorithm Execution
        populated_grid = place_pedestrians(
            grid, count=TARGET_PEDESTRIANS, k_candidates=CANDIDATES_PER_STEP
        )

        # 3. Save Output
        save_grid(populated_grid, OUTPUT_FILE.replace("NUM", str(i + 1)))
        last_populated_grid = populated_grid

    # 4. Visualization
    app = QApplication(sys.argv)
    window = GridVisualizer(last_populated_grid)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
