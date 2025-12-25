import os
import random
import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

# --- Configuration ---
INPUT_FILE = "obstacle_grid_walled_400.txt"
OUTPUT_FILE = "obstacle_grid_walled_with_access_400.txt"

# Cell Values
VAL_EMPTY = 0
VAL_WALL = 1
VAL_FURNITURE = 2
VAL_ACCESS = 4

# Generator Settings
ACCESS_COUNT = 2
MIN_ACCESS_LEN = 3
MAX_ACCESS_LEN = 4

# Visualization Colors (R, G, B)
COLOR_MAP = {
    VAL_EMPTY: [245, 245, 245],  # Off-White
    VAL_WALL: [50, 50, 50],  # Dark Grey/Black
    VAL_FURNITURE: [70, 130, 180],  # Steel Blue
    VAL_ACCESS: [0, 255, 0],  # Bright Green (The new access points)
}

# --- Logic Layer ---


def load_grid(filepath: str) -> np.ndarray:
    if not os.path.exists(filepath):
        # Create a dummy file for testing if one doesn't exist
        print(f"[!] {filepath} not found. Generating dummy 400x400 grid...")
        dummy = np.zeros((400, 400), dtype=int)
        # Add border walls
        dummy[0, :] = VAL_WALL
        dummy[-1, :] = VAL_WALL
        dummy[:, 0] = VAL_WALL
        dummy[:, -1] = VAL_WALL
        return dummy

    with open(filepath, "r") as f:
        lines = [list(map(int, list(line.strip()))) for line in f if line.strip()]
    return np.array(lines, dtype=np.uint8)


def save_grid(grid: np.ndarray, filepath: str):
    with open(filepath, "w") as f:
        for row in grid:
            f.write("".join(map(str, row)) + "\n")
    print(f"[-] Grid saved to {filepath}")


def add_root_access(grid: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape

    # (Wall Name, Fixed Axis Index (0=Y, 1=X), Fixed Coordinate, Range Max)
    walls = [
        ("TOP", 0, 0, cols),
        ("BOTTOM", 0, rows - 1, cols),
        ("LEFT", 1, 0, rows),
        ("RIGHT", 1, cols - 1, rows),
    ]

    count = 0
    attempts = 0

    while count < ACCESS_COUNT and attempts < 100:
        attempts += 1
        name, fix_axis, fix_coord, var_max = random.choice(walls)
        length = random.randint(MIN_ACCESS_LEN, MAX_ACCESS_LEN)

        if var_max - length <= 0:
            continue

        start = random.randint(0, var_max - length)
        end = start + length

        # Slice based on axis
        if fix_axis == 0:  # Horizontal
            segment = grid[fix_coord, start:end]
        else:  # Vertical
            segment = grid[start:end, fix_coord]

        # Overlap check
        if np.any(segment == VAL_ACCESS):
            continue

        # Apply
        if fix_axis == 0:
            grid[fix_coord, start:end] = VAL_ACCESS
        else:
            grid[start:end, fix_coord] = VAL_ACCESS

        print(f"[+] Added Access: {name}, Length: {length}, Index: {start}")
        count += 1

    return grid


# --- Visualization Layer (PyQt6) ---


class GridViewer(QMainWindow):
    def __init__(self, grid_data):
        super().__init__()
        self.setWindowTitle("Floor Plan Viewer")
        self.grid_data = grid_data

        # Determine scale factor based on screen size vs grid size
        # We want the window to be reasonably sized (e.g., 800px)
        h, w = grid_data.shape
        self.scale = max(1, 800 // max(h, w))

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # 1. Convert Index Grid to RGB Grid using NumPy broadcasting
        # Create an empty RGB array
        h, w = self.grid_data.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Apply colors map
        for key, color in COLOR_MAP.items():
            # Boolean indexing: where grid is 'key', set color
            rgb_image[self.grid_data == key] = color

        # 2. Create QImage from the RGB array
        # QImage needs data to be contiguous in memory
        if not rgb_image.flags["C_CONTIGUOUS"]:
            rgb_image = np.ascontiguousarray(rgb_image)

        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width

        q_img = QImage(
            rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )

        # 3. Scale and Display
        pixmap = QPixmap.fromImage(q_img)

        # Scale up for visibility using FastTransformation or SmoothTransformation
        pixmap = pixmap.scaled(
            width * self.scale,
            height * self.scale,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(label)

        # Add info label
        info = QLabel(f"Grid Size: {width}x{height} | Green = Access Points (Value 4)")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)


# --- Main Execution ---

if __name__ == "__main__":
    # 1. Process Data
    try:
        raw_grid = load_grid(INPUT_FILE)
        processed_grid = add_root_access(raw_grid)
        save_grid(processed_grid, OUTPUT_FILE)
    except Exception as e:
        print(f"Data Processing Error: {e}")
        sys.exit(1)

    # 2. Launch GUI
    print("[-] Launching Visualization...")
    app = QApplication(sys.argv)
    viewer = GridViewer(processed_grid)
    viewer.show()
    sys.exit(app.exec())
