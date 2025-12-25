import random
import sys
from typing import List, Tuple

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget

# Grid Constants
EMPTY = 0
WALL = 1
OBSTACLE = 2  # New value for furniture


class FurniturePlacer:
    """
    Loads a grid and intelligently populates it with furniture
    respecting strict distance constraints.
    """

    def __init__(self, filename: str):
        self.grid = self._load_grid(filename)
        self.height, self.width = self.grid.shape

        # 1. Create a "Valid Mask"
        # True = Placeable area, False = Restricted (too close to walls/borders)
        self.valid_mask = np.ones_like(self.grid, dtype=bool)
        self._compute_valid_zones()

    def save_to_file(self, filename: str):
        """
        Saves the populated grid to a text file.
        """
        try:
            # fmt='%d' saves as integers. delimiter='' makes it a dense block of text.
            np.savetxt(filename, self.grid, fmt="%d", delimiter="")
            print(f"[Success] Furnished grid saved to local file: {filename}")
        except Exception as e:
            print(f"[Error] Failed to save grid: {e}")

    def _load_grid(self, filename: str) -> np.ndarray:
        """
        Manually reads the dense 0/1 text file (e.g., '00100') into a NumPy array.
        """
        try:
            with open(filename, "r") as f:
                # Read each line, strip whitespace, convert each char to int
                # This handles the lack of spaces between numbers
                lines = []
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        lines.append([int(c) for c in stripped])

            return np.array(lines)
        except Exception as e:
            print(f"[Error] Failed to load file: {e}")
            # Return a default 100x80 grid if load fails so the GUI doesn't crash
            return np.zeros((100, 100), dtype=int)

    def _compute_valid_zones(self):
        """
        Pre-calculates where furniture is allowed to exist based on static constraints.
        Constraint 3: Gap > 3 cells from room walls (We use 4 to be safe '>', or 3 strict).
        Constraint 4: Gap > 5 cells from environment boundary.
        """
        WALL_MARGIN = 2
        BORDER_MARGIN = 4

        # A. Apply Border Constraints
        self.valid_mask[0:BORDER_MARGIN, :] = False
        self.valid_mask[self.height - BORDER_MARGIN :, :] = False
        self.valid_mask[:, 0:BORDER_MARGIN] = False
        self.valid_mask[:, self.width - BORDER_MARGIN :] = False

        # B. Apply Wall Constraints
        # We iterate through all WALL cells and mark their neighbors as invalid.
        # This is a manual dilation for simplicity and dependency reduction (vs scipy).
        walls_y, walls_x = np.where(self.grid == WALL)

        for y, x in zip(walls_y, walls_x):
            # Define the exclusion box around this wall pixel
            y_min = max(0, y - WALL_MARGIN)
            y_max = min(self.height, y + WALL_MARGIN + 1)
            x_min = max(0, x - WALL_MARGIN)
            x_max = min(self.width, x + WALL_MARGIN + 1)

            self.valid_mask[y_min:y_max, x_min:x_max] = False

    def _can_place(self, x, y, w, h) -> bool:
        """
        Checks if a specific rectangle can be placed.
        1. Must be inside the pre-calculated 'valid_mask'.
        2. Must not overlap existing furniture (OBSTACLE).
        3. Must have 1-cell gap from existing furniture.
        """
        # 1. Boundary Check
        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            return False

        # 2. Check Valid Mask (Wall/Border proximity)
        # All cells in the target rect must be True in valid_mask
        sub_mask = self.valid_mask[y : y + h, x : x + w]
        if not np.all(sub_mask):
            return False

        # 3. Check Overlap + 1-cell Gap with existing obstacles
        # We look at the region extended by 1 cell
        OBSTACLE_MARGIN = random.randint(1, 2)
        check_y_min = max(0, y - OBSTACLE_MARGIN)
        check_y_max = min(self.height, y + h + OBSTACLE_MARGIN)
        check_x_min = max(0, x - OBSTACLE_MARGIN)
        check_x_max = min(self.width, x + w + OBSTACLE_MARGIN)

        sub_grid = self.grid[check_y_min:check_y_max, check_x_min:check_x_max]
        if np.any(sub_grid == OBSTACLE):
            return False

        return True

    def _place_rect(self, x, y, w, h):
        self.grid[y : y + h, x : x + w] = OBSTACLE

    def populate(self):
        """
        Main logic loop to fill valid zones with furniture sets.
        """
        # We attempt to place items N times.
        # We prioritize "Sets" to make it intelligent.

        attempts = 100  # Total attempts to find spots for groups

        for _ in range(attempts):
            # Random starting point within valid bounds
            # Speed optimization: Pick from valid indices directly
            valid_y, valid_x = np.where(self.valid_mask)
            if len(valid_x) == 0:
                break

            idx = random.randint(0, len(valid_x) - 1)
            rx, ry = valid_x[idx], valid_y[idx]

            # Strategy Selection: 60% Living Room Set, 40% Library/Study
            strategy = random.choice(["living", "library"])

            if strategy == "living":
                self._try_place_living_set(rx, ry)
            else:
                self._try_place_bookshelf(rx, ry)

        return self.grid

    def _try_place_living_set(self, x, y):
        """
        Tries to place a Sofa, then a Table nearby, then a Chair nearby.
        """
        # 1. Place Sofa (2-3 W, 3-5 L)
        # Randomize orientation (vertical vs horizontal dimensions)
        dim1 = random.randint(1, 3)
        dim2 = random.randint(3, 5)
        sw, sh = (dim1, dim2) if random.choice([True, False]) else (dim2, dim1)

        if self._can_place(x, y, sw, sh):
            self._place_rect(x, y, sw, sh)

            # 2. Try Place Table nearby (1x2 or 2x3)
            # Try 4 directions around the sofa with a small gap (2-3 cells)
            tw, th = random.choice([(1, 2), (2, 1), (2, 3), (3, 2)])

            # Simple offset attempts relative to Sofa
            offsets = [
                (x + sw + 2, y),  # Right
                (x - tw - 2, y),  # Left
                (x, y + sh + 2),  # Below
                (x, y - th - 2),  # Above
            ]
            random.shuffle(offsets)

            for tox, toy in offsets:
                if self._can_place(tox, toy, tw, th):
                    self._place_rect(tox, toy, tw, th)

                    # 3. Try Place Chair nearby the table
                    cw, ch = random.choice([(1, 1), (2, 2)])
                    # Try placing chair relative to Table
                    c_offsets = [
                        (tox + tw + 1, toy),
                        (tox - cw - 1, toy),
                        (tox, toy + th + 1),
                        (tox, toy - ch - 1),
                    ]
                    for cox, coy in c_offsets:
                        if self._can_place(cox, coy, cw, ch):
                            self._place_rect(cox, coy, cw, ch)
                            break  # Chair placed
                    break  # Table placed

    def _try_place_bookshelf(self, x, y):
        """
        Tries to place a bookshelf.
        Bookshelves look better if they are 'long' and placed.
        """
        # Bookshelf: W 1-2, L 3-5 (or 7)
        length = random.choice([3, 4, 5, 5, 7])  # Weighted towards 3-5
        width = random.randint(1, 2)

        bw, bh = (width, length) if random.choice([True, False]) else (length, width)

        if self._can_place(x, y, bw, bh):
            self._place_rect(x, y, bw, bh)


class DetailedMapVisualizer(QMainWindow):
    """
    Visualization with added support for Obstacles (Value 2).
    """

    def __init__(self, grid: np.ndarray):
        super().__init__()
        self.setWindowTitle("Procedural Environment - Furnished")
        self.setGeometry(100, 100, 900, 700)

        self.grid = grid
        self.scale_factor = 8

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        print("Rendering furnished map...")

    def paintEvent(self, event):
        painter = QPainter(self)
        h, w = self.grid.shape
        img = QImage(w, h, QImage.Format.Format_RGB32)

        # Color Palette
        COLOR_EMPTY = 0xFFEEEEEE  # Light Grey
        COLOR_WALL = 0xFF222222  # Dark Grey
        COLOR_OBSTACLE = 0xFF8B4513  # SaddleBrown (Wood-like)

        for y in range(h):
            for x in range(w):
                val = self.grid[y, x]
                if val == WALL:
                    img.setPixel(x, y, COLOR_WALL)
                elif val == OBSTACLE:
                    img.setPixel(x, y, COLOR_OBSTACLE)
                else:
                    img.setPixel(x, y, COLOR_EMPTY)

        # Scale
        scaled_w = w * self.scale_factor
        scaled_h = h * self.scale_factor
        pixmap = QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(
            scaled_w, scaled_h, Qt.AspectRatioMode.KeepAspectRatio
        )

        # Center
        draw_x = (self.width() - scaled_w) // 2
        draw_y = (self.height() - scaled_h) // 2

        painter.drawPixmap(draw_x, draw_y, scaled_pixmap)

        # Border
        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(draw_x, draw_y, scaled_w, scaled_h)

        # Legend (Simple text on top left)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(20, 30, "Legend: Dark=Wall, Brown=Furniture, Light=Empty")


def main():
    app = QApplication(sys.argv)

    # 1. Load and Populate
    # Ensure this matches the filename saved in the previous step
    FILENAME = "generated_grid_400.txt"

    placer = FurniturePlacer(FILENAME)

    # Check if grid loaded correctly (not all zeros)
    if np.all(placer.grid == 0):
        print(
            "Warning: Grid seems empty or file not found. Please run the generation script first."
        )

    final_grid = placer.populate()

    placer.save_to_file("obstacle_grid.txt")

    # 2. Visualize
    window = DetailedMapVisualizer(final_grid)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
