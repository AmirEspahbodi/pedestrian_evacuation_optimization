import random
import sys
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

# Constants for grid values
EMPTY = 0
OBSTACLE = 1


class ProceduralGenerator:
    """
    Handles the algorithmic generation of the 2D floor plan.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)

        # Constraints
        self.min_dist_between_rooms = 4  # "> 3" implies 4
        self.min_dist_to_wall = 6  # "> 5" implies 6
        self.door_len_min = 4
        self.door_len_max = 6

    def func1(self, x: int, y: int, w: int, h: int):
        """
        Stub function to populate obstacles.
        Logic: Within the specified sub-region, randomly toggle cells to 1.
        """
        # NOTE: Stubbed as requested.
        # Real implementation would go here (e.g., cellular automata or random noise)
        return

    def save_to_file(self, filename: str):
        """
        Saves the current grid to a text file using NumPy.
        0 = Empty, 1 = Wall.
        """
        try:
            # fmt='%d' saves as integers. delimiter='' makes it a dense block of text.
            np.savetxt(filename, self.grid, fmt="%d", delimiter="")
            print(f"[Success] Grid saved to local file: {filename}")
        except Exception as e:
            print(f"[Error] Failed to save grid: {e}")

    def _draw_room_walls(self, x: int, y: int, w: int, h: int) -> None:
        """
        Draws a rectangular border (walls) with a door cut out.
        """
        # Draw the box walls
        self.grid[y, x : x + w] = OBSTACLE  # Top
        self.grid[y + h - 1, x : x + w] = OBSTACLE  # Bottom
        self.grid[y : y + h, x] = OBSTACLE  # Left
        self.grid[y : y + h, x + w - 1] = OBSTACLE  # Right

        # Cut a door
        valid_walls = []
        door_len = random.randint(self.door_len_min, self.door_len_max)

        # Check constraints for door placement
        if w > door_len + 2:
            valid_walls.extend(["top", "bottom"])
        if h > door_len + 2:
            valid_walls.extend(["left", "right"])

        if not valid_walls:
            return

        wall_choice = random.choice(valid_walls)

        if wall_choice == "top":
            start = random.randint(x + 1, x + w - 1 - door_len)
            self.grid[y, start : start + door_len] = EMPTY
        elif wall_choice == "bottom":
            start = random.randint(x + 1, x + w - 1 - door_len)
            self.grid[y + h - 1, start : start + door_len] = EMPTY
        elif wall_choice == "left":
            start = random.randint(y + 1, y + h - 1 - door_len)
            self.grid[start : start + door_len, x] = EMPTY
        elif wall_choice == "right":
            start = random.randint(y + 1, y + h - 1 - door_len)
            self.grid[start : start + door_len, x + w - 1] = EMPTY

    def _apply_room_logic(self, zone_x, zone_y, zone_w, zone_h):
        """
        Calculates room padding and walls, returns inner area.
        """
        pad_left = pad_right = pad_top = pad_bottom = 2

        # Boundary checks for padding > 5 (set to 6)
        if zone_x == 0:
            pad_left = self.min_dist_to_wall
        if zone_y == 0:
            pad_top = self.min_dist_to_wall
        if zone_x + zone_w >= self.width:
            pad_right = self.min_dist_to_wall
        if zone_y + zone_h >= self.height:
            pad_bottom = self.min_dist_to_wall

        room_x = zone_x + pad_left
        room_y = zone_y + pad_top
        room_w = zone_w - (pad_left + pad_right)
        room_h = zone_h - (pad_top + pad_bottom)

        if room_w < 4 or room_h < 4:
            return zone_x, zone_y, zone_w, zone_h

        self._draw_room_walls(room_x, room_y, room_w, room_h)
        return room_x + 1, room_y + 1, room_w - 2, room_h - 2

    def generate(self):
        """
        Main execution logic.
        """
        self.grid.fill(EMPTY)

        # Step 1: Primary Division
        v1 = random.randint(int(self.width * 0.30), int(self.width * 0.40))
        v2 = random.randint(int(self.width * 0.65), int(self.width * 0.75))
        h1 = random.randint(int(self.height * 0.40), int(self.height * 0.60))

        zones = [
            (0, 0, v1, h1),
            (v1, 0, v2 - v1, h1),
            (v2, 0, self.width - v2, h1),
            (0, h1, v1, self.height - h1),
            (v1, h1, v2 - v1, self.height - h1),
            (v2, h1, self.width - v2, self.height - h1),
        ]

        # Step 2: Primary Room Assignment
        num_rooms = random.randint(2, 4)
        room_indices = set(random.sample(range(6), num_rooms))

        # Step 3: Processing
        for i, (zx, zy, zw, zh) in enumerate(zones):
            cx, cy, cw, ch = zx, zy, zw, zh
            if i in room_indices:
                cx, cy, cw, ch = self._apply_room_logic(zx, zy, zw, zh)

            is_case_a = cw < 10 and ch < 10
            is_case_b = cw > 10 and ch > 10

            if is_case_a:
                self.func1(cx, cy, cw, ch)

            elif is_case_b:
                sub_v = random.randint(int(cw * 0.45), int(cw * 0.55))
                sub_h = random.randint(int(ch * 0.45), int(ch * 0.55))

                sub_zones = [
                    (cx, cy, sub_v, sub_h),
                    (cx + sub_v, cy, cw - sub_v, sub_h),
                    (cx, cy + sub_h, sub_v, ch - sub_h),
                    (cx + sub_v, cy + sub_h, cw - sub_v, ch - sub_h),
                ]

                sub_room_count = random.randint(1, 2)
                sub_room_indices = set(random.sample(range(4), sub_room_count))

                for si, (sx, sy, sw, sh) in enumerate(sub_zones):
                    fsx, fsy, fsw, fsh = sx, sy, sw, sh
                    if si in sub_room_indices:
                        fsx, fsy, fsw, fsh = self._apply_room_logic(sx, sy, sw, sh)
                    self.func1(fsx, fsy, fsw, fsh)
            else:
                self.func1(cx, cy, cw, ch)

        return self.grid


class MapVisualizer(QMainWindow):
    """
    PyQt6 Main Window to visualize the generated grid.
    """

    def __init__(self, width: int, height: int, grid: np.ndarray):
        super().__init__()
        self.setWindowTitle("Procedural 2D Environment")
        self.setGeometry(100, 100, 800, 600)

        self.sim_width = width
        self.sim_height = height
        self.grid = grid
        self.scale_factor = 8

        # Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        print("Rendering map...")

    def paintEvent(self, event):
        painter = QPainter(self)
        h, w = self.grid.shape
        img = QImage(w, h, QImage.Format.Format_RGB32)

        COLOR_EMPTY = 0xFFDDDDDD
        COLOR_WALL = 0xFF222222

        for y in range(h):
            for x in range(w):
                if self.grid[y, x] == OBSTACLE:
                    img.setPixel(x, y, COLOR_WALL)
                else:
                    img.setPixel(x, y, COLOR_EMPTY)

        scaled_w = w * self.scale_factor
        scaled_h = h * self.scale_factor
        pixmap = QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(
            scaled_w, scaled_h, Qt.AspectRatioMode.KeepAspectRatio
        )

        win_w = self.width()
        win_h = self.height()
        draw_x = (win_w - scaled_w) // 2
        draw_y = (win_h - scaled_h) // 2

        painter.drawPixmap(draw_x, draw_y, scaled_pixmap)

        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(draw_x, draw_y, scaled_w, scaled_h)


def main():
    app = QApplication(sys.argv)

    # 1. Define Dimensions
    SIM_WIDTH = 100
    SIM_HEIGHT = 100

    # 2. Generate Grid
    generator = ProceduralGenerator(SIM_WIDTH, SIM_HEIGHT)
    grid = generator.generate()

    # 3. Save to File (The new requirement)
    # Using 'generated_grid.txt' in the current directory
    generator.save_to_file(f"initial_gird_{str(2 * SIM_WIDTH + 2 * SIM_HEIGHT)}.txt")

    # 4. Visualize
    window = MapVisualizer(SIM_WIDTH, SIM_HEIGHT, grid)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
