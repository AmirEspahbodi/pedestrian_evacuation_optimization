import os
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget


class GridWidget(QWidget):
    def __init__(self, grid_data, pedestrians_data):
        super().__init__()
        self.grid_data = grid_data
        self.pedestrians_data = pedestrians_data

        # Color definitions for faster rendering
        self.colors = {
            0: QColor(240, 240, 240),  # Empty space (Light Gray)
            2: QColor(100, 149, 237),  # Furniture/Obstacle (Cornflower Blue)
            3: QColor(
                50, 205, 50
            ),  # Exit (Lime Green) - Assuming 3 is exit based on digits provided
            4: QColor(50, 50, 50),  # Walls (Dark Gray/Black)
        }
        self.pedestrian_color = QColor(220, 20, 60)  # Crimson Red

    def update_pedestrians(self, new_pedestrians_data):
        """
        Updates the pedestrian overlay and triggers a redraw.

        Args:
            new_pedestrians_data: A 2D list/array where 1 indicates a pedestrian.
        """
        self.pedestrians_data = new_pedestrians_data
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        self.draw_grid(qp)

    def draw_grid(self, qp):
        if not self.grid_data or not self.grid_data[0]:
            return

        rows = len(self.grid_data)
        cols = len(self.grid_data[0])

        # Calculate dynamic cell size to fit the window
        widget_width = self.width()
        widget_height = self.height()

        cell_w = widget_width / cols
        cell_h = widget_height / rows

        # Turn off borders for a cleaner look on dense grids
        qp.setPen(Qt.PenStyle.NoPen)

        for r in range(rows):
            for c in range(cols):
                # 1. Draw Static Grid Elements (Walls, Furniture, Exits)
                cell_value = self.grid_data[r][c]

                if cell_value in self.colors:
                    qp.setBrush(QBrush(self.colors[cell_value]))
                    # Draw rectangle: x, y, width, height
                    qp.drawRect(
                        int(c * cell_w),
                        int(r * cell_h),
                        int(cell_w) + 1,
                        int(cell_h) + 1,
                    )

                # 2. Draw Pedestrians (Overlay)
                # Assuming pedestrians_data is same dim as grid, where 1 = person
                if r < len(self.pedestrians_data) and c < len(self.pedestrians_data[0]):
                    if self.pedestrians_data[r][c] == 1:
                        qp.setBrush(QBrush(self.pedestrian_color))
                        # Draw pedestrian as a circle
                        qp.drawEllipse(
                            int(c * cell_w), int(r * cell_h), int(cell_w), int(cell_h)
                        )


class MainWindow(QMainWindow):
    def __init__(self, gird, pedestrians):
        super().__init__()
        self.setWindowTitle("Floor Plan & Pedestrian Visualizer")
        self.resize(800, 800)

        # 1. Load the Grid
        self.grid = gird
        self.pedestrians = pedestrians

        # Setup UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Initialize the Visualization Widget
        self.grid_viewer = GridWidget(self.grid, self.pedestrians)
        self.layout.addWidget(self.grid_viewer)
