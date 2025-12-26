import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class GridVisualizer(QMainWindow):
    def __init__(self, filename, scale=10):
        super().__init__()
        self.filename = filename
        self.scale_factor = scale  # How many screen pixels represent 1 grid cell

        self.initUI()
        self.load_and_render_grid()

    def initUI(self):
        self.setWindowTitle("Procedural Floor Plan Viewer")
        self.resize(800, 600)

        # Main widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Scroll Area to handle large grids
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Label to hold the map image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

    def load_and_render_grid(self):
        """Loads the text file and converts it to a QImage."""
        if not os.path.exists(self.filename):
            QMessageBox.critical(self, "Error", f"File not found: {self.filename}")
            return

        try:
            # 1. Read the file using the logic provided
            with open(self.filename, "r") as f:
                grid = [[int(c) for c in line.strip()] for line in f]

            if not grid:
                raise ValueError("File is empty or invalid.")

            height = len(grid)
            width = len(grid[0])

            # 2. Create a QImage
            # Format_RGB32 is efficient for pixel manipulation
            image = QImage(width, height, QImage.Format.Format_RGB32)

            # 3. Define Color Map
            # 0: Empty, 1: Structure/Old Wall, 2: Furniture, 3: Doors/Other, 4: Walls
            colors = {
                0: QColor("#FFFFFF").rgb(),  # White (Walkable)
                1: QColor("#A9A9A9").rgb(),  # Dark Gray (Structure)
                2: QColor("#3498db").rgb(),  # Blue (Furniture)
                3: QColor("#2ecc71").rgb(),  # Green (Doors/Zones)
                4: QColor("#000000").rgb(),  # Black (Walls)
            }
            default_color = QColor("#FF00FF").rgb()  # Magenta for errors

            # 4. Populate the QImage pixel by pixel
            for y in range(height):
                row = grid[y]
                for x in range(min(width, len(row))):
                    val = row[x]
                    image.setPixel(x, y, colors.get(val, default_color))

            # 5. Scale the image for visibility (Nearest Neighbor to keep pixels sharp)
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(
                width * self.scale_factor,
                height * self.scale_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

            self.image_label.setPixmap(scaled_pixmap)
            self.statusBar().showMessage(
                f"Loaded Grid: {width}x{height} | Scale: {self.scale_factor}x"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load grid:\n{str(e)}")


if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)

    # Name of your file
    FILE_NAME = "dataset/p400/finall_grid_400_1.txt"

    # Create and show window
    # scale=5 means 1 grid cell will be a 5x5 pixel square on screen
    viewer = GridVisualizer(FILE_NAME, scale=5)
    viewer.show()

    sys.exit(app.exec())
