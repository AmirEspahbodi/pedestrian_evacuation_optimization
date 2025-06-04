import sys
import random
from .domain import Domain, CAState


from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStatusBar,
    QScrollArea,
    QFrame,
)
from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QMouseEvent, QFont


class GridWidget(QWidget):
    """Custom widget for rendering the domain grid with cells."""

    cellClicked = pyqtSignal(int, int, str)  # x, y, state

    def __init__(self, domain: "Domain", parent=None):
        super().__init__(parent)
        self.domain = domain
        self.cell_size = 7  # Base size for each cell
        self.grid_line_width = 1
        self.hover_cell = None
        self.selected_cell = None

        # Initialize random numbers for each cell
        self.cell_numbers = {}
        for y in range(self.domain.height):
            for x in range(self.domain.width):
                self.cell_numbers[(x, y)] = random.randint(0, 99)

        # Color scheme with modern, accessible colors
        self.colors = {
            CAState.EMPTY: QColor(255, 255, 255),
            CAState.OBSTACLE: QColor(33, 33, 33),
            CAState.OCCUPIED: QColor(41, 128, 185),
            CAState.ACCESS: QColor(39, 174, 96),
            CAState.ACCESS_OCCUPIED: QColor(128, 0, 128),
            CAState.EMERGENCY_ACCESS: QColor(238, 75, 43),
            CAState.EMERGENCY_ACCESS_OCCUPIED: QColor(128, 0, 128),
        }

        # Additional UI colors
        self.grid_color = QColor(200, 200, 200)
        self.hover_color = QColor(255, 235, 59, 100)  # Semi-transparent yellow
        self.selected_color = QColor(255, 152, 0, 150)  # Semi-transparent orange

        self.setMouseTracking(True)
        self.updateSize()

    def updateSize(self):
        """Update widget size based on domain dimensions."""
        width = int(self.domain.width * self.cell_size + self.grid_line_width)
        height = int(self.domain.height * self.cell_size + self.grid_line_width)
        self.setFixedSize(width, height)

    def paintEvent(self, event):
        """Paint the grid and cells."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), Qt.GlobalColor.white)

        # Draw cells
        for y in range(self.domain.height):
            for x in range(self.domain.width):
                self.drawCell(painter, x, y)

        # Draw grid lines
        self.drawGrid(painter)

        # Draw hover highlight
        if self.hover_cell:
            self.drawHighlight(
                painter, self.hover_cell[0], self.hover_cell[1], self.hover_color
            )

        # Draw selection highlight
        if self.selected_cell:
            self.drawHighlight(
                painter,
                self.selected_cell[0],
                self.selected_cell[1],
                self.selected_color,
            )

    def drawCell(self, painter: QPainter, x: int, y: int):
        """Draw a single cell."""
        state = self.domain._get_state(y, x)
        color = self.colors.get(state, QColor(255, 255, 255))

        rect = QRect(
            int(x * self.cell_size + self.grid_line_width),
            int(y * self.cell_size + self.grid_line_width),
            int(self.cell_size - self.grid_line_width),
            int(self.cell_size - self.grid_line_width),
        )

        painter.fillRect(rect, color)

        # Draw random number in cell if cell is large enough
        # if self.domain.is_simulation_finished:
        #     number = "{:.2f}".format(self.domain.cells[y][x].static_field)

        #     # Set font size based on cell size
        #     font_size = max(4, self.cell_size // 2 - 5)
        #     font = QFont("Arial", font_size)
        #     painter.setFont(font)

        #     # Set text color based on background (white text on dark backgrounds, black on light)
        #     if state in [CAState.OBSTACLE]:
        #         painter.setPen(QColor(255, 255, 255))  # White text
        #     else:
        #         painter.setPen(QColor(0, 0, 0))  # Black text

        #     # Draw the number centered in the cell
        #     painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(number))

    def drawGrid(self, painter: QPainter):
        """Draw grid lines."""
        pen = QPen(self.grid_color, self.grid_line_width)
        painter.setPen(pen)

        # Vertical lines
        for x in range(self.domain.width + 1):
            x_pos = x * self.cell_size
            painter.drawLine(x_pos, 0, x_pos, self.height())

        # Horizontal lines
        for y in range(self.domain.height + 1):
            y_pos = y * self.cell_size
            painter.drawLine(0, y_pos, self.width(), y_pos)

    def drawHighlight(self, painter: QPainter, x: int, y: int, color: QColor):
        """Draw highlight overlay for a cell."""
        rect = QRect(
            int(x * self.cell_size + self.grid_line_width),
            int(y * self.cell_size + self.grid_line_width),
            int(self.cell_size - self.grid_line_width),
            int(self.cell_size - self.grid_line_width),
        )
        painter.fillRect(rect, color)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement for hover effects."""
        x = int(event.position().x() // self.cell_size)
        y = int(event.position().y() // self.cell_size)

        if 0 <= x < self.domain.width and 0 <= y < self.domain.height:
            if self.hover_cell != (x, y):
                self.hover_cell = (x, y)
                self.update()
        else:
            if self.hover_cell:
                self.hover_cell = None
                self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse clicks."""
        if event.button() == Qt.MouseButton.LeftButton:
            x = int(event.position().x() // self.cell_size)
            y = int(event.position().y() // self.cell_size)

            if 0 <= x < self.domain.width and 0 <= y < self.domain.height:
                self.selected_cell = (x, y)
                state = self.domain._get_state(y, x)
                self.cellClicked.emit(x, y, state)
                self.update()

    def leaveEvent(self, event):
        """Clear hover when mouse leaves widget."""
        self.hover_cell = None
        self.update()

    def setZoom(self, zoom_level: int):
        """Adjust cell size for zoom functionality."""
        self.cell_size = max(5, min(100, zoom_level))
        self.updateSize()
        self.update()


class DomainVisualizerWindow(QMainWindow):
    """Main window for the domain grid visualizer."""

    def __init__(self, domain: "Domain"):
        super().__init__()
        self.domain = domain
        self.initUI()

    def updateGrid(self):
        """Update the grid widget to reflect changes in domain.cells.
        This method should be called whenever the cellular automaton process
        modifies the domain cells to ensure the visualization stays synchronized."""
        if hasattr(self, "grid_widget"):
            # Force the grid widget to repaint with updated cell states
            self.grid_widget.update()

            # Update step counter
            self.step_count += 1
            self.step_label.setText(f"Step: {self.step_count}")

            # Update status bar with current step information
            occupied_count = sum(self.domain.peds)
            self.status_bar.showMessage(
                f"Step {self.step_count} - Occupied cells: {occupied_count}"
            )

            # Update the application's event loop to ensure smooth animation
            QApplication.processEvents()

            # Force immediate repaint for smooth animation
            self.grid_widget.repaint()

            # Ensure the scroll area updates if needed
            self.scroll_area.viewport().update()

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle("Domain Grid Visualizer")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                font-size: 14px;
                padding: 5px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QStatusBar {
                background-color: #e0e0e0;
                border-top: 1px solid #bdbdbd;
            }
        """)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create header with title and info
        header_layout = QHBoxLayout()
        title_label = QLabel("Domain Grid Visualization")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        header_layout.addWidget(title_label)

        self.info_label = QLabel(
            f"Grid Size: {self.domain.width} x {self.domain.height}"
        )
        self.info_label.setStyleSheet("color: #666;")
        header_layout.addWidget(self.info_label)

        # Add step counter for process tracking
        self.step_label = QLabel("Step: 0")
        self.step_label.setStyleSheet("color: #666;")
        header_layout.addWidget(self.step_label)

        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Create legend
        legend_frame = QFrame()
        legend_frame.setFrameStyle(QFrame.Shape.Box)
        legend_frame.setStyleSheet(
            "background-color: white; border: 1px solid #ddd; padding: 10px;"
        )
        legend_layout = QHBoxLayout(legend_frame)

        legend_label = QLabel("Legend:")
        legend_label.setStyleSheet("font-weight: bold;")
        legend_layout.addWidget(legend_label)

        legend_items = [
            ("Empty", "white", "#000"),
            ("Obstacle", "#212121", "#fff"),
            ("Occupied", "#2980b9", "#fff"),
            ("Access", "#27ae60", "#fff"),
        ]

        for name, bg_color, text_color in legend_items:
            item_label = QLabel(name)
            item_label.setStyleSheet(f"""
                background-color: {bg_color};
                color: {text_color};
                padding: 4px 8px;
                border-radius: 3px;
                font-weight: bold;
            """)
            legend_layout.addWidget(item_label)

        legend_layout.addStretch()
        main_layout.addWidget(legend_frame)

        # Create control panel
        control_layout = QHBoxLayout()

        zoom_in_btn = QPushButton("Zoom In (+)")
        zoom_in_btn.clicked.connect(self.zoomIn)
        control_layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("Zoom Out (-)")
        zoom_out_btn.clicked.connect(self.zoomOut)
        control_layout.addWidget(zoom_out_btn)

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.resetView)
        control_layout.addWidget(reset_btn)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # Create scroll area for grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: white;
                border: 2px solid #ddd;
            }
        """)

        # Create grid widget
        self.grid_widget = GridWidget(self.domain)
        self.grid_widget.cellClicked.connect(self.onCellClicked)

        self.scroll_area.setWidget(self.grid_widget)
        self.scroll_area.setWidgetResizable(False)
        main_layout.addWidget(self.scroll_area, 1)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Initialize step counter
        self.step_count = 0

        # self.showFullScreen()
        self.resize(800, 900)
        # self.centerWindow()

    def centerWindow(self):
        """Center the window on screen."""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2
        )

    def onCellClicked(self, x: int, y: int, state: str):
        """Handle cell click events."""
        self.status_bar.showMessage(f"Cell clicked at ({x}, {y}) - State: {state}")

    def zoomIn(self):
        """Increase zoom level."""
        current_size = self.grid_widget.cell_size
        self.grid_widget.setZoom(current_size + 3)
        self.status_bar.showMessage(
            f"Zoom: {self.grid_widget.cell_size}px per cell", 2000
        )

    def zoomOut(self):
        """Decrease zoom level."""
        current_size = self.grid_widget.cell_size
        self.grid_widget.setZoom(current_size - 3)
        self.status_bar.showMessage(
            f"Zoom: {self.grid_widget.cell_size}px per cell", 2000
        )

    def resetView(self):
        """Reset to default zoom level."""
        self.grid_widget.setZoom(3)
        self.grid_widget.selected_cell = None
        self.grid_widget.update()
        self.status_bar.showMessage("View reset", 2000)
