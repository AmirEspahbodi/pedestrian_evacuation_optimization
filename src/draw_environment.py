import sys
from .simulator import Environment, Domain, CAState
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QStatusBar,
                            QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QMouseEvent



class GridWidget(QWidget):
    """Custom widget for rendering the domain grid with cells."""
    
    cellClicked = pyqtSignal(int, int, str)  # x, y, state
    
    def __init__(self, domain: 'Domain', parent=None):
        super().__init__(parent)
        self.domain = domain
        self.cell_size = 30  # Base size for each cell
        self.grid_line_width = 1
        self.hover_cell = None
        self.selected_cell = None
        
        # Color scheme with modern, accessible colors
        self.colors = {
            CAState.EMPTY: QColor(255, 255, 255),      # White
            CAState.OBSTACLE: QColor(33, 33, 33),       # Near black
            CAState.OCCUPIED: QColor(41, 128, 185),     # Professional blue
            CAState.ACCESS: QColor(39, 174, 96),        # Vibrant green
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
            self.drawHighlight(painter, self.hover_cell[0], self.hover_cell[1], 
                             self.hover_color)
        
        # Draw selection highlight
        if self.selected_cell:
            self.drawHighlight(painter, self.selected_cell[0], self.selected_cell[1], 
                             self.selected_color)
    
    def drawCell(self, painter: QPainter, x: int, y: int):
        """Draw a single cell."""
        cell = self.domain.cells[y][x]
        color = self.colors.get(cell.state, QColor(255, 255, 255))
        
        rect = QRect(
            int(x * self.cell_size + self.grid_line_width),
            int(y * self.cell_size + self.grid_line_width),
            int(self.cell_size - self.grid_line_width),
            int(self.cell_size - self.grid_line_width)
        )
        
        painter.fillRect(rect, color)
    
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
            int(self.cell_size - self.grid_line_width)
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
                cell = self.domain.cells[y][x]
                self.cellClicked.emit(x, y, cell.state)
                self.update()
    
    def leaveEvent(self, event):
        """Clear hover when mouse leaves widget."""
        self.hover_cell = None
        self.update()
    
    def setZoom(self, zoom_level: int):
        """Adjust cell size for zoom functionality."""
        self.cell_size = max(10, min(100, zoom_level))
        self.updateSize()
        self.update()


class DomainVisualizerWindow(QMainWindow):
    """Main window for the domain grid visualizer."""
    
    def __init__(self, domain: 'Domain'):
        super().__init__()
        self.domain = domain
        self.initUI()
        
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
        
        info_label = QLabel(f"Grid Size: {self.domain.width} x {self.domain.height}")
        info_label.setStyleSheet("color: #666;")
        header_layout.addWidget(info_label)
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Create legend
        legend_frame = QFrame()
        legend_frame.setFrameStyle(QFrame.Shape.Box)
        legend_frame.setStyleSheet("background-color: white; border: 1px solid #ddd; padding: 10px;")
        legend_layout = QHBoxLayout(legend_frame)
        
        legend_label = QLabel("Legend:")
        legend_label.setStyleSheet("font-weight: bold;")
        legend_layout.addWidget(legend_label)
        
        legend_items = [
            ("Empty", "white", "#000"),
            ("Obstacle", "#212121", "#fff"),
            ("Occupied", "#2980b9", "#fff"),
            ("Access", "#27ae60", "#fff")
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
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: white;
                border: 2px solid #ddd;
            }
        """)
        
        # Create grid widget
        self.grid_widget = GridWidget(self.domain)
        self.grid_widget.cellClicked.connect(self.onCellClicked)
        
        scroll_area.setWidget(self.grid_widget)
        scroll_area.setWidgetResizable(False)
        main_layout.addWidget(scroll_area, 1)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set window size
        self.resize(800, 600)
        self.centerWindow()
    
    def centerWindow(self):
        """Center the window on screen."""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def onCellClicked(self, x: int, y: int, state: str):
        """Handle cell click events."""
        self.status_bar.showMessage(f"Cell clicked at ({x}, {y}) - State: {state}")
    
    def zoomIn(self):
        """Increase zoom level."""
        current_size = self.grid_widget.cell_size
        self.grid_widget.setZoom(current_size + 10)
        self.status_bar.showMessage(f"Zoom: {self.grid_widget.cell_size}px per cell", 2000)
    
    def zoomOut(self):
        """Decrease zoom level."""
        current_size = self.grid_widget.cell_size
        self.grid_widget.setZoom(current_size - 15)
        self.status_bar.showMessage(f"Zoom: {self.grid_widget.cell_size}px per cell", 2000)
    
    def resetView(self):
        """Reset to default zoom level."""
        self.grid_widget.setZoom(30)
        self.grid_widget.selected_cell = None
        self.grid_widget.update()
        self.status_bar.showMessage("View reset", 2000)


def create_sample_domain():
    """Create a sample domain for testing."""
    # This is a mock implementation since we don't have the actual Domain class
    class MockCell:
        def __init__(self, state, x, y):
            self.state = state
            self.x = x
            self.y = y
    
    class MockDomain:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.cells = []
            
            # Create sample grid with different cell types
            for y in range(height):
                row = []
                for x in range(width):
                    if x == 0 or x == width-1 or y == 0 or y == height-1:
                        state = CAState.OBSTACLE
                    elif x == 5 and y == 5:
                        state = CAState.ACCESS
                    elif (x + y) % 7 == 0:
                        state = CAState.OCCUPIED
                    else:
                        state = CAState.EMPTY
                    row.append(MockCell(state, x, y))
                self.cells.append(row)
    
    return MockDomain(20, 15)



def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    
    # Create sample domain (replace with actual domain instance)
    environment = Environment.from_json_file("dataset/environments/environment-example-supermarket.json")
    domain = environment.domains[0]
    
    # Create and show main window
    window = DomainVisualizerWindow(domain)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
