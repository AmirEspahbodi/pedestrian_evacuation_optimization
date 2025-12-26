import sys
from time import sleep

import numpy as np
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

from src.config import load_simulation_config
from src.simulator.draw import MainWindow
from src.simulator.simulation_engine import main as engine
from src.utils import add_emergency_exits


class SimulationWorker(QThread):
    def __init__(self, simulator_config, grid, pedestrian, window):
        super().__init__()
        self.simulator_config = simulator_config
        self.grid = grid
        self.pedestrian = pedestrian
        self.window = window

    def run(self):
        # This method runs in a background thread
        # It allows the main GUI thread to keep breathing
        t_star = engine(self.simulator_config, self.grid, self.pedestrian, self.window)
        print(t_star)


if __name__ == "__main__":
    FILE_NAME = "dataset/p400/finall_grid_400_1.txt"
    simulator_config = load_simulation_config("dataset/simulation.json")

    with open(FILE_NAME, "r") as f:
        raw_grid = [[int(c) for c in line.strip()] for line in f]

    pedestrian = np.array(
        [[1 if cell == 3 else 0 for cell in row] for row in raw_grid], int
    )

    clean_grid = [[0 if cell == 3 else cell for cell in row] for row in raw_grid]

    add_emergency_exits(clean_grid, [(298, 5), (200, 3), (10, 3)])

    app = QApplication(sys.argv)

    window = MainWindow(clean_grid, pedestrian)
    window.show()

    sim_thread = SimulationWorker(simulator_config, clean_grid, pedestrian, window)

    app.aboutToQuit.connect(sim_thread.terminate)

    sim_thread.start()

    sys.exit(app.exec())
