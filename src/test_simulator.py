import sys
from .simulator.environment import Environment
from .simulator.draw import DomainVisualizerWindow
from .simulator.simulation_engine import main as engine_main
from .optimizer.fitness import fitness
import time

# from .simulator.pedestrian_evacuation.pedestrian_movement_process import PedestrianMovementModelProcess
from PyQt6.QtWidgets import QApplication
import numpy as np


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)

    # Create sample domain (replace with actual domain instance)
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )

    # Create and show main window
    window = DomainVisualizerWindow(environment.domains[5])
    window.show()

    engine_main(domain=environment.domains[5], window=window)
    environment.domains[5].is_simulation_finished = True
    environment.domains[5].calculate_nearest_exit_distances()

    window.updateGrid()

    fitness(domain=environment.domains[5])

    print("Simulation completed.")
    print(
        f"pedestrian left in are {environment.domains[5].get_left_pedestrians_count()}."
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
