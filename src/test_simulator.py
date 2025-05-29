import sys
from .simulator.environment import Environment
from .simulator.draw import DomainVisualizerWindow
from .simulator.simulation_engine import main as engine_main

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
    window = DomainVisualizerWindow(environment.domains[0])
    window.show()

    engine_main(domain=environment.domains[0], window=window)
    print(f"len obstacles = {len(environment.domains[0].obstacles)} .")

    print("Simulation completed.")
    print(f"pedestrian left in are {environment.domains[0].get_left_pedestrians()}.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
