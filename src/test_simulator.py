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
        "dataset/environments/environment-example-supermarket.json"
    )
    domain = environment.domains[0]

    # Create and show main window
    window = DomainVisualizerWindow(domain)
    window.show()

    engine_main(domain=domain, window=window)

    print("Simulation completed.")
    print(f"pedestrian left in are {domain.get_left_pedestrians()}.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
