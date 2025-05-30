import sys
from .simulator.environment import Environment
from .simulator.draw import DomainVisualizerWindow
from .simulator.simulation_engine import main as engine_main
from src.optimizer.psi import psi
from PyQt6.QtWidgets import QApplication


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
    environment.domains[5].calculate_peds_distance_to_nearest_exit()
    window.updateGrid()
    print(psi(domain=environment.domains[5]), (1, 2), (4, 2))

    print("Simulation completed.")
    print(
        f"pedestrian left in are {environment.domains[5].get_left_pedestrians_count()}."
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
