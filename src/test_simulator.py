import sys
from .simulator.environment import Environment
from .simulator.draw import DomainVisualizerWindow
from .simulator.simulation_engine import main as engine_main
from src.config import SimulationConfig
from PyQt6.QtWidgets import QApplication
from src.optimizer.greedy import greedy_algorithm
from src.optimizer.ea import evolutionary_algorithm
from src.optimizer.iea import island_evolutionary_algorithm


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)

    # Create sample domain (replace with actual domain instance)
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )

    emergency_accesses, fitness_value, evals = evolutionary_algorithm(
        domain=environment.domains[5]
    )

    environment.domains[5].add_emergency_accesses(emergency_accesses)
    # Create and show main window
    window = DomainVisualizerWindow(environment.domains[5])
    window.show()

    environment.domains[5].is_simulation_finished = True
    environment.domains[5].calculate_peds_distance_to_nearest_exit()
    window.updateGrid()
    engine_main(domain=environment.domains[5], window=window)

    print("Simulation completed.")
    print(
        f"pedestrian left in are {environment.domains[5].get_left_pedestrians_count()}."
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
