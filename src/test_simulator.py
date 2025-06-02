import sys
from .simulator.environment import Environment
from .simulator.draw import DomainVisualizerWindow
from .simulator.simulation_engine import main as engine_main
from src.config import SimulationConfig, OptimizerStrategy
from PyQt6.QtWidgets import QApplication
from src.optimizer.greedy import greedy_algorithm
from src.optimizer.ea import evolutionary_algorithm
from src.optimizer.iea import island_evolutionary_algorithm
from src.optimizer.psi import psi as psi_calcolator

def main():
    """Main entry point for the application."""
    # app = QApplication(sys.argv)

    # Create sample domain (replace with actual domain instance)
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    domain = [domain for domain in environment.domains if domain.id == 10][0]

    # emergency_accesses, fitness_value, evals = evolutionary_algorithm(
    #     domain=domain
    # )

    # domain.add_emergency_accesses(emergency_accesses)
    # # Create and show main window
    # window = DomainVisualizerWindow(domain)
    # window.show()

    # domain.is_simulation_finished = True
    # domain.calculate_peds_distance_to_nearest_exit()
    # window.updateGrid()
    # engine_main(domain=domain, window=window)

    # print("Simulation completed.")
    # print(
    #     f"pedestrian left in are {domain.get_left_pedestrians_count()}."
    # )

    # sys.exit(app.exec())
    
    psi_calcolator(domain, OptimizerStrategy.IEA, [124, 42, 277])


if __name__ == "__main__":
    main()
