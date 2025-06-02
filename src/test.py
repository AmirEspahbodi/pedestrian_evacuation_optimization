import sys
from .simulator.environment import Environment
from .simulator.draw import DomainVisualizerWindow
from PyQt6.QtWidgets import QApplication
import time
from src.simulator.simulation_engine import main as engine_main

def main():
    """Main entry point for the application."""

    # Create sample domain (replace with actual domain instance)
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    domain = [domain for domain in environment.domains if domain.id == 5][0]

    app = QApplication(sys.argv)
    
    domain.add_emergency_accesses([(112, 2)])
    # Create and show main window
    window = DomainVisualizerWindow(domain)
    window.show()

    domain.is_simulation_finished = True
    domain.calculate_peds_distance_to_nearest_exit()
    window.updateGrid()
    # time.sleep(10)
    # engine_main(domain=domain, window=window)

    print("Simulation completed.")
    print(
        f"pedestrian left in are {domain.get_left_pedestrians_count()}."
    )

    sys.exit(app.exec())
    
    # domain.test_emergency_accesses((0, 2), {(0, 0), (0, 1)})
    # domain.test_emergency_accesses((1, 2), {(0, 1), (0, 2)})
    # domain.test_emergency_accesses((65, 2), {(0, 65), (0, 66)})
    # domain.test_emergency_accesses((69, 2), {(0, 69), (0, 70)})
    # domain.test_emergency_accesses((70, 2), {(0, 70), (1, 70)})
    # domain.test_emergency_accesses((71, 2), {(1, 70), (2, 70)})
    # domain.test_emergency_accesses((72, 2), {(2, 70), (3, 70)})
    # domain.test_emergency_accesses((100, 2), {(30, 70), (31, 70)})
    # domain.test_emergency_accesses((109, 2), {(39, 70), (40, 70)})
    # domain.test_emergency_accesses((110, 2), {(40, 70), (40, 69)})
    # domain.test_emergency_accesses((111, 2), {(40, 69), (40, 68)})
    # domain.test_emergency_accesses((112, 2), {(40, 68), (40, 67)})
    # domain.test_emergency_accesses((113, 2), {(40, 67), (40, 66)})
    # domain.test_emergency_accesses((140, 2), {(40, 40), (40, 39)})
    # domain.test_emergency_accesses((179, 2), {(40, 1), (40, 0)})
    # domain.test_emergency_accesses((180, 2), {(40, 0), (39, 0)})
    # domain.test_emergency_accesses((181, 2), {(39, 0), (38, 0)})
    # domain.test_emergency_accesses((200, 2), {(20, 0), (19, 0)})
    # domain.test_emergency_accesses((219, 2), {(1, 0), (0, 0)})

if __name__ == "__main__":
    main()
