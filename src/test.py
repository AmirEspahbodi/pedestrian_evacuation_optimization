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
    domain = [domain for domain in environment.domains if domain.id == 10][0]

    app = QApplication(sys.argv)
    
    domain.add_emergency_accesses([(160, 3), (135, 3), (200, 3)])
    # Create and show main window
    window = DomainVisualizerWindow(domain)
    window.show()

    # time.sleep(10)
    engine_main(domain=domain, window=window)

    print("Simulation completed.")
    print(
        f"pedestrian left in are {domain.get_left_pedestrians_count()}."
    )

    sys.exit(app.exec())
    

if __name__ == "__main__":
    main()
