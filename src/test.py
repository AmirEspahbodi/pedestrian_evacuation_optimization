import sys
from .simulator.environment import Environment
from .simulator.draw import DomainVisualizerWindow
from PyQt6.QtWidgets import QApplication
from src.simulator.simulation_engine import main as engine_main
from src.optimizer.psi import psi_helper


def main():
    """Main entry point for the application."""

    # Create sample domain (replace with actual domain instance)
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    for domain in environment.domains:
        print(f"id={domain.id}, p={2*(domain.height+domain.width)}")
    domain = [domain for domain in environment.domains if domain.id == 10][0]

    app = QApplication(sys.argv)
    e_exits = [(40, 2), (140, 2), (210, 2)]
    
    domain.add_emergency_accesses(e_exits)
    # Create and show main window
    window = DomainVisualizerWindow(domain)
    window.show()

    # time.sleep(10)
    engine_main(domain=domain, window=window)

    print("Simulation completed.")
    print(
        f"pedestrian left in are {domain.get_left_pedestrians_count()}."
    )
    print(psi_helper(7, domain, e_exits))

    sys.exit(app.exec())
    

if __name__ == "__main__":
    main()
