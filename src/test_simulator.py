import sys
from .simulator.environment import Environment
from .simulator.draw_environment import DomainVisualizerWindow
from .simulator.pedestrian_movement_process import PedestrianMovementModelProcess
from PyQt6.QtWidgets import QApplication


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

    PedestrianMovementModelProcess(environment,window=window).process()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
