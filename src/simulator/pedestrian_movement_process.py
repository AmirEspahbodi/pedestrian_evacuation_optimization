import math
from random import choice
import time

from src.utils import select_by_probability_normalized
from src.config import SimulationConfig
from .environment import Environment, CAState, CellularAutomata, Domain
from .draw_environment import DomainVisualizerWindow


class PedestrianMovementModelProcess:
    def __init__(self, environment: Environment, window: DomainVisualizerWindow|None = None):
        self.window = window
        self.environment = environment
        pass

    def process(
        self,
    ):
        for time_step in range(0, SimulationConfig.simulator.time_limit):
            self._calculate_desirability(self.environment.domains[0])
            self._move_pedestrian(self.environment.domains[0])

            if self.window:
                self.window.updateGrid()
            print(f"Time step: {time_step}")

    def _move_pedestrian(self, domain: Domain):
        pedestrians = domain.get_pedestrians()
        pedestrians.sort(key=lambda x: x.static_filed)
        while pedestrians:
            pedestrian = pedestrians.pop()
            y, x = pedestrian.y, pedestrian.x
            if pedestrian.state == CAState.ACCESS_OCCUPIED:
                # If the pedestrian is already in an access cell, skip moving
                pedestrian.state = CAState.ACCESS
                continue
            transition_probability = self._transition_function(domain, y, x)
            neighbor_to_move = select_by_probability_normalized(transition_probability)
            if neighbor_to_move.state == CAState.EMPTY:
                neighbor_to_move.state = CAState.OCCUPIED
                pedestrian.state = CAState.EMPTY
            elif neighbor_to_move.state == CAState.ACCESS:
                neighbor_to_move.state = CAState.ACCESS_OCCUPIED
                pedestrian.state = CAState.EMPTY
            
                

            if self.window:
                self.window.updateGrid()
                time.sleep(0.01)

    def _transition_function(
        self, domain: Domain, y: int, x: int
    ) -> list[tuple[CellularAutomata, float]]:
        neighbors = self.__reachable_neighborhood(domain, y, x)

        # calculate transition function for each neighbor Ci=current cell and Cj=neighbor
        # neighbor cells with transition value of moving pedestrian to move from current cell to that cell
        sum_desirability = sum(
            [math.exp(neighbor.desirability) for neighbor in neighbors]
        )
        neighbors_transition = [
            (neighbor_cell, math.exp(neighbor_cell.desirability) / sum_desirability)
            for neighbor_cell in neighbors
        ]
        return neighbors_transition

    def _calculate_desirability(self, domain: Domain):
        self._calculate_attraction(domain)
        for y in range(domain.height):
            for x in range(domain.width):
                cell: CellularAutomata = domain.cells[y][x]
                if cell.state == CAState.OBSTACLE:
                    cell.desirability = 0.0
                    continue
                min_attraction = min(
                    [
                        neighbor.attraction
                        for neighbor in self.__reachable_neighborhood(domain, y, x)
                    ],
                    default=0.0,
                )
                cell.desirability = 0.00001 + cell.attraction - min_attraction

    def _calculate_attraction(self, domain: Domain):
        for y in range(domain.height):
            for x in range(domain.width):
                cell: CellularAutomata = domain.cells[y][x]
                if cell.state == CAState.OBSTACLE:
                    cell.attraction = 0.0
                    cell.crowd_repulsion = float("inf")
                    continue
                cell.crowd_repulsion = 1 / (
                    1 + len(self.__reachable_neighborhood(domain, y, x))
                )
                cell.attraction = math.exp(
                    cell.static_filed * choice(SimulationConfig.crowd.attraction_bias)
                    - cell.crowd_repulsion
                    * choice(SimulationConfig.crowd.crowd_repulsion)
                )

    def __reachable_neighborhood(
        self,
        domain: Domain,
        y: int,
        x: int,
    ) -> list[CellularAutomata]:
        """Get reachable neighborhood cells."""
        return [
            domain.cells[y + yp][x + xp]
            for yp, xp in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
            if 0 <= y + yp < domain.height
            and 0 <= x + xp < domain.width
            and (
                domain.cells[y + yp][x + xp].state != CAState.OBSTACLE
                and domain.cells[y + yp][x + xp].state != CAState.OCCUPIED
            )
        ]
