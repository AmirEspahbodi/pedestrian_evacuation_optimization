from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing import List, Tuple, Any
import heapq
import math
from random import choices

from src.config import SimulationConfig
from .cellular_automata import CellularAutomata
from .ca_state import CAState
from .access import Access
from .obstacle import Obstacle


class Domain(BaseModel):
    width: int
    height: int
    accesses: list[Access]
    obstacles: list[Obstacle]

    _storage_cells: List[List[CellularAutomata]] = PrivateAttr()

    @property
    def cells(self) -> List[List[CellularAutomata]]:
        return self._storage_cells

    model_config = ConfigDict(validate_assignment=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        print(f"Initializing Domain with width={self.width}, height={self.height}, ")
        self._storage_cells = self._initialize_grid_cells()
        self._initialize_cell_states()
        self._initialize_pedestrians()
        self._calculate_shortest_path_dijkstra()
        self._calculate_static_field()

    def _initialize_grid_cells(self) -> List[List[CellularAutomata]]:
        grid = [
            [
                CellularAutomata(state=CAState.EMPTY, y=row_idx, x=col_idx)
                for col_idx in range(self.width)
            ]
            for row_idx in range(self.height)
        ]
        return grid

    def _initialize_pedestrians(self):
        empty_cells: list[tuple[int, int]] = []
        for y in range(0, len(self._storage_cells)):
            for x in range(0, len(self._storage_cells[y])):
                if self._storage_cells[y][x].state == CAState.EMPTY:
                    empty_cells.append((y, x))
        num_pedestrians = choices(SimulationConfig.crowd.num_pedestrians, k=1)[0]
        selected_pedestrians = choices(empty_cells, k=num_pedestrians)
        for y, x in selected_pedestrians:
            self._storage_cells[y][x].state = CAState.OCCUPIED
            print(f"Pedestrian initialized at ({y}, {x})")

    def _get_perimeter_coordinates(self, perimeter):
        width = (
            perimeter
            if perimeter < self.width
            else self.width
            if self.width < perimeter and perimeter < self.width + self.height
            else (self.width) - (perimeter - (self.width + self.height))
            if self.width + self.height < perimeter < (2 * self.width + self.height)
            else 0
        )
        height = (
            0
            if perimeter < self.width
            else (perimeter - self.width)
            if self.width < perimeter < self.width + self.height
            else self.height
            if self.width + self.height < perimeter < (2 * self.width + self.height)
            else (self.height) - (perimeter - (2 * self.width + self.height))
        )
        return height - 1, width - 1

    def _initialize_cell_states(
        self,
    ):
        for access in self.accesses:
            pa, wa = access.shape.pa, access.shape.wa  # Pα, Wα
            for i in range(pa, pa + wa):
                height, width = self._get_perimeter_coordinates(i)
                self._storage_cells[height][width].state = CAState.ACCESS
        for obstacle in self.obstacles:
            print(
                f"Obstacle at {obstacle.shape.topLeft.x}, {obstacle.shape.topLeft.y} with width={obstacle.shape.width} and height={obstacle.shape.height}"
            )
            # Fill the grid with the obstacle state
            for y in range(
                obstacle.shape.topLeft.y,
                obstacle.shape.topLeft.y + obstacle.shape.height,
            ):
                for x in range(
                    obstacle.shape.topLeft.x,
                    obstacle.shape.topLeft.x + obstacle.shape.width,
                ):
                    self._storage_cells[y][x].state = CAState.OBSTACLE

    def _get_neighbors(self, y: int, x: int) -> List[Tuple[int, int, float]]:
        neighbors = []

        # 8-directional movement
        directions = [
            (-1, -1, math.sqrt(2)),
            (-1, 0, 1.0),
            (-1, 1, math.sqrt(2)),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (1, -1, math.sqrt(2)),
            (1, 0, 1.0),
            (1, 1, math.sqrt(2)),
        ]

        for dy, dx, dist in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                neighbors.append((ny, nx, dist))

        return neighbors

    def _calculate_shortest_path_dijkstra(self) -> None:
        pq = []

        distances = {}

        for y in range(self.height):
            for x in range(self.width):
                cell = self._storage_cells[y][x]
                if cell.state == CAState.ACCESS:
                    heapq.heappush(pq, (0.0, y, x))
                    distances[(y, x)] = 0.0
                    cell.static_filed = 0.0

        # dijkstra algorithm
        while pq:
            current_dist, y, x = heapq.heappop(pq)

            # Skip if we've already found a shorter path
            if (y, x) in distances and current_dist > distances[(y, x)]:
                continue

            # Check all neighbors
            for ny, nx, edge_dist in self._get_neighbors(y, x):
                neighbor_cell = self._storage_cells[ny][nx]

                # Cannot pass through OBSTACLE cells
                if neighbor_cell.state == CAState.OBSTACLE:
                    continue

                # Calculate new distance
                new_dist = current_dist + edge_dist

                # Update if this is a shorter path
                if (ny, nx) not in distances or new_dist < distances[(ny, nx)]:
                    distances[(ny, nx)] = new_dist
                    neighbor_cell.static_filed = new_dist
                    heapq.heappush(pq, (new_dist, ny, nx))

        # Set infinite distance for unreachable cells (OBSTACLE cells and cells blocked by OBSTACLE cells)
        for y in range(self.height):
            for x in range(self.width):
                cell = self._storage_cells[y][x]
                if (y, x) not in distances:
                    cell.static_filed = float("inf")

    def _calculate_static_field(self) -> None:
        max_shortest_path = max(
            cell.static_filed
            for row in self._storage_cells
            for cell in row
            if cell.state != CAState.OBSTACLE
        )
        print(f"Max shortest path: {max_shortest_path}")
        for y in range(self.height):
            for x in range(self.width):
                cell = self._storage_cells[y][x]
                if cell.state == CAState.OBSTACLE:
                    continue
                cell.static_filed = 1 - (cell.static_filed / max_shortest_path)

    def get_pedestrians(self) -> list[CellularAutomata]:
        """Get all pedestrians in the environment."""
        pedestrians = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[y][x]
                if cell.state == "occupied":
                    pedestrians.append(cell)
        return pedestrians
