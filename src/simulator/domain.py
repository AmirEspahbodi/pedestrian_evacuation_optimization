from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing import List, Any
from random import Random, choices

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

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        print(f"Initializing Domain with width={self.width}, height={self.height}, ")
        self._storage_cells = self._initialize_grid_cells()
        self._initialize_cell_states()
        self._initialize_pedestrians()

    def _initialize_pedestrians(self):
        empty_cells: list[tuple[int, int]] = []
        for y in range(0, len(self._storage_cells)):
            for x in range(0, len(self._storage_cells[y])):
                if self._storage_cells[y][x].state == CAState.EMPTY:
                    empty_cells.append((y, x))
        num_pedestrians = choices(SimulationConfig.crowd.num_pedestrians, k=1)[0]
        selected_pedestrians = choices(
            empty_cells, k=num_pedestrians
        )
        for y, x in selected_pedestrians:
            self._storage_cells[y][x].state = CAState.OCCUPIED
            print(f"Pedestrian initialized at ({y}, {x})")

    def _initialize_grid_cells(self) -> List[List[CellularAutomata]]:
        grid = [
            [
                CellularAutomata(state=CAState.EMPTY, y=row_idx, x=col_idx)
                for col_idx in range(self.width)
            ]
            for row_idx in range(self.height)
        ]
        return grid

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

    @property
    def cells(self) -> List[List[CellularAutomata]]:
        return self._storage_cells

    model_config = ConfigDict(
        validate_assignment=True  # Good practice for models you might modify post-init.
    )
