from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing import List, Any

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
        self._storage_cells = self._initialize_grid_cells()
        self._initialize_grid_cells()

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
        return height, width


    def _initialize_cell_states(
        self,
        accesses: list[Access],  # Pα, Wα
        obstacles: list[Obstacle],
    ):
        for access in accesses:
            pa, wa = access.shape.pa, access.shape.wa  # Pα, Wα
            for i in range(pa, pa + wa):
                height, width = self._get_perimeter_coordinates(i)
                self.cells[height][width].state = CAState.ACCESS
        for obstacle in obstacles:
            for y in range(obstacle.shape.topLeft.x, obstacle.shape.topLeft.y + obstacle.shape.height):
                for x in range(obstacle.shape.topLeft.x, obstacle.shape.topLeft.y + obstacle.shape.width):
                    self.cells[y][x].state = CAState.OBSTACLE


    @property
    def cells(self) -> List[List[CellularAutomata]]:
        return self._storage_cells

    model_config = ConfigDict(
        validate_assignment=True # Good practice for models you might modify post-init.
    )

