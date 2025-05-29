from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo
from typing import List, Any
from random import randint
from pydantic import BaseModel, ConfigDict
from dataclasses import dataclass
from random import shuffle
import numpy as np
from numpy.typing import NDArray

from src.config import SimulationConfig
from src.utils.measure_time import timing_decorator
from .cellular_automata import CellularAutomata
from .ca_state import CAState
from .access import Access
from .obstacle import Obstacle


@dataclass
class Pedestrian:
    """Individual pedestrian with unique identifier."""

    id: int
    x: int
    y: int
    steps_taken: int = 0

    def move_to(self, new_x: int, new_y: int) -> None:
        """Move pedestrian to new position."""
        self.x = new_x
        self.y = new_y
        self.steps_taken += 1


class Domain(BaseModel):
    id: int
    width: int
    height: int
    accesses: list[Access]
    obstacles: list[Obstacle]

    peds: NDArray[np.int_] | None = None
    walls: NDArray[np.int_] | None = None

    @property
    @timing_decorator
    def cells(self) -> List[List[CellularAutomata]]:
        grid = [
            [
                CellularAutomata(
                    state=self._get_state(row_idx, col_idx), y=row_idx, x=col_idx
                )
                for col_idx in range(self.width)
            ]
            for row_idx in range(self.height)
        ]
        return grid

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        print(f"Initializing Domain {self.id} with width={self.width}, height={self.height}, ")
        self._init_walls()
        self._initialize_cell_states()
        self._initialize_pedestrians()

    def _get_perimeter_coordinates(self, perimeter):
        width = (
            perimeter
            if perimeter <= self.width
            else self.width
            if self.width < perimeter and perimeter <= self.width + self.height
            else (self.width) - (perimeter - (self.width + self.height))
            if self.width + self.height < perimeter <= (2 * self.width + self.height)
            else 0
        )
        height = (
            0
            if perimeter <= self.width
            else (perimeter - self.width)
            if self.width < perimeter <= self.width + self.height
            else self.height
            if self.width + self.height < perimeter <= (2 * self.width + self.height)
            else (self.height) - (perimeter - (2 * self.width + self.height))
        )
        return height - 1, width - 1

    @timing_decorator
    def _initialize_cell_states(
        self,
    ):
        for access in self.accesses:
            pa, wa = access.pa, access.wa  # Pα, Wα
            for i in range(pa, pa + wa):
                height, width = self._get_perimeter_coordinates(i)
                self.walls[width][height] = 1
        for obstacle in self.obstacles:
            # Fill the grid with the obstacle state
            for y in range(
                obstacle.shape.topLeft.y,
                obstacle.shape.topLeft.y + obstacle.shape.height,
            ):
                for x in range(
                    obstacle.shape.topLeft.x,
                    obstacle.shape.topLeft.x + obstacle.shape.width,
                ):
                    self.walls[x][y] = -1

    @timing_decorator
    def _initialize_pedestrians(self):
        self.peds = np.zeros((self.width, self.height), dtype=np.int_)

        empty_cells: List[tuple[int, int]] = []
        for y in range(0, self.height):
            for x in range(0, self.width):
                if self._get_state(y, x) == CAState.EMPTY:
                    empty_cells.append((y, x))
        shuffle(empty_cells)
        num_pedestrians = randint(
            SimulationConfig.num_pedestrians[0],
            SimulationConfig.num_pedestrians[1],
        )
        selected_pedestrians = empty_cells[:num_pedestrians]
        print(
            f"create {num_pedestrians} pedestrians at {len(selected_pedestrians)} selected positions"
        )
        for y, x in selected_pedestrians:
            self.peds[x][y] = 1  # Mark the cell as occupied
            print(f"Adding pedestrian at (x={x}, y={y})")

    @field_validator("peds", "walls", mode="before")
    @classmethod
    def convert_to_numpy_array_if_needed(cls, value: Any) -> Any:
        if not isinstance(value, np.ndarray):
            try:
                # Attempt to convert list of lists or similar to a NumPy array
                converted_value = np.array(value)
                return converted_value
            except Exception as e:
                raise ValueError(f"Could not convert value to NumPy array: {e}")
        return value

    @field_validator("peds", "walls", mode="after")
    @classmethod
    def check_is_2d(cls, value: NDArray, info: ValidationInfo) -> NDArray:
        if value.ndim != 2:
            raise ValueError(
                f"Field '{info.field_name}' must be a 2D array, got {value.ndim} dimensions."
            )
        return value

    def _init_obstacles(self):
        return np.ones((self.width, self.height), int)

    @timing_decorator
    def _init_walls(
        self,
    ):
        """
        define where are the walls. Consider the exits
        """
        OBST = self._init_obstacles()
        OBST[0, :] = OBST[-1, :] = OBST[:, -1] = OBST[:, 0] = -1
        self.walls = OBST

    def _get_state(self, y, x) -> CAState:
        if self.walls[x][y] == -1:
            return CAState.OBSTACLE
        elif self.walls[x][y] == 1:
            if self.peds[x][y] == 1:
                return CAState.OCCUPIED
            else:
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    return CAState.ACCESS
                else:
                    return CAState.EMPTY

    def get_left_pedestrians(self) -> List[Pedestrian]:
        pedestrians = 0
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.peds[x][y] == 1:
                    pedestrians += 1
        return pedestrians

    def get_exit_cells(self) -> List[tuple[int, int]]:
        exit_cells = []
        for y in range(0, self.height):
            if self.walls[0][y] == 1:
                exit_cells.append((0, y))
            if self.walls[-1][y] == 1:
                exit_cells.append((self.width - 1, y))
        for x in range(0, self.width):
            if self.walls[x][0] == 1:
                exit_cells.append((x, 0))
            if self.walls[x][-1] == 1:
                exit_cells.append((x, self.height - 1))
        return exit_cells
