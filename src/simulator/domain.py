from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo, PrivateAttr
from typing import List, Any, Tuple, Deque
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
from enum import StrEnum
from typing import List
from collections import deque
import math
from pydantic import BaseModel, ConfigDict


@dataclass
class Pedestrian:
    """Individual pedestrian with unique identifier."""

    id: int
    x: int
    y: int
    t_star: int
    is_exited: bool = False
    d_star: int | None = None


class Domain(BaseModel):
    id: int
    width: int
    height: int
    accesses: list[Access]
    obstacles: list[Obstacle]

    is_simulation_finished: bool = False

    _storage_cells: List[List[CellularAutomata]] = PrivateAttr(default_factory=list)
    _pedestrians: List[Pedestrian] = PrivateAttr(default_factory=list)

    peds: NDArray[np.int_] | None = None
    walls: NDArray[np.int_] | None = None

    @property
    @timing_decorator
    def cells(self) -> List[List[CellularAutomata]]:
        return self._storage_cells

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        print(
            f"Initializing Domain {self.id} with width={self.width}, height={self.height}, perimeter={2*(self.width+self.height)}"
        )

        self._init_cells()
        self._init_walls()
        self._initialize_cell_states()
        self._initialize_pedestrians()

    def increase_pedestrians_t_star(self):
        for ped in self._pedestrians:
            if ped.is_exited:
                continue
            ped.t_star += 1

    def move_ped(self, cell: Tuple[int, int], neighbor: Tuple[int, int]):
        current_ped = None
        for ped in self._pedestrians:
            if (ped.x, ped.y) == cell:
                current_ped = ped
                break
        else:
            raise RuntimeError(f"there is no pedestrian in cell {cell}")
        if self._storage_cells[neighbor[1]][neighbor[0]].state == CAState.EMPTY:
            self._storage_cells[current_ped.y][current_ped.x].state = CAState.EMPTY
            self._storage_cells[neighbor[1]][neighbor[0]].state = CAState.OCCUPIED
        if self._storage_cells[neighbor[1]][neighbor[0]].state == CAState.ACCESS:
            self._storage_cells[current_ped.y][current_ped.x].state = CAState.EMPTY
            self._storage_cells[neighbor[1]][
                neighbor[0]
            ].state = CAState.ACCESS_OCCUPIED
        if (
            self._storage_cells[neighbor[1]][neighbor[0]].state
            == CAState.EMERGENCY_ACCESS
        ):
            self._storage_cells[current_ped.y][current_ped.x].state = CAState.EMPTY
            self._storage_cells[neighbor[1]][
                neighbor[0]
            ].state = CAState.EMERGENCY_ACCESS_OCCUPIED

        current_ped.x = neighbor[0]
        current_ped.y = neighbor[1]

        if self.self_is_access(neighbor):
            current_ped.is_exited = True

    def self_is_access(self, cell):
        x, y = cell
        if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
            if self.walls[cell[0], cell[1]] == 1 or self.walls[cell[0], cell[1]] == 2:
                return True
        return False

    def _get_state(self, y, x) -> CAState:
        if self.walls[x][y] == -1:
            return CAState.OBSTACLE
        elif self.walls[x][y] == 1:
            if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                if self.peds[x, y] == 1:
                    return CAState.ACCESS_OCCUPIED
                else:
                    return CAState.ACCESS
            else:
                if self.peds[x, y] == 1:
                    return CAState.OCCUPIED
                else:
                    return CAState.EMPTY
        elif self.walls[x][y] == 2:
            if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                if self.peds[x, y] == 1:
                    return CAState.EMERGENCY_ACCESS_OCCUPIED
                else:
                    return CAState.EMERGENCY_ACCESS
            else:
                raise RuntimeError("incorrect emergency emaccess! ")

    def get_left_pedestrians_count(self) -> int:
        pedestrians = 0
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.peds[x][y] == 1:
                    pedestrians += 1
        return pedestrians

    def get_exit_cells(self) -> List[tuple[int, int]]:
        exit_cells = []
        for y in range(0, self.height):
            if self.walls[0][y] == 1 or self.walls[0][y] == 2:
                exit_cells.append((0, y))
            if self.walls[-1][y] == 1 or self.walls[-1][y] == 2:
                exit_cells.append((self.width - 1, y))
        for x in range(0, self.width):
            if self.walls[x][0] == 1 or self.walls[x][0] == 2:
                exit_cells.append((x, 0))
            if self.walls[x][-1] == 1 or self.walls[x][-1] == 2:
                exit_cells.append((x, self.height - 1))
        return exit_cells

    def _get_neighbors_with_distance(
        self, y: int, x: int
    ) -> List[Tuple[int, int, float]]:
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

    def test_pedestrian(self):
        count = self.get_left_pedestrians_count()
        test_count = 0
        for ped in self._pedestrians:
            if not ped.is_exited:
                test_count += 1
        print(test_count)
        print(count)
        assert count == test_count

    def _init_cells(self):
        self._storage_cells = [
            [
                CellularAutomata(state=CAState.EMPTY, y=row_idx, x=col_idx)
                for col_idx in range(self.width)
            ]
            for row_idx in range(self.height)
        ]

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
                if i>=2*(self.width-1 + self.height-1):
                    i -= 2*(self.width-1 + self.height-1)
                height, width = self._get_perimeter_point(i)
                self.walls[width][height] = 1
                self._storage_cells[height][width].state = CAState.ACCESS
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
                    self._storage_cells[y][x].state = CAState.OBSTACLE

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
        for y, x in selected_pedestrians:
            self.peds[x][y] = 1  # Mark the cell as occupied
            self._pedestrians.append(
                Pedestrian(x=x, y=y, id=np.sum(self.peds), t_star=0)
            )

    def reset_pedestrians(self):
        del self.peds
        self._pedestrians.clear()
        for y in range(self.height):
            for x in range(self.width):
                if self._storage_cells[y][x].state == CAState.OCCUPIED:
                    self._storage_cells[y][x].state == CAState.EMPTY
                elif self._storage_cells[y][x].state == CAState.ACCESS_OCCUPIED:
                    self._storage_cells[y][x].state == CAState.ACCESS
                elif (
                    self._storage_cells[y][x].state == CAState.EMERGENCY_ACCESS_OCCUPIED
                ):
                    self._storage_cells[y][x].state == CAState.EMERGENCY_ACCESS
        self._initialize_pedestrians()

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

    def _calculate_nearest_exit_distances(self):
        """
        Calculate the nearest exit distance for each non-obstacle cell using multi-source BFS.
        This algorithm efficiently finds the shortest path from each cell to any access point.

        Time Complexity: O(rows * cols)
        Space Complexity: O(rows * cols)
        """
        if not self.cells or not self.cells[0]:
            return

        # Moore neighborhood: 8 directions (including diagonals)
        directions = [
            (-1, -1, math.sqrt(2)),
            (-1, 0, 1),
            (-1, 1, math.sqrt(2)),  # Top row
            (0, -1, 1),
            (0, 1, 1),  # Middle row (excluding center)
            (1, -1, math.sqrt(2)),
            (1, 0, 1),
            (1, 1, math.sqrt(2)),  # Bottom row
        ]

        # Initialize queue for multi-source BFS
        queue: Deque[Tuple[int, int, float]] = deque()
        visited = set()

        # Reset static_field for all cells and find access cells
        for i in range(self.height):
            for j in range(self.width):
                cell = self.cells[i][j]

                # Reset distance
                cell.static_field = float("inf")

                # Add access cells as starting points for BFS
                if cell.state in [CAState.ACCESS, CAState.ACCESS_OCCUPIED]:
                    cell.static_field = 0.0
                    queue.append((i, j, 0.0))
                    visited.add((i, j))

        # Multi-source BFS to find shortest distances
        while queue:
            row, col, current_distance = queue.popleft()

            # Skip if we've found a better path to this cell
            if current_distance > self.cells[row][col].static_field:
                continue

            # Explore all 8 neighbors (Moore neighborhood)
            for dr, dc, distance in directions:
                new_row, new_col = row + dr, col + dc

                # Check bounds
                if not (0 <= new_row < self.height and 0 <= new_col < self.width):
                    continue

                neighbor = self.cells[new_row][new_col]

                # Skip obstacles (pedestrians can't walk through them)
                if neighbor.state == CAState.OBSTACLE:
                    continue

                # Calculate distance (1 unit for all moves in Moore neighborhood)
                new_distance = current_distance + distance

                # Update if we found a shorter path
                if new_distance < neighbor.static_field:
                    neighbor.static_field = new_distance

                    # Only add to queue if not visited or if we found a better path
                    if (
                        new_row,
                        new_col,
                    ) not in visited or new_distance < neighbor.static_field:
                        queue.append((new_row, new_col, new_distance))
                        visited.add((new_row, new_col))

    def calculate_peds_distance_to_nearest_exit(self):
        self._calculate_nearest_exit_distances()
        for ped in self._pedestrians:
            ped.d_star = self._storage_cells[ped.y][ped.x].static_field

    def get_pedestrians(self):
        return self._pedestrians
    
    def get_pedestrians_np_count(self):
        return np.sum(self.peds)

    def add_emergency_accesses(self, emergency_accesses: list[tuple[int, int]]):
        for access in emergency_accesses:
            try:
                pa, wa = access
                for i in range(pa, pa + wa):
                    if i>=2*(self.width-1 + self.height-1):
                        i -= 2*(self.width-1 + self.height-1)
                    height, width = self._get_perimeter_point(i)
                    if self.walls[width][height] == -1:
                        self.walls[width][height] = 2
                        self._storage_cells[height][
                            width
                        ].state = CAState.EMERGENCY_ACCESS
            except BaseException as e:
                print(f"error {e}")
                print(emergency_accesses)

    def remove_emergency_accesses(self):
        self.walls[0, :] = self.walls[-1, :] = self.walls[:, -1] = self.walls[:, 0] = -1
        for access in self.accesses:
            pa, wa = access.pa, access.wa  # Pα, Wα
            for i in range(pa, pa + wa):
                if i>=2*(self.width-1 + self.height-1):
                    i -= 2*(self.width-1 + self.height-1)
                height, width = self._get_perimeter_point(i)
                self.walls[width][height] = 1
                self._storage_cells[height][width].state = CAState.ACCESS



    def _get_perimeter_point(self, p):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("self.Width and self.height must be positive")
        
        if self.width == 1 and self.height == 1:
            if p != 0:
                raise ValueError("For 1x1 area, only point 0 is valid")
            return (0, 0)
        
        perimeter_length = 2 * (self.width - 1 + self.height - 1)
        
        if p < 0 or p >= perimeter_length:
            raise ValueError(f"Point p must be between 0 and {perimeter_length - 1}")
        
        if self.width == 1:
            if p < self.height:
                return (p, 0)
            else:
                return (self.height - 1 - (p - self.height), 0)
        
        if self.height == 1:
            if p < self.width:
                return (0, p)
            else:
                return (0, self.width - 1 - (p - self.width))
        
        if p < self.width:
            return (0, p)
        
        p -= self.width
        
        if p < self.height - 1:
            return (p + 1, self.width - 1)
        
        p -= (self.height - 1)
        
        if p < self.width - 1:
            return (self.height - 1, self.width - 2 - p)
        
        p -= (self.width - 1)
        
        return (self.height - 2 - p, 0)

    def test_emergency_accesses(self, emergency_exit: Tuple[int, int], real_corrdinates: set[Tuple[int, int]]):
        pa, wa = emergency_exit  # Pα, Wα
        calculated_corrdinates = set()
        for i in range(pa, pa + wa):
            if i>=2*(self.width-1 + self.height-1):
                i -= 2*(self.width-1 + self.height-1)
            height, width = self._get_perimeter_point(self.width, self.height, i)
            calculated_corrdinates.add((height, width))
        print(calculated_corrdinates)
        assert calculated_corrdinates == real_corrdinates 
