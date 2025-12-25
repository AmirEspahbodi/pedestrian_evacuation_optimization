"""
Grid to Vector JSON Converter
Parses 2D grid files and converts them to structured JSON for simulation environments.
"""

import json
import random
import string
from collections import deque
from typing import Dict, List, Set, Tuple


class GridVectorizer:
    """Converts 2D grid data into vectorized JSON format."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.grid = []
        self.height = 0
        self.width = 0
        self.visited = set()

    def load_grid(self):
        """Load grid from file using specified logic."""
        with open(self.file_path, "r") as f:
            self.grid = [[int(c) for c in line.strip()] for line in f]
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0
        print(f"Loaded grid: {self.width}x{self.height}")

    def extract_walls(self) -> List[Dict]:
        """Extract and vectorize walls (value=1) into rectangles."""
        walls = []
        visited = [[False] * self.width for _ in range(self.height)]
        wall_id = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 1 and not visited[y][x]:
                    # Find maximal rectangle starting from this point
                    rect = self._find_maximal_rectangle(x, y, visited, 1)
                    if rect:
                        top_x, top_y, w, h = rect

                        # Determine orientation and length
                        if w > h:
                            orientation = "Horizontal"
                            length = w
                        elif h > w:
                            orientation = "Vertical"
                            length = h
                        else:
                            # Equal dimensions - default to horizontal
                            orientation = "Horizontal"
                            length = w

                        walls.append(
                            {
                                "name": f"wall_{wall_id}",
                                "shape": {
                                    "type": "rectangle",
                                    "topLeft": {"x": top_x, "y": top_y},
                                    "orientation": orientation,
                                    "length": length,
                                },
                            }
                        )
                        wall_id += 1

        return walls

    def _find_maximal_rectangle(
        self, start_x: int, start_y: int, visited: List[List[bool]], target_value: int
    ) -> Tuple:
        """Find the largest rectangle of target_value starting from (start_x, start_y)."""
        if visited[start_y][start_x] or self.grid[start_y][start_x] != target_value:
            return None

        # Try to extend horizontally first
        max_width = 0
        for x in range(start_x, self.width):
            if self.grid[start_y][x] == target_value and not visited[start_y][x]:
                max_width += 1
            else:
                break

        # Try to extend vertically
        max_height = 1
        for y in range(start_y + 1, self.height):
            # Check if entire row can be extended
            can_extend = True
            for x in range(start_x, start_x + max_width):
                if x >= self.width or self.grid[y][x] != target_value or visited[y][x]:
                    can_extend = False
                    break

            if can_extend:
                max_height += 1
            else:
                break

        # Mark all cells in rectangle as visited
        for y in range(start_y, start_y + max_height):
            for x in range(start_x, start_x + max_width):
                visited[y][x] = True

        return (start_x, start_y, max_width, max_height)

    def extract_obstacles(self) -> List[Dict]:
        """Extract obstacles (value=2) as clustered bounding boxes."""
        obstacles = []
        visited = [[False] * self.width for _ in range(self.height)]
        obs_id = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 2 and not visited[y][x]:
                    # Find connected component using BFS
                    component = self._find_connected_component(x, y, visited, 2)
                    if component:
                        # Calculate bounding box
                        min_x = min(coord[0] for coord in component)
                        max_x = max(coord[0] for coord in component)
                        min_y = min(coord[1] for coord in component)
                        max_y = max(coord[1] for coord in component)

                        width = max_x - min_x + 1
                        height = max_y - min_y + 1

                        # Generate random string for name
                        rand_str = "".join(
                            random.choices(string.ascii_lowercase + string.digits, k=4)
                        )

                        obstacles.append(
                            {
                                "name": f"obs_{rand_str}_{obs_id}",
                                "shape": {
                                    "type": "rectangle",
                                    "topLeft": {"x": min_x, "y": min_y},
                                    "width": width,
                                    "height": height,
                                },
                            }
                        )
                        obs_id += 1

        return obstacles

    def _find_connected_component(
        self, start_x: int, start_y: int, visited: List[List[bool]], target_value: int
    ) -> List[Tuple]:
        """Find all connected cells with target_value using BFS."""
        component = []
        queue = deque([(start_x, start_y)])
        visited[start_y][start_x] = True

        while queue:
            x, y = queue.popleft()
            component.append((x, y))

            # Check 4-connected neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and not visited[ny][nx]
                    and self.grid[ny][nx] == target_value
                ):
                    visited[ny][nx] = True
                    queue.append((nx, ny))

        return component

    def extract_pedestrians(self) -> List[Dict]:
        """Extract pedestrian positions (value=3)."""
        pedestrians = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 3:
                    pedestrians.append({"x": x, "y": y})
        return pedestrians

    def extract_access_points(self) -> List[Dict]:
        """Extract access points (value=4) from perimeter with perimeter address."""
        accesses = []
        access_id = 0

        # Find all perimeter cells with value 4
        perimeter_cells = []

        # Top edge (y=0)
        for x in range(self.width):
            if self.grid[0][x] == 4:
                perimeter_cells.append((x, 0, "top"))

        # Right edge (x=width-1)
        for y in range(self.height):
            if self.grid[y][self.width - 1] == 4:
                perimeter_cells.append((self.width - 1, y, "right"))

        # Bottom edge (y=height-1)
        if self.height > 1:
            for x in range(self.width):
                if self.grid[self.height - 1][x] == 4:
                    perimeter_cells.append((x, self.height - 1, "bottom"))

        # Left edge (x=0)
        if self.width > 1:
            for y in range(self.height):
                if self.grid[y][0] == 4:
                    perimeter_cells.append((0, y, "left"))

        # Group contiguous access points and calculate pa/wa
        visited_perimeter = set()

        for cell in perimeter_cells:
            x, y, edge = cell
            if (x, y) in visited_perimeter:
                continue

            # Find contiguous group on this edge
            group = self._find_contiguous_access(x, y, edge, visited_perimeter)

            if group:
                # Calculate perimeter address (pa) and width (wa)
                pa, wa = self._calculate_perimeter_address(group, edge)

                accesses.append({"id": access_id, "pa": pa, "wa": wa})
                access_id += 1

        return accesses

    def _find_contiguous_access(
        self, start_x: int, start_y: int, edge: str, visited: Set
    ) -> List[Tuple]:
        """Find contiguous access points on the same edge."""
        group = []

        if edge == "top":
            x = start_x
            while x < self.width and self.grid[0][x] == 4 and (x, 0) not in visited:
                group.append((x, 0))
                visited.add((x, 0))
                x += 1

        elif edge == "right":
            y = start_y
            while (
                y < self.height
                and self.grid[y][self.width - 1] == 4
                and (self.width - 1, y) not in visited
            ):
                group.append((self.width - 1, y))
                visited.add((self.width - 1, y))
                y += 1

        elif edge == "bottom":
            x = start_x
            while (
                x >= 0
                and self.grid[self.height - 1][x] == 4
                and (x, self.height - 1) not in visited
            ):
                group.append((x, self.height - 1))
                visited.add((x, self.height - 1))
                x -= 1

        elif edge == "left":
            y = start_y
            while y >= 0 and self.grid[y][0] == 4 and (0, y) not in visited:
                group.append((0, y))
                visited.add((0, y))
                y -= 1

        return group

    def _calculate_perimeter_address(
        self, group: List[Tuple], edge: str
    ) -> Tuple[int, int]:
        """Calculate perimeter address (pa) and width (wa) for access group."""
        if not group:
            return 0, 0

        wa = len(group)

        # Get the starting position of the group
        x, y = group[0]

        W = self.width
        H = self.height

        # Calculate pa based on edge and position
        if edge == "top":  # y = 0
            pa = x
        elif edge == "right":  # x = W - 1
            pa = W + y
        elif edge == "bottom":  # y = H - 1
            pa = W + H + (W - 1 - x)
        elif edge == "left":  # x = 0
            pa = 2 * W + H + (H - 1 - y)
        else:
            pa = 0

        return pa, wa

    def convert_to_json(self) -> Dict:
        """Convert grid to complete JSON structure."""
        print("Extracting walls...")
        walls = self.extract_walls()

        print("Extracting obstacles...")
        obstacles = self.extract_obstacles()

        print("Extracting pedestrians from first file...")
        pedestrians = self.extract_pedestrians()

        print("Extracting access points...")
        accesses = self.extract_access_points()

        result = {
            "domains": [
                {
                    "id": 1,
                    "height": self.height,
                    "width": self.width,
                    "walls": walls,
                    "obstacles": obstacles,
                    "accesses": accesses,
                    "pedestrians_1": pedestrians,
                }
            ]
        }

        print(f"Conversion complete:")
        print(f"  - Walls: {len(walls)}")
        print(f"  - Obstacles: {len(obstacles)}")
        print(f"  - Access Points: {len(accesses)}")
        print(f"  - Pedestrians (file 1): {len(pedestrians)}")

        return result

    def add_pedestrians_from_file(self, file_path: str, file_number: int, result: Dict):
        """Add pedestrians from an additional grid file."""
        print(f"\nProcessing pedestrians from file {file_number}...")

        # Load the grid
        with open(file_path, "r") as f:
            grid = [[int(c) for c in line.strip()] for line in f]

        # Extract pedestrians
        pedestrians = []
        for y in range(len(grid)):
            for x in range(len(grid[0]) if len(grid) > 0 else 0):
                if grid[y][x] == 3:
                    pedestrians.append({"x": x, "y": y})

        # Add to result
        result["domains"][0][f"pedestrians_{file_number}"] = pedestrians
        print(f"  - Pedestrians (file {file_number}): {len(pedestrians)}")

    def save_json(self, output_path: str):
        """Save converted data to JSON file."""
        data = self.convert_to_json()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to: {output_path}")

    def process_multiple_grids(self, base_name: str, num_files: int, output_path: str):
        """Process multiple grid files with same structure but different pedestrians."""
        # Load first file and extract all data
        first_file = f"{base_name}_1.txt"
        self.file_path = first_file
        self.load_grid()

        # Get complete JSON with pedestrians_1
        result = self.convert_to_json()

        # Process remaining files for pedestrians only
        for i in range(2, num_files + 1):
            file_path = f"{base_name}_{i}.txt"
            self.add_pedestrians_from_file(file_path, i, result)

        # Save final JSON
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ All files processed and saved to: {output_path}")


def main():
    """Main execution function."""
    # Configuration
    base_name = "finall_grid_400"  # Base name without number and extension
    num_files = 10  # Number of grid files to process
    output_file = "output_simulation.json"  # Output JSON file

    # Process all grid files
    vectorizer = GridVectorizer("")
    vectorizer.process_multiple_grids(base_name, num_files, output_file)


if __name__ == "__main__":
    main()
