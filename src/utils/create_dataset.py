import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import argparse


class LocationType(Enum):
    SUPERMARKET = "supermarket"
    OFFICE = "office"
    MEDICAL_FACILITY = "medical_facility"
    SHOPPING_MALL = "shopping_mall"
    RESTAURANT = "restaurant"


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Shape:
    type: str
    topLeft: Point
    width: int
    height: int


@dataclass
class Obstacle:
    name: str
    shape: Shape


@dataclass
class Access:
    id: int
    pa: int
    wa: int


@dataclass
class Environment:
    id: int
    height: int
    width: int
    obstacles: List[Obstacle]
    accesses: List[Access]


class EnvironmentGenerator:
    def __init__(self, min_width: int = 30, max_width: int = 60, 
                 min_height: int = 25, max_height: int = 50):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        
        # Obstacle templates for different location types
        self.obstacle_templates = {
            LocationType.SUPERMARKET: [
                {"name_prefix": "shelf", "min_width": 8, "max_width": 15, "min_height": 2, "max_height": 4},
                {"name_prefix": "freezer", "min_width": 6, "max_width": 12, "min_height": 2, "max_height": 3},
                {"name_prefix": "checkout", "min_width": 3, "max_width": 5, "min_height": 2, "max_height": 3},
                {"name_prefix": "display", "min_width": 4, "max_width": 8, "min_height": 2, "max_height": 4},
                {"name_prefix": "column", "min_width": 2, "max_width": 3, "min_height": 2, "max_height": 3},
                {"name_prefix": "meat_counter", "min_width": 6, "max_width": 10, "min_height": 2, "max_height": 3},
                {"name_prefix": "bakery", "min_width": 5, "max_width": 8, "min_height": 3, "max_height": 4},
                {"name_prefix": "dairy", "min_width": 4, "max_width": 7, "min_height": 2, "max_height": 3},
            ],
            LocationType.OFFICE: [
                {"name_prefix": "desk", "min_width": 3, "max_width": 6, "min_height": 2, "max_height": 4},
                {"name_prefix": "conference_table", "min_width": 6, "max_width": 12, "min_height": 4, "max_height": 8},
                {"name_prefix": "filing_cabinet", "min_width": 2, "max_width": 4, "min_height": 1, "max_height": 2},
                {"name_prefix": "reception", "min_width": 4, "max_width": 8, "min_height": 3, "max_height": 5},
                {"name_prefix": "column", "min_width": 2, "max_width": 3, "min_height": 2, "max_height": 3},
                {"name_prefix": "printer_station", "min_width": 3, "max_width": 5, "min_height": 2, "max_height": 3},
                {"name_prefix": "server_rack", "min_width": 2, "max_width": 3, "min_height": 3, "max_height": 5},
            ],
            LocationType.MEDICAL_FACILITY: [
                {"name_prefix": "examination_bed", "min_width": 3, "max_width": 5, "min_height": 2, "max_height": 4},
                {"name_prefix": "equipment_cart", "min_width": 2, "max_width": 3, "min_height": 2, "max_height": 3},
                {"name_prefix": "reception_desk", "min_width": 6, "max_width": 10, "min_height": 3, "max_height": 4},
                {"name_prefix": "waiting_chairs", "min_width": 4, "max_width": 8, "min_height": 2, "max_height": 3},
                {"name_prefix": "medical_cabinet", "min_width": 2, "max_width": 4, "min_height": 1, "max_height": 2},
                {"name_prefix": "xray_machine", "min_width": 4, "max_width": 6, "min_height": 3, "max_height": 5},
            ]
        }
    
    def generate_environments(self, count: int, location_type: LocationType = LocationType.SUPERMARKET) -> List[Environment]:
        """Generate multiple environments of specified type"""
        environments = []
        for i in range(count):
            env = self._generate_single_environment(i + 1, location_type)
            environments.append(env)
        return environments
    
    def _generate_single_environment(self, env_id: int, location_type: LocationType) -> Environment:
        """Generate a single environment with realistic obstacles and exits"""
        # Generate random dimensions
        width = random.randint(self.min_width, self.max_width)
        height = random.randint(self.min_height, self.max_height)
        
        # Create obstacle grid for collision detection
        obstacle_grid = np.zeros((height, width), dtype=bool)
        
        # Generate obstacles
        obstacles = self._generate_obstacles(width, height, obstacle_grid, location_type)
        
        # Generate exits ensuring they're not blocked
        accesses = self._generate_exits(width, height, obstacle_grid)
        
        return Environment(
            id=env_id,
            height=height,
            width=width,
            obstacles=obstacles,
            accesses=accesses
        )
    
    def _generate_obstacles(self, width: int, height: int, obstacle_grid: np.ndarray, 
                          location_type: LocationType) -> List[Obstacle]:
        """Generate realistic obstacles based on location type"""
        obstacles = []
        templates = self.obstacle_templates[location_type]
        
        # Calculate target number of obstacles based on area
        area = width * height
        target_obstacles = max(30, min(80, int(area * 0.05)))  # 5% coverage roughly
        
        obstacle_count = 0
        attempts = 0
        max_attempts = target_obstacles * 10
        
        while obstacle_count < target_obstacles and attempts < max_attempts:
            attempts += 1
            
            # Choose random template
            template = random.choice(templates)
            
            # Generate obstacle dimensions
            obs_width = random.randint(template["min_width"], template["max_width"])
            obs_height = random.randint(template["min_height"], template["max_height"])
            
            # Find valid position
            position = self._find_valid_position(width, height, obs_width, obs_height, obstacle_grid)
            
            if position:
                x, y = position
                
                # Create obstacle
                obstacle = Obstacle(
                    name=f"{template['name_prefix']}_{obstacle_count + 1}",
                    shape=Shape(
                        type="rectangle",
                        topLeft=Point(x=x, y=y),
                        width=obs_width,
                        height=obs_height
                    )
                )
                
                obstacles.append(obstacle)
                
                # Mark cells as occupied
                obstacle_grid[y:y+obs_height, x:x+obs_width] = True
                obstacle_count += 1
        
        return obstacles
    
    def _find_valid_position(self, width: int, height: int, obs_width: int, obs_height: int, 
                           obstacle_grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find a valid position for an obstacle that doesn't overlap with existing ones"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Leave margin from edges to ensure pathways
            margin = 2
            x = random.randint(margin, width - obs_width - margin)
            y = random.randint(margin, height - obs_height - margin)
            
            # Check if position is free and has some surrounding space
            extended_area = obstacle_grid[
                max(0, y-1):min(height, y+obs_height+1),
                max(0, x-1):min(width, x+obs_width+1)
            ]
            
            if not np.any(extended_area):
                return (x, y)
        
        return None
    
    def _generate_exits(self, width: int, height: int, obstacle_grid: np.ndarray) -> List[Access]:
        """Generate exits on the perimeter ensuring they're not blocked"""
        accesses = []
        perimeter = 2 * (width + height)
        num_exits = random.randint(2, 5)
        
        # Calculate minimum distance between exits
        min_distance = perimeter // (num_exits + 1)
        
        exit_positions = []
        
        for i in range(num_exits):
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts:
                attempts += 1
                
                # Generate random position on perimeter
                pa = random.randint(0, perimeter - 5)
                wa = random.randint(2, 4)  # Exit width
                
                # Ensure exit doesn't exceed perimeter
                if pa + wa >= perimeter:
                    continue
                
                # Check minimum distance from other exits
                too_close = False
                for existing_pa in exit_positions:
                    distance = min(
                        abs(pa - existing_pa),
                        perimeter - abs(pa - existing_pa)
                    )
                    if distance < min_distance:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Check if exit area is clear
                if self._is_exit_clear(pa, wa, width, height, obstacle_grid):
                    accesses.append(Access(id=i, pa=pa, wa=wa))
                    exit_positions.append(pa)
                    break
        
        # Ensure at least 2 exits
        if len(accesses) < 2:
            accesses = self._generate_fallback_exits(width, height)
        
        return accesses
    
    def _is_exit_clear(self, pa: int, wa: int, width: int, height: int, 
                      obstacle_grid: np.ndarray) -> bool:
        """Check if exit area is clear of obstacles"""
        perimeter = 2 * (width + height)
        
        for offset in range(wa):
            pos = (pa + offset) % perimeter
            coords = self._perimeter_to_coords(pos, width, height)
            
            if coords:
                x, y = coords
                # Check if the exit cell and adjacent cells are clear
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        check_x, check_y = x + dx, y + dy
                        if (0 <= check_x < width and 0 <= check_y < height and
                            obstacle_grid[check_y, check_x]):
                            return False
        
        return True
    
    def _perimeter_to_coords(self, pos: int, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Convert perimeter position to x,y coordinates"""
        if pos < width:  # Top edge
            return (pos, 0)
        elif pos < width + height:  # Right edge
            return (width - 1, pos - width)
        elif pos < 2 * width + height:  # Bottom edge
            return (2 * width + height - 1 - pos, height - 1)
        elif pos < 2 * (width + height):  # Left edge
            return (0, 2 * (width + height) - 1 - pos)
        return None
    
    def _generate_fallback_exits(self, width: int, height: int) -> List[Access]:
        """Generate fallback exits when normal generation fails"""
        perimeter = 2 * (width + height)
        return [
            Access(id=0, pa=width // 4, wa=3),  # Top
            Access(id=1, pa=width + height // 2, wa=3),  # Right
        ]
    
    def save_environments_to_json(self, environments: List[Environment], filename: str):
        """Save environments to JSON file"""
        # Convert to dictionary format
        env_dicts = []
        for env in environments:
            obstacles_dict = []
            for obs in env.obstacles:
                obstacles_dict.append({
                    "name": obs.name,
                    "shape": {
                        "type": obs.shape.type,
                        "topLeft": {"x": obs.shape.topLeft.x, "y": obs.shape.topLeft.y},
                        "width": obs.shape.width,
                        "height": obs.shape.height
                    }
                })
            
            accesses_dict = []
            for acc in env.accesses:
                accesses_dict.append({
                    "id": acc.id,
                    "pa": acc.pa,
                    "wa": acc.wa
                })
            
            env_dicts.append({
                "id": env.id,
                "height": env.height,
                "width": env.width,
                "obstacles": obstacles_dict,
                "accesses": accesses_dict
            })
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(env_dicts, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully generated {len(environments)} environments and saved to {filename}")


def main():
    from src.config import SimulationConfig
    
    # Create generator
    generator = EnvironmentGenerator(
        min_width=60,
        max_width=100,
        min_height=40,
        max_height=80
    )
    
    # Map string to enum
    location_type = LocationType('supermarket')
    
    # Generate environments
    print(f"Generating {SimulationConfig.num_simulations} {'rectangle'} environments...")
    environments = generator.generate_environments(SimulationConfig.num_simulations, location_type)
    
    # Save to JSON
    generator.save_environments_to_json(environments, f"./dataset/environments/environments_{location_type.value}.json")
    
    # Print summary
    print("\nGeneration Summary:")
    for env in environments:
        print(f"Environment {env.id}: {env.width}x{env.height}, "
              f"{len(env.obstacles)} obstacles, {len(env.accesses)} exits")


if __name__ == "__main__":
    main()