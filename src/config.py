from pathlib import Path
from typing import List, Literal
import json
import os
from functools import lru_cache
from threading import Lock


from pydantic import BaseModel, Field, field_validator


class CellularAutomatonParameters(BaseModel):
    """Parameters for cellular automaton simulation."""

    cell_dimension: float = Field(alias="cellDimension", gt=0)
    neighborhood: Literal["Moore", "VonNeumann"] = Field(alias="neighborhood")
    floor_field: str = Field(alias="floorField")

    class Config:
        populate_by_name = True


class Simulator(BaseModel):
    """Simulator configuration settings."""

    time_limit: int = Field(alias="timeLimit", gt=0)
    simulator_type: str = Field(alias="simulatorType")
    cellular_automaton_parameters: CellularAutomatonParameters = Field(
        alias="cellularAutomatonParameters"
    )

    class Config:
        populate_by_name = True


class Crowd(BaseModel):
    """Crowd dynamics configuration."""

    num_pedestrians: List[int] = Field(alias="numPedestrians", min_length=1)
    pedestrian_reference_velocity: float = Field(
        alias="pedestrianReferenceVelocity", gt=0
    )
    attraction_bias: List[float] = Field(alias="attractionBias", min_length=1)
    crowd_repulsion: List[float] = Field(alias="crowdRepulsion", min_length=1)
    velocity_factor: List[float] = Field(alias="velocityFactor", min_length=1)

    @field_validator("num_pedestrians")
    @classmethod
    def validate_pedestrians(cls, v):
        if any(p < 0 for p in v):
            raise ValueError("Number of pedestrians must be non-negative")
        return v

    @field_validator("attraction_bias", "crowd_repulsion", "velocity_factor")
    @classmethod
    def validate_positive_values(cls, v):
        if any(val < 0 for val in v):
            raise ValueError("Values must be non-negative")
        return v

    class Config:
        populate_by_name = True


class EnvironmentConfig(BaseModel):
    """Root configuration model for simulation environment."""

    num_simulations: int = Field(alias="numSimulations", gt=0)
    simulator: Simulator
    crowd: Crowd

    class Config:
        populate_by_name = True


class ConfigurationReader:
    """Handles reading and validation of environment configuration files."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self._config = None

    @property
    def config(self) -> EnvironmentConfig:
        """Lazy loading of configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def load_config(self) -> EnvironmentConfig:
        """Load and validate configuration from JSON file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.file_path}")

        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
            return EnvironmentConfig(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")


_reader = ConfigurationReader("dataset/simulation.json")
SimulationConfig = _reader.config
