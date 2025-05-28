from pydantic import BaseModel, Field, field_validator

from pathlib import Path
from typing import List
import json
from functools import cached_property
from .types import UpdateStrategy, Neighborhood


class CellularAutomatonParameters(BaseModel):
    """Parameters for cellular automaton simulation."""

    kappa_static: float = Field(alias="kappaStatic")
    kappa_dynamic: float = Field(alias="kappaDynamic")
    decay_probability: float = Field(alias="decayProbability")
    diffusion_probability: float = Field(alias="diffusionProbability")
    num_runs: int = Field(alias="numRuns")
    enable_parallel: bool = Field(alias="enableParallel")
    neighborhood: Neighborhood = Field(alias="neighborhood")
    diffusion: float
    plotS: bool
    plotD: bool
    plotAvgD: bool
    plotP: bool
    shuffle: bool
    reverse: bool
    log: str
    decay: float
    clean: bool
    parallel: bool

    class Config:
        populate_by_name = True


class PhysicalConstants(BaseModel):
    """Physical constants for the simulation."""

    cell_size: float = Field(alias="cellSize")  # meters
    max_vlocity: float = Field(alias="maxVlocity")  # m/s

    @cached_property
    def time_step(self) -> float:
        """Calculate time step based on physical constraints."""
        return self.cell_size / self.max_vlocity

    class Config:
        populate_by_name = True


class Simulator(BaseModel):
    """Simulator configuration settings."""

    time_limit: int = Field(alias="timeLimit", gt=0)
    update_strategy: UpdateStrategy = Field(alias="updateStrategy")
    cellular_automaton_parameters: CellularAutomatonParameters = Field(
        alias="cellularAutomatonParameters"
    )
    physical_constants: PhysicalConstants = Field(alias="physicalConstants")

    class Config:
        populate_by_name = True


class EnvironmentConfig(BaseModel):
    """Root configuration model for simulation environment."""

    num_simulations: int = Field(alias="numSimulations", gt=0)
    simulator: Simulator
    num_pedestrians: List[int] = Field(alias="numPedestrians", min_length=1)

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
