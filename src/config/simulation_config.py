import json
from pathlib import Path

from pydantic import BaseModel, Field


class CellularAutomatonParameters(BaseModel):
    kappaStatic: float
    kappaDynamic: float
    decayProbability: float
    diffusionProbability: float
    numRuns: int
    enableParallel: bool
    neighborhood: str
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


class PhysicalConstants(BaseModel):
    cellSize: float
    max_velocity: int = Field(alias="maxVlocity")


class Simulator(BaseModel):
    timeLimit: int
    cellularAutomatonParameters: CellularAutomatonParameters
    updateStrategy: str
    physicalConstants: PhysicalConstants


class SimulationConfig(BaseModel):
    numEmergencyExits: int
    omega: int
    simulator: Simulator


def load_simulation_config(path: str) -> SimulationConfig:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    json_content = config_path.read_text()
    config = SimulationConfig.model_validate_json(json_content)
    return config
