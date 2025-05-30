from typing import List
from pydantic import BaseModel
from .config_reader import ConfigurationReader


class InitializationConfig(BaseModel):
    name: str


class SelectionConfig(BaseModel):
    name: str
    parameters: List[str] = []  # Default to empty list if not present


class VariationConfig(BaseModel):
    name: str
    parameters: List[str] = []  # Default to empty list if not present


class ReplacementConfig(BaseModel):
    name: str


class IslandConfig(BaseModel):
    numislands: int
    popsize: int
    offspring: int
    maxevals: int
    initialization: InitializationConfig
    selection: SelectionConfig
    variation: List[VariationConfig]
    replacement: ReplacementConfig


class ExperimentConfig(BaseModel):
    numruns: int
    seed: int
    islands: List[IslandConfig]


EAConfig = ConfigurationReader[ExperimentConfig](
    "dataset/ea.json", ExperimentConfig
).load_config()
