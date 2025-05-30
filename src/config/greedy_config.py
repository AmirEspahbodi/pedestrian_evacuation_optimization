from typing import List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from .config_reader import ConfigurationReader


class Initialization(BaseModel):
    name: str  # Or use an Enum like InitializationName above
    parameters: Optional[List[str]] = (
        None  # Making parameters optional if they might not always exist
    )


class Selection(BaseModel):
    name: str
    parameters: Optional[List[str]] = (
        None  # Assuming parameters could exist, like in initialization
    )


class VariationComponent(BaseModel):
    name: str
    parameters: Optional[List[str]] = None


class Replacement(BaseModel):
    name: str
    parameters: Optional[List[str]] = None  # Assuming parameters could exist


class Island(BaseModel):
    numislands: int
    popsize: int
    offspring: int
    maxevals: int
    initialization: Initialization
    selection: Selection
    variation: List[VariationComponent]  # This is a list of variation components
    replacement: Replacement


class ExperimentConfig(BaseModel):
    numruns: int
    seed: int
    islands: List[Island]


GreedyConfig = ConfigurationReader[ExperimentConfig](
    "dataset/greedy.json", ExperimentConfig
).load_config()
