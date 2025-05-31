from typing import List, Any, Optional
from pydantic import BaseModel
from .config_reader import ConfigurationReader


class TopologyConfig(BaseModel):
    name: str
    parameters: List[
        Any
    ]  # Parameters can be of mixed types or you might want to make it more specific if known


class InitializationConfig(BaseModel):
    name: str


class OperatorConfig(
    BaseModel
):  # A general class for operations like selection, replacement, send, receive
    name: str
    parameters: Optional[List[str]] = None  # Parameters are optional for some operators


class VariationOperatorConfig(BaseModel):
    name: str
    parameters: List[str]


class MigrationConfig(BaseModel):
    frequency: int
    individuals: int
    send: OperatorConfig
    receive: OperatorConfig


class IslandConfig(BaseModel):
    numislands: int
    popsize: int
    offspring: int
    maxevals: int
    recombination_prob: float
    mutation_gamma: float
    migration_frequency_generations: int
    initialization: InitializationConfig
    selection: OperatorConfig
    variation: List[VariationOperatorConfig]
    replacement: OperatorConfig
    migration: MigrationConfig


class ExperimentConfig(BaseModel):
    numruns: int
    seed: int
    topology: TopologyConfig
    islands: List[IslandConfig]


IEAConfig = ConfigurationReader[ExperimentConfig](
    "dataset/iea.json", ExperimentConfig
).load_config()
