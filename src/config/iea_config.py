import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


# 1. Define the Pydantic Model
class EvolutionaryConfig(BaseModel):
    numruns: int
    seed: int
    numislands: int = Field(..., gt=0, description="Number of islands must be positive")
    popsize: int
    offspring: int
    recombination_prob: float = Field(
        ..., ge=0.0, le=1.0, description="Probability must be between 0 and 1"
    )
    mutation_gamma: float
    migration_frequency_generations: int


def load_iea_config(file_path: str) -> EvolutionaryConfig:
    """Reads a JSON file and validates it against the EvolutionaryConfig model."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path.absolute()}")

    try:
        # Read the file content
        with open(path, "r") as f:
            data = json.load(f)

        # Parse and validate using Pydantic
        config = EvolutionaryConfig(**data)
        return config

    except json.JSONDecodeError:
        print("Error: The file contains invalid JSON.")
        raise
    except ValidationError as e:
        print("Error: Configuration validation failed.")
        print(e.json())
        raise
