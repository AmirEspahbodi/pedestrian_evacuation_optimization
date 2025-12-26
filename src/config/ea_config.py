import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


# 1. Define the Pydantic Model
class EAConfig(BaseModel):
    offspring: int = Field(..., gt=0, description="Number of offspring generated")
    recombination_prob: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of recombination"
    )
    mutation_gamma: float
    popsize: int = Field(..., gt=0, description="Population size")
    max_evals: int


def load_ea_config(file_path: str) -> EAConfig:
    """Reads a JSON file and validates it against the AlgorithmConfig model."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"The configuration file was not found at: {path}")

    try:
        # Read the file content
        with open(path, "r") as f:
            raw_data = json.load(f)

        # Validate data using Pydantic
        config = EAConfig(**raw_data)
        return config

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
        raise
    except ValidationError as e:
        print("Error: Data validation failed.")
        print(e.json())
        raise
