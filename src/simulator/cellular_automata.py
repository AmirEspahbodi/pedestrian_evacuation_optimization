from .ca_state import CAState
from pydantic import BaseModel, ConfigDict


class CellularAutomata(BaseModel):
    """Represents a single cell in the grid."""

    state: CAState
    x: int
    y: int

    model_config = ConfigDict(validate_assignment=True)
