from pydantic import BaseModel, ConfigDict
from .ca_state import CAState


class CellularAutomata(BaseModel):
    state: CAState
    x: int
    y: int
    static_filed: float = float('inf')

    model_config = ConfigDict(validate_assignment=True)
