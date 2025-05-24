from pydantic import BaseModel
from typing import List, Literal


class ShapeAccess(BaseModel):
    type: Literal["rectangle"]
    pa: int
    wa: int


class Access(BaseModel):
    id: int
    shape: ShapeAccess
