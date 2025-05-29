from pydantic import BaseModel
from typing import List, Literal

class Access(BaseModel):
    id: int
    pa: int
    wa: int
