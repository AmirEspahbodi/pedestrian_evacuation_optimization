from typing import List
from pydantic import BaseModel

class TopLeft(BaseModel):
    x: int
    y: int

class Shape(BaseModel):
    type: str
    topLeft: TopLeft
    width: int
    height: int

class Obstacle(BaseModel):
    name: str
    shape: Shape