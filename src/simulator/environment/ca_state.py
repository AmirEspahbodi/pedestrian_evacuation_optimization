from enum import StrEnum


class CAState(StrEnum):
    EMPTY = "empty"
    OBSTACLE = "abstacle"
    OCCUPIED = "occupied"
    ACCESS = "access"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value
