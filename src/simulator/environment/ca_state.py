from enum import StrEnum


class CAState(StrEnum):
    EMPTY = "empty"
    OBSTACLE = "abstacle"
    OCCUPIED = "occupied"
    ACCESS = "access"
    ACCESS_OCCUPIED = "access_occupied"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value
